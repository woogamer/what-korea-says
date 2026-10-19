"""가치관 10문항 → 가장 닮은 실제 WVS 응답자 → 같은 나라·지역·성별·연령대 Nemotron 페르소나 얼굴 → 그 페르소나의 AI 답과 비교.

화면 문구 원칙: 가치관 = 실제 조사 응답자(WVS 7차), 모습 = AI 합성 인물(Nemotron-Personas). 원자료는 재배포 금지라 data/wvs 밖으로 내보내지 않는다.
실행: .venv/bin/python -m game.match [--profile 이름] [--no-ai]
"""
import argparse
import json
import zipfile

import numpy as np
import pandas as pd

from game.values import QUESTIONNAIRE, Answers, ai_system, no_dup_goal
from game.worldwide import load
from game.wvs import ISO, ITEMS, WVS_DIR, Map, code_ai, code_wvs, read_raw, wmean
from llm.router import Router

SEX = {"Korea": ("남자", "여자"), "Japan": ("男", "女"), "USA": ("Male", "Female"), "India": ("Male", "Female"),
       "Brazil": ("Masculino", "Feminino"), "Singapore": ("Male", "Female"), "Vietnam": ("Nam", "Nữ")}
KR = {"Seoul": "서울", "Busan": "부산", "Daegu": "대구", "Incheon": "인천", "Gwangju": "광주", "Daejeon": "대전", "Ulsan": "울산",
      "Gyeonggi": "경기", "Gangwon": "강원", "North Chungcheong": "충청북", "South Chungcheong": "충청남", "North Jeolla": "전북",
      "South Jeolla": "전라남", "North Gyeongsang": "경상북", "South Gyeongsang": "경상남", "Jeju": "제주", "Sejong": "세종"}
JP = ("北海道 青森県 岩手県 宮城県 秋田県 山形県 福島県 茨城県 栃木県 群馬県 埼玉県 千葉県 東京都 神奈川県 新潟県 富山県 石川県 福井県 山梨県 長野県 "
      "岐阜県 静岡県 愛知県 三重県 滋賀県 京都府 大阪府 兵庫県 奈良県 和歌山県 鳥取県 島根県 岡山県 広島県 山口県 徳島県 香川県 愛媛県 高知県 "
      "福岡県 佐賀県 長崎県 熊本県 大分県 宮崎県 鹿児島県 沖縄県").split()
US = dict(zip("Alabama Alaska Arizona Arkansas California Colorado Connecticut Delaware Florida Georgia Hawaii Idaho Illinois Indiana Iowa "
              "Kansas Kentucky Louisiana Maine Maryland Massachusetts Michigan Minnesota Mississippi Missouri Montana Nebraska Nevada "
              "New_Hampshire New_Jersey New_Mexico New_York North_Carolina North_Dakota Ohio Oklahoma Oregon Pennsylvania Rhode_Island "
              "South_Carolina South_Dakota Tennessee Texas Utah Vermont Virginia Washington West_Virginia Wisconsin Wyoming "
              "District_of_Columbia".replace("_", "~").split(),
              "AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD "
              "TN TX UT VT VA WA WV WI WY DC".split()))
US = {k.replace("~", " "): v for k, v in US.items()}
LABEL = {"god": "신의 중요성(높을수록 낮음)", "autonomy": "자율 vs 순종", "abortion": "낙태 허용", "pride": "국가 자부심(높을수록 낮음)",
         "authority": "권위 존중(높을수록 싫음)", "postmat": "탈물질주의", "happy": "행복", "homosexuality": "동성애 허용",
         "petition": "청원 참여", "trust": "타인 신뢰"}


def persona_region(country, wvs_label):
    """WVS 지역 이름 → 페르소나 region 열 값(없으면 None = 나라 전체에서 찾음)."""
    if country == "Korea":
        return KR.get(wvs_label)
    if country == "Japan":
        return None if wvs_label is None else next((p for p in JP if p.startswith(wvs_label)), None)
    if country == "USA":
        return US.get(wvs_label)
    if country in ("India", "Brazil"):
        return wvs_label
    return None   # 싱가포르(5권역 vs 계획구역), 베트남(첫 조각 6개 시·성뿐)은 나라 단위


class Matcher:
    def __init__(self):
        raw = read_raw()
        w = code_wvs(raw)
        self.map = Map().fit(w.groupby("iso").apply(lambda g: wmean(g, ITEMS), include_groups=False))
        regions = {int(k): v for k, v in json.loads((WVS_DIR / "regions.json").read_text()).items()}
        jp = {f"JP-{i + 1:02d}": n for i, n in enumerate(JP)}
        w = w[w.iso.isin(ISO) & w[ITEMS].notna().all(axis=1)].copy()
        w["country"] = w.iso.map(ISO)
        w["region_name"] = w.region.map(lambda r: regions.get(int(r)) if pd.notna(r) else None)
        # 일본은 코드북에 'Tokyo'처럼 영문 → 코드 순서(JIS)로 한자 현명
        w.loc[w.country == "Japan", "region_name"] = w.loc[w.country == "Japan", "region"].map(lambda r: jp.get(f"JP-{int(r) % 1000:02d}"))
        w.loc[w.country.isin(["Singapore", "Vietnam"]), "region_name"] = None   # 페르소나 지역과 대응 안 됨 → 나라만 표시
        w[["secular", "selfexp"]] = self.map.project(w)
        self.full = w.reset_index(drop=True)
        # 나라별 인원이 다르면 큰 나라가 더 자주 '최근접'이 됨 → 가장 작은 나라 크기로 나라마다 무작위 추출
        n = w.country.value_counts().min()
        w = w.groupby("country").sample(n, random_state=0)
        self.resp = w.reset_index(drop=True)
        self.mu, self.sd = w[ITEMS].mean(), w[ITEMS].std()
        self.con = load()

    def nearest(self, user_items, k=5, seed=0):
        Z = ((self.resp[ITEMS] - self.mu) / self.sd).values
        u = ((user_items[ITEMS] - self.mu) / self.sd).values.astype(float)
        d = np.sqrt(np.nanmean((Z - u) ** 2, axis=1))
        top = np.argsort(d)[:k]
        i = top[np.random.default_rng(seed).integers(len(top))] if k > 1 else top[0]
        return self.resp.iloc[i], float(d[i]), self.resp.iloc[top]

    def persona(self, r, seed=0):
        c = r.country
        sex = SEX[c][0] if r.sex == 1 else SEX[c][1]
        reg = persona_region(c, r.region_name if isinstance(r.region_name, str) else None)
        age = int(r.age) if pd.notna(r.age) else 40
        for band, use_reg in [(3, True), (7, True), (7, False), (15, False)]:
            q = "SELECT * FROM people WHERE country=? AND sex=? AND age BETWEEN ? AND ?"
            args = [c, sex, age - band, age + band]
            if use_reg and reg:
                q += " AND region=?"; args.append(reg)
            rows = self.con.execute(q + " ORDER BY hash(uuid || ?) LIMIT 1", args + [str(seed)]).df()
            if len(rows):
                return rows.iloc[0].to_dict(), {"band": band, "region_matched": bool(use_reg and reg)}
        raise LookupError(f"no persona for {c} {sex} {age}")


def ask_ai(p):
    router = Router(budget_usd=0.2)
    a, _ = router.call("respondent", [{"role": "system", "content": ai_system(p)}, {"role": "user", "content": QUESTIONNAIRE}],
                       schema=Answers, validator=no_dup_goal, thinking=True, temperature=0.9, max_tokens=4000, retries=2, tag="match-ai")
    return a


# 테스트용 가짜 사용자 3명(내가 사용자 역할)
PROFILES = {
    "seoul_liberal": dict(god=2, child_qualities=["independence", "imagination", "tolerance and respect for other people", "feeling of responsibility"],
                          abortion=8, pride="not very proud", authority="bad thing", goal_first="protecting freedom of speech",
                          goal_second="giving people more say in important government decisions", happy="rather happy", homosexuality=8,
                          petition="have done", trust="need to be very careful"),
    "devout_traditional": dict(god=10, child_qualities=["religious faith", "obedience", "good manners", "hard work"], abortion=1,
                               pride="very proud", authority="good thing", goal_first="maintaining order in the nation",
                               goal_second="fighting rising prices", happy="very happy", homosexuality=1, petition="would never do",
                               trust="need to be very careful"),
    "middle": dict(god=5, child_qualities=["good manners", "hard work", "feeling of responsibility", "thrift, saving money and things"],
                   abortion=5, pride="quite proud", authority="good thing", goal_first="fighting rising prices",
                   goal_second="maintaining order in the nation", happy="rather happy", homosexuality=4, petition="might do",
                   trust="need to be very careful"),
}


def run(m, name, answers, use_ai=True, seed=0):
    a = Answers(**answers)
    u = code_ai(pd.DataFrame([{"country": "you", **a.model_dump()}]))
    pos = m.map.project(u).iloc[0]
    r, dist, top = m.nearest(u, seed=seed)
    p, how = m.persona(r, seed=seed)
    sex = "남성" if r.sex == 1 else "여성"
    print(f"\n=== {name} ===  내 위치: 세속 {pos.secular:+.2f} · 자기표현 {pos.selfexp:+.2f}")
    print(f"당신은 {r.country}{' ' + r.region_name if isinstance(r.region_name, str) else ''}에 사는 {int(r.age) if pd.notna(r.age) else '?'}세 {sex}과 가장 닮았습니다 (거리 {dist:.2f}, 상위5 나라 {list(top.country)})")
    print(f"  모습(AI 합성): {p['age']}세 · {p['occupation']} · {p['place']}, {p['region']}  [연령±{how['band']}, 지역일치 {how['region_matched']}]")
    print(f"  {p['persona'][:140]}…")
    rows = {"나": u.iloc[0][ITEMS], "실제 응답자": r[ITEMS]}
    if use_ai:
        ai = ask_ai(p)
        if ai:
            rows["그 사람의 AI"] = code_ai(pd.DataFrame([{"country": r.country, **ai.model_dump()}])).iloc[0][ITEMS]
    t = pd.DataFrame(rows).astype(float).round(2)
    t.index = [LABEL[i] for i in t.index]
    print(t.to_string())
    if "그 사람의 AI" in t:
        gap = ((t["그 사람의 AI"] - t["실제 응답자"]).abs() / m.sd.values).round(2)
        print(f"  AI가 실제 응답자와 가장 다른 항목: {list(gap.sort_values(ascending=False).index[:3])}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="all")
    ap.add_argument("--no-ai", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    m = Matcher()
    print("응답자 풀:", m.resp.country.value_counts().to_dict())
    for n in PROFILES if a.profile == "all" else [a.profile]:
        run(m, n, PROFILES[n], use_ai=not a.no_ai, seed=a.seed)
