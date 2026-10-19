"""여론조사 문제 은행으로 새 방식(정답판 = 직접 추정, 프로필·교차표 = 가중 페르소나) 한 판씩 시뮬레이션 + 실측 대조.

대조 ① 정답판: 갤럽 실측에서 '없음'을 빼고 100으로 맞춘 분포와 겹침
대조 ② 집단별: 갤럽 본문에 적힌 성별·지역 수치 vs 가중 페르소나 교차표 (프로필이 믿을 만한지)
실행: .venv/bin/python -m game.sim_bank [--n 200]
"""
import argparse
import asyncio

from game.feud import crosstab, frame, hybrid, judge, profile
from game.validate_feud import NONE
from game.validate_feud2 import NEW, REPEAT, T24, Y26
from game.worldwide import load, sample
from llm.router import Router

R24 = {q: real for q, real, _ in REPEAT} | NEW
R26 = {q: (real, t) for q, real, t in Y26}
# (문항, 실측, 대상, 집단 열, {집단: {답: %}})  집단 수치는 갤럽 본문 그대로(전체 응답자 기준)
BANK = [
    ("가장 좋아하는 계절은 무엇입니까?", R24["가장 좋아하는 계절은 무엇입니까?"], T24, "성별",
     {"여자": {"봄": 48, "가을": 33}, "남자": {"가을": 39, "봄": 33}}),
    ("직접 하는 운동/스포츠 중에서 가장 즐겨하는 운동은 무엇입니까?", R24["직접 하는 운동/스포츠 중에서 가장 즐겨하는 운동은 무엇입니까?"], T24, "성별",
     {"남자": {"축구": 19, "골프": 8, "헬스": 8, "걷기": 8}, "여자": {"걷기": 20, "요가": 8, "수영": 6}}),
    ("가장 좋아하는 과일주스는 무엇입니까?", R24["가장 좋아하는 과일주스는 무엇입니까?"], T24, "성별",
     {"남자": {"오렌지주스": 22, "사과주스": 16, "딸기주스": 13}, "여자": {"딸기주스": 20, "오렌지주스": 17, "사과주스": 14}}),
    ("가장 좋아하는 국내 프로야구팀은 어디입니까?", *R26["가장 좋아하는 국내 프로야구팀은 어디입니까?"], "연고지",
     {"광주/전라": {"KIA 타이거즈": 40}, "부산/울산/경남": {"롯데 자이언츠": 40}, "대구/경북": {"삼성 라이온즈": 40}}),
]
HOME = {"광주": "광주/전라", "전북": "광주/전라", "전라남": "광주/전라", "부산": "부산/울산/경남", "울산": "부산/울산/경남",
        "경상남": "부산/울산/경남", "대구": "대구/경북", "경상북": "대구/경북"}


def overlap(real, ai):
    return sum(min(real[k], ai.get(k, 0)) for k in real) / sum(real.values())


async def one(router, people, q, real, target, col, groups):
    labels, pct, rows = await hybrid(router, people, q, target)
    d = frame(rows)
    d["연고지"] = d.region.map(HOME).fillna("그 외")
    print(f"\n━━━ 한국인에게 물었습니다: {q}")
    # 정답판 칸 ↔ 실측 항목
    known = {k: v for k, v in real.items() if v is not None and k != NONE}
    tot = sum(known.values())
    real_n = {k: 100 * v / tot for k, v in known.items()}   # '없음' 뺀 답한 사람 기준(미공개 항목 제외 근사)
    keys = list(known)
    to_real = await asyncio.gather(*[judge(router, q, keys, l) for l in labels])
    ai = {}
    for l, p, k in zip(labels, pct, to_real):
        if k >= 0:
            ai[keys[k]] = ai.get(keys[k], 0) + p
    print(f"{'순위':<3} {'정답판(AI 추정)':<16} {'AI%':>4} {'실측%':>5}  페르소나 수")
    cnt = d.slot.value_counts()
    for s, (l, p) in enumerate(zip(labels, pct)):
        rk = keys[to_real[s]] if to_real[s] >= 0 else None
        print(f"{s + 1:<4} {l:<16} {p:4.0f} {real_n[rk] if rk else 0:5.0f}  {cnt.get(s, 0)}명{'  ← 프로필 없음' if cnt.get(s, 0) == 0 else '  ← 표본 적음' if cnt.get(s, 0) < 10 else ''}")
    top_real = sorted(real_n, key=lambda k: -real_n[k])[:8]
    missed = [k for k in top_real if k not in ai]
    print(f"정답판 겹침 {overlap(real_n, ai):.2f} · 실측 상위 8 중 정답판에 없는 답: {missed or '없음'}")
    # 1위 칸 공개 화면
    print(f"\n[1위 「{labels[0]}」 공개]\n{profile(d, 0)}")
    # 집단 대조
    print(f"\n[{col}별: 실측(전체 응답자 기준) vs AI 가중 페르소나(답한 사람 기준)]")
    for g, ans in groups.items():
        sub = d[d[col] == g]
        cells = []
        for a, rp in ans.items():
            s = next((i for i, k in enumerate(to_real) if k >= 0 and keys[k] == a), None)
            mine = 100 * sub.w[sub.slot == s].sum() / sub.w.sum() if s is not None and len(sub) else 0
            cells.append(f"{a} {rp}{'+' if col == '연고지' else ''}/{mine:.0f}")
        print(f"  {g:<12} (페르소나 {len(sub)}명) " + " · ".join(cells))
    print("\n[여론조사 결과표(%) — AI 가중]")
    print(crosstab(d, labels, k=4).to_string())
    return overlap(real_n, ai)


async def main(n, seed):
    router = Router(budget_usd=2.0)
    con = load()
    people = sample(con, "Korea", n, seed)
    scores = [await one(router, people, *b) for b in BANK]
    print(f"\n정답판 겹침 평균 {sum(scores) / len(scores):.2f} · 비용 ${router.total_cost:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    asyncio.run(main(a.n, a.seed))
