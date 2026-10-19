"""WVS 7차 원자료 → 문화 지도 2축(전통↔세속, 생존↔자기표현) + AI 시민 투영 + 가장 닮은 응답자 매칭.

원자료는 data/wvs/ (재배포 금지 조건 → git 제외). 축은 잉글하트-웰젤 10문항을 표준화해
WVS 전체 국가 개인 단위 PCA(2성분, varimax)로 만든다(공식 지도와 같은 방식의 근사).
실행: .venv/bin/python -m game.wvs  → data/wvs_map.json, docs 그림용 표 출력
"""
import glob
import json
import zipfile

import numpy as np
import pandas as pd

from world.config import ROOT

WVS_DIR = ROOT / "data" / "wvs"
ISO = {"KOR": "Korea", "JPN": "Japan", "USA": "USA", "IND": "India", "BRA": "Brazil", "SGP": "Singapore", "VNM": "Vietnam", "FRA": "France"}
# 전통↔세속 (높을수록 세속), 생존↔자기표현 (높을수록 자기표현) 방향으로 코딩
ITEMS = ["god", "autonomy", "abortion", "pride", "authority", "postmat", "happy", "homosexuality", "petition", "trust"]
NEED = ["B_COUNTRY_ALPHA", "Q164", "Q8", "Q14", "Q15", "Q17", "Q184", "Q254", "Q45", "Q154", "Q155",
        "Q46", "Q182", "Q209", "Q57", "Q260", "Q262", "N_REGION_ISO", "W_WEIGHT"]


def read_raw():
    files = sorted(glob.glob(str(WVS_DIR / "*")))
    for f in files:
        if f.endswith(".zip"):
            with zipfile.ZipFile(f) as z:
                csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
                if csvs and "wvs" in csvs[0].lower():
                    with z.open(csvs[0]) as fh:
                        return pd.read_csv(fh, low_memory=False)
        if f.endswith(".csv") and "wvs" in f.lower():
            return pd.read_csv(f, low_memory=False)
    raise FileNotFoundError(f"WVS csv not found in {WVS_DIR}: {files}")


def postmat(first, second):
    """1 물질주의(질서·물가 둘 다) / 2 혼합 / 3 탈물질주의(발언권·표현 둘 다). WVS 코드 1 질서 2 발언권 3 물가 4 표현."""
    a, b = np.isin(np.asarray(first), [1, 3]), np.isin(np.asarray(second), [1, 3])
    return np.where(a & b, 1, np.where(~a & ~b, 3, 2)).astype(float)


def code_wvs(df):
    miss = [c for c in NEED if c not in df.columns]
    if miss:
        raise KeyError(f"WVS 변수 없음: {miss}")
    d = df[NEED].copy()
    d = d.mask(d.select_dtypes("number") < 0)   # 음수 = 무응답·모름
    d.loc[d.Q254 == 5, "Q254"] = np.nan          # 5 = 자국민 아님
    out = pd.DataFrame({
        "iso": df["B_COUNTRY_ALPHA"],
        "god": 11 - d.Q164,                                   # 높을수록 세속
        "autonomy": (d.Q8 == 1).astype(float) + (d.Q14 == 1) - (d.Q15 == 1) - (d.Q17 == 1),
        "abortion": d.Q184,
        "pride": d.Q254,                                      # 1 매우 자랑 … 4 전혀 → 높을수록 세속
        "authority": d.Q45,                                   # 1 좋다 … 3 나쁘다
        "postmat": postmat(d.Q154, d.Q155),
        "happy": 5 - d.Q46,                                   # 높을수록 행복
        "homosexuality": d.Q182,
        "petition": 4 - d.Q209,                               # 3 해봤다
        "trust": (d.Q57 == 1).astype(float),
        "sex": d.Q260, "age": d.Q262, "region": df["N_REGION_ISO"], "w": d.W_WEIGHT.fillna(1.0),
    })
    out.loc[d.Q8.isna() | d.Q17.isna(), "autonomy"] = np.nan
    out.loc[d.Q154.isna() | d.Q155.isna(), "postmat"] = np.nan
    out.loc[d.Q57.isna(), "trust"] = np.nan
    return out


def code_ai(ai):
    q = ai.child_qualities
    g = {"maintaining order in the nation": 1, "giving people more say in important government decisions": 2,
         "fighting rising prices": 3, "protecting freedom of speech": 4}
    return pd.DataFrame({
        "country": ai.country,
        "god": 11 - ai.god,
        "autonomy": q.apply(lambda x: ("independence" in x) + ("determination, perseverance" in x) - ("religious faith" in x) - ("obedience" in x)),
        "abortion": ai.abortion,
        "pride": ai.pride.map({"very proud": 1, "quite proud": 2, "not very proud": 3, "not at all proud": 4}),
        "authority": ai.authority.map({"good thing": 1, "don't mind": 2, "bad thing": 3}),
        "postmat": postmat(ai.goal_first.map(g).values, ai.goal_second.map(g).values),
        "happy": ai.happy.map({"very happy": 4, "rather happy": 3, "not very happy": 2, "not at all happy": 1}),
        "homosexuality": ai.homosexuality,
        "petition": ai.petition.map({"have done": 3, "might do": 2, "would never do": 1}),
        "trust": (ai.trust == "most people can be trusted").astype(float),
    })


def varimax(L, iters=100):
    p, k = L.shape; R = np.eye(k)
    for _ in range(iters):
        Lr = L @ R
        u, _, vt = np.linalg.svd(L.T @ (Lr ** 3 - Lr @ np.diag((Lr ** 2).sum(0)) / p))
        R = u @ vt
    return L @ R, R


class Map:
    """공식 지도처럼 나라 평균으로 적합(생태학적 PCA·varimax) → 실측 개인·AI·사용자 모두 같은 변환."""

    def fit(self, X):
        X = X[ITEMS].dropna()
        self.mu, self.sd = X.mean(), X.std()
        Z = ((X - self.mu) / self.sd).values
        evals, evecs = np.linalg.eigh(np.cov(Z.T))
        top = evecs[:, ::-1][:, :2] * np.sqrt(evals[::-1][:2])
        L, self.R = varimax(top)
        # 축 이름 붙이기: god 적재가 큰 쪽 = 전통↔세속, 부호는 항목 방향(높을수록 세속/자기표현)에 맞춤
        trad = int(np.argmax(np.abs(L[ITEMS.index("god")])))
        self.order = [trad, 1 - trad]
        self.sign = [np.sign(L[ITEMS.index("god"), trad]), np.sign(L[ITEMS.index("homosexuality"), 1 - trad])]
        self.W = np.linalg.pinv(L).T   # 회귀 점수 근사
        self.L = L
        return self

    def project(self, X):
        Z = ((X[ITEMS] - self.mu) / self.sd).fillna(0).values
        S = Z @ self.W
        return pd.DataFrame({"secular": S[:, self.order[0]] * self.sign[0], "selfexp": S[:, self.order[1]] * self.sign[1]}, index=X.index)

    def loadings(self):
        return pd.DataFrame(self.L[:, self.order] * self.sign, index=ITEMS, columns=["secular", "selfexp"]).round(2)


def wmean(df, cols, w="w"):
    """열마다 그 열이 응답된 행의 가중치로 나눔(무응답 많은 나라 부풀림 방지)."""
    return df[cols].mul(df[w], axis=0).sum() / df[cols].notna().mul(df[w], axis=0).sum()


def main():
    raw = read_raw()
    wvs = code_wvs(raw)
    real_items = wvs.groupby("iso").apply(lambda g: wmean(g, ITEMS), include_groups=False)
    m = Map().fit(real_items)
    print("적재(축 방향 확인):\n", m.loadings())
    wvs[["secular", "selfexp"]] = m.project(wvs)
    real = m.project(real_items)
    ai = code_ai(pd.read_parquet(ROOT / "data" / "values_ai.parquet"))
    ai[["secular", "selfexp"]] = m.project(ai)
    ai_c = m.project(ai.groupby("country")[ITEMS].mean())
    # AI 나라 평균 위치의 95% 부트스트랩 구간(AI 시민 재표본 1,000회). 실측은 1,000명+라 오차 무시
    rng = np.random.default_rng(0)
    boot = {c: m.project(pd.DataFrame([g[ITEMS].iloc[rng.integers(len(g), size=len(g))].mean() for _ in range(1000)]))
            for c, g in ai.groupby("country")}
    rows = []
    for iso, c in ISO.items():
        if iso in real.index and c in ai_c.index:
            r, a = real.loc[iso], ai_c.loc[c]
            rows.append({"country": c, "real_secular": r.secular, "ai_secular": a.secular, "real_selfexp": r.selfexp,
                         "ai_selfexp": a.selfexp, "shift": float(np.hypot(a.secular - r.secular, a.selfexp - r.selfexp)),
                         "shift_lo": float(np.percentile(np.hypot(boot[c].secular - r.secular, boot[c].selfexp - r.selfexp), 2.5)),
                         "ai_secular_ci": [round(float(x), 2) for x in np.percentile(boot[c].secular, [2.5, 97.5])],
                         "n_ai": int((ai.country == c).sum())})
    comp = pd.DataFrame(rows).set_index("country").round(2)
    print(comp)
    # 항목별 실측 vs AI 평균(같은 코딩)
    items_real = real_items[real_items.index.isin(ISO)].rename(index=ISO)
    items_ai = ai.groupby("country")[ITEMS].mean()
    print("실측 항목 평균\n", items_real.round(2), "\nAI 항목 평균\n", items_ai.round(2))
    # 항목별 AI−실측 차이 중 95% 부트스트랩 구간이 0을 넘지 않는 것만(실측 sd 단위)
    sd = wvs[ITEMS].std()
    sig = {}
    for c, g in ai.groupby("country"):
        if c not in items_real.index:
            continue
        bm = np.array([g[ITEMS].iloc[rng.integers(len(g), size=len(g))].mean().values for _ in range(1000)])
        lo, hi = np.percentile((bm - items_real.loc[c].values) / sd.values, [2.5, 97.5], axis=0)
        sig[c] = {i: round(float((l + h) / 2), 2) for i, l, h in zip(ITEMS, lo, hi) if l > 0 or h < 0}
    print("유의한 항목 차이(AI−실측, sd 단위)\n", json.dumps(sig, ensure_ascii=False, indent=1))
    # 나라 안 다양성: 개인 점수 표준편차
    spread = pd.DataFrame({"real_sd": wvs[wvs.iso.isin(ISO)].groupby("iso")[["secular", "selfexp"]].std().rename(index=ISO).mean(axis=1),
                           "ai_sd": ai.groupby("country")[["secular", "selfexp"]].std().mean(axis=1)}).round(2)
    print("나라 안 퍼짐\n", spread)
    (ROOT / "data" / "wvs_map.json").write_text(json.dumps({
        "real_all": real.round(3).reset_index().to_dict("records"), "comparison": comp.reset_index().to_dict("records"),
        "loadings": m.loadings().to_dict(), "spread": spread.reset_index().to_dict("records"), "item_gaps": sig}, ensure_ascii=False, indent=1))
    return m, wvs, ai


if __name__ == "__main__":
    main()
