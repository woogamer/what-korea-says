"""검증 시나리오 V7·V8 (docs/scenario-seoul-ev.md §10). LLM 호출 없음.

  합성 쪽 = §7 E0 셀별 원시 시뮬 분포(data/calibration.json) 고정. 실측 쪽 = 서울서베이 A에서 n명 추출.
  V7: n별 200회, 방법 D R M S BP BS 를 B 셀 분포로 채점. V8: 셀 하나를 A에서 통째로 빼고 R M S_loco 비교.
  .venv/bin/python -m scripts.validate_learning   → data/validation_learning.json
"""
import json
import sys
import time
from datetime import datetime
from math import lgamma
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from evidence.seoul_survey import load  # noqa: E402
from scripts.validate_seoul import BANDS, housing_group  # noqa: E402

KEYS = ["ev", "hybrid", "hydrogen", "none"]
HG = ["비아파트", "아파트"]
LABELS = [f"{b} × {h}" for b in BANDS for h in HG]          # 셀 c = 연령 a·2 + 주거 h
NS = [25, 50, 100, 175, 400, 875, 1750, 3500, 5245]
REPS = 200
LAM = 0.4
ALPHAS = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000], float)
SMOOTH = 2.0                                                 # R: 셀마다 표본 전체 비율대로 가산하는 건수
LG = np.frompyfunc(lgamma, 1, 1)


def table(cell, out, w, n_cells=12):
    """가중 건수 W[c,k]와 셀 표본 수 n[c]."""
    W = np.zeros((n_cells, 4)); np.add.at(W, (cell, out), w)
    n = np.bincount(cell, minlength=n_cells).astype(float)
    return W, n


def norm(x):
    s = x.sum(-1, keepdims=True)
    return np.divide(x, s, out=np.zeros_like(x), where=s > 0)


def direct(W, m):
    P = norm(W)
    P[W.sum(1) == 0] = m                                     # 빈 셀은 표본 전체 비율
    return P


def synth_blend(sim, W, m, lam=LAM):
    """비율 보정(시뮬 주변분포는 표본 셀 구성으로 가중) 후 표본 주변분포와 혼합 = 현 제품 보정."""
    pi = W.sum(1) / W.sum()
    sim_m = pi @ sim
    w = m / np.maximum(sim_m, 1e-3)
    return lam * norm(sim * w) + (1 - lam) * m


def regression(W, n, m, iters=60):
    """연령 + 주거 주효과 로그선형 모형(IPF). 관측 표 = 유효 건수 + 셀당 SMOOTH 건(전체 비율)."""
    N = (n[:, None] * norm(W) + SMOOTH * m).reshape(6, 2, 4)
    F = np.ones_like(N)
    r = lambda a, b: np.divide(a, b, out=np.zeros_like(a), where=b > 0)  # noqa: E731  표본에 없는 선택지(0/0)는 0
    for _ in range(iters):
        F *= r(N.sum(2), F.sum(2))[:, :, None]
        F *= r(N.sum(1), F.sum(1))[:, None, :]
        F *= r(N.sum(0), F.sum(0))[None, :, :]
    return norm(F.reshape(12, 4))


def bayes(prior, W, n):
    """셀별 디리클레(α·prior) 사전, α는 디리클레-다항 주변우도 최대(경험적 베이즈). 사후 평균."""
    x = n[:, None] * norm(W)                                 # 유효 건수
    best, best_ll = ALPHAS[0], -np.inf
    for a in ALPHAS:
        ap = a * prior
        ok = ap > 0
        ll = float(np.sum(LG(a) - LG(a + n)))
        ll += float(np.sum(np.where(ok, LG(np.where(ok, ap + x, 1.0)) - LG(np.where(ok, ap, 1.0)), 0.0)))
        if ll > best_ll:
            best, best_ll = a, ll
    return (best * prior + x) / (best + n[:, None]), best, best_ll


BETAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def bayes_mix(S, R, W, n):
    """§10.5 BSR: 사전 평균 = β·S + (1−β)·R, (β, α)를 주변우도로 함께 선택."""
    fits = [(bayes(b * S + (1 - b) * R, W, n), b) for b in BETAS]
    (post, a, _), b = max(fits, key=lambda f: f[0][2])
    return post, a, b


def neighbor_band(W, band):
    """§10.5 V8b 기준선 ①: 같은 주거 유형의 인접 연령대 셀 직접 추정(양옆이 있으면 평균)."""
    P = norm(W).reshape(6, 2, 4)
    nb = [b for b in (band - 1, band + 1) if 0 <= b < 6]
    return P[nb].mean(0)


def linear_age_logit(W, n, steps=4000, lr=0.5):
    """§10.5 V8b 기준선 ②: logit_k = b_k + g_k·연령순번 + h_k·아파트 (기준 범주 none), 가중 유효 건수로 경사상승."""
    x = (n[:, None] * norm(W))
    age = np.repeat(np.arange(6), 2) / 5.0; apt = np.tile([0.0, 1.0], 6)
    X = np.stack([np.ones(12), age, apt], 1)                 # 12×3
    theta = np.zeros((3, 4))
    tot = x.sum()
    for _ in range(steps):
        Z = X @ theta; Z[:, 3] = 0
        P = np.exp(Z - Z.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
        grad = X.T @ (x - x.sum(1, keepdims=True) * P) / tot
        grad[:, 3] = 0
        theta += lr * 10 * grad
    Z = X @ theta; Z[:, 3] = 0
    P = np.exp(Z - Z.max(1, keepdims=True))
    return P / P.sum(1, keepdims=True)


def score(P, B, Bn):
    return float(np.sum(0.5 * np.abs(P - B).sum(1) * Bn) / Bn.sum())


def main():
    t0 = time.time()
    df = load()
    df = df.dropna(subset=["ev_plan", "age_band", "housing_type"])
    df["c"] = df.age_band.map(BANDS.index) * 2 + df.housing_type.map(housing_group).map(HG.index)
    df["k"] = df.ev_plan.map(KEYS.index)
    A, Bdf = df[df.split == "A"], df[df.split == "B"]
    WB, nB = table(Bdf.c.values, Bdf.k.values, Bdf.w.values)
    B = norm(WB)
    cal = json.load(open(ROOT / "data/calibration.json"))["conditions"]["E0 단일 질문"]["cells"]
    sim = np.array([[cal[lab]["raw"][k] for k in KEYS] for lab in LABELS])
    Ac, Ak, Aw = A.c.values, A.k.values, A.w.values
    WA, nA = table(Ac, Ak, Aw)
    floor = score(norm(WA), B, nB)
    print(f"A {len(A)} · B {len(Bdf)} · A 전체 직접 {floor:.4f} · 원시 시뮬 {score(sim, B, nB):.4f}", flush=True)

    # V7 학습곡선
    rng = np.random.default_rng(7)
    methods = ["D", "R", "M", "S", "BP", "BS", "BR", "BSR"]   # BR = 사후 탐색(§10.3), BSR = §10.5 사전 등록
    V7 = {}
    for n_s in NS:
        res = {k: [] for k in methods}; alph = {"BP": [], "BS": [], "BR": [], "BSR": []}; betas = []
        for _ in range(REPS):
            idx = rng.choice(len(A), n_s, replace=False)
            W, n = table(Ac[idx], Ak[idx], Aw[idx])
            m = W.sum(0) / W.sum()
            M = np.tile(m, (12, 1)); S = synth_blend(sim, W, m)
            R = regression(W, n, m)
            bp, a_bp, _ = bayes(M, W, n); bs, a_bs, _ = bayes(S, W, n); br, a_br, _ = bayes(R, W, n)
            bsr, a_bsr, beta = bayes_mix(S, R, W, n)
            for k, P in zip(methods, [direct(W, m), R, M, S, bp, bs, br, bsr]):
                res[k].append(score(P, B, nB))
            alph["BP"].append(a_bp); alph["BS"].append(a_bs); alph["BR"].append(a_br); alph["BSR"].append(a_bsr); betas.append(beta)
        D = np.array(res["D"])
        V7[n_s] = {k: {"mean": round(float(np.mean(v)), 4), "p2.5": round(float(np.percentile(v, 2.5)), 4), "p97.5": round(float(np.percentile(v, 97.5)), 4),
                       "win_vs_D": round(float(np.mean(np.array(v) < D)), 3)} for k, v in res.items()}
        V7[n_s]["BS_win_vs_BR"] = round(float(np.mean(np.array(res["BS"]) < np.array(res["BR"]))), 3)
        V7[n_s]["alpha_median"] = {k: float(np.median(v)) for k, v in alph.items()}
        V7[n_s]["BSR_beta_mean"] = round(float(np.mean(betas)), 3)
        V7[n_s]["BSR_win_vs_best_real"] = round(float(np.mean(np.array(res["BSR"]) <= np.min([res[k] for k in ["D", "R", "M", "BP", "BR"]], 0))), 3)
        print(n_s, " ".join(f"{k}={V7[n_s][k]['mean']:.4f}" for k in methods), V7[n_s]["alpha_median"], "β", V7[n_s]["BSR_beta_mean"], flush=True)

    # V8 조사 안 한 고객층: 셀 c를 A에서 통째로 제외
    V8 = {"cells": {}}
    pred = {k: np.zeros((12, 4)) for k in ["R", "M", "S_loco", "ratio_loco"]}
    for c in range(12):
        keep = Ac != c
        W, n = table(Ac[keep], Ak[keep], Aw[keep])
        m = W.sum(0) / W.sum()
        pred["R"][c] = regression(W, n, m)[c]
        pred["M"][c] = m
        pred["S_loco"][c] = synth_blend(sim, W, m)[c]
        pred["ratio_loco"][c] = synth_blend(sim, W, m, lam=1.0)[c]
        V8["cells"][LABELS[c]] = {"n_B": int(nB[c]), **{k: round(0.5 * float(np.abs(P[c] - B[c]).sum()), 4) for k, P in pred.items()},
                                  "ev": {"B": round(float(B[c, 0]), 4), **{k: round(float(P[c, 0]), 4) for k, P in pred.items()}}}
    V8["weighted"] = {k: round(score(P, B, nB), 4) for k, P in pred.items()}
    V8["wins_S_vs_R"] = sum(V8["cells"][lab]["S_loco"] < V8["cells"][lab]["R"] for lab in LABELS)
    print("V8", V8["weighted"], "S<R", V8["wins_S_vs_R"], "/12", flush=True)

    # V8b 연령대 통째 제외 (§10.5)
    V8b = {"bands": {}}
    predb = {k: np.zeros((12, 4)) for k in ["neighbor", "linear_logit", "M", "S_loco", "ratio_loco"]}
    for band in range(6):
        cs = [band * 2, band * 2 + 1]
        keep = ~np.isin(Ac, cs)
        W, n = table(Ac[keep], Ak[keep], Aw[keep])
        m = W.sum(0) / W.sum()
        nbp, ll = neighbor_band(W, band), linear_age_logit(W, n)
        for j, c in enumerate(cs):
            predb["neighbor"][c] = nbp[j]; predb["linear_logit"][c] = ll[c]; predb["M"][c] = m
            predb["S_loco"][c] = synth_blend(sim, W, m)[c]; predb["ratio_loco"][c] = synth_blend(sim, W, m, lam=1.0)[c]
        sub = np.zeros(12, bool); sub[cs] = True
        V8b["bands"][BANDS[band]] = {k: round(score(P[sub], B[sub], nB[sub]), 4) for k, P in predb.items()} | {
            "ev": {"B": [round(float(B[c, 0]), 3) for c in cs], **{k: [round(float(P[c, 0]), 3) for c in cs] for k, P in predb.items()}}}
    V8b["weighted"] = {k: round(score(P, B, nB), 4) for k, P in predb.items()}
    print("V8b", V8b["weighted"], flush=True)

    # V8c 빈 셀 + 적은 실측 (§10.5)
    V8c = {}
    rng_c = np.random.default_rng(8)
    others = {c: np.flatnonzero(Ac != c) for c in range(12)}
    for n_s in [175, 400, 875]:
        res = {"R": [], "S_loco": [], "M": []}
        for _ in range(REPS):
            P = {k: np.zeros((12, 4)) for k in res}
            for c in range(12):
                idx = rng_c.choice(others[c], n_s, replace=False)
                W, n = table(Ac[idx], Ak[idx], Aw[idx])
                m = W.sum(0) / W.sum()
                P["R"][c] = regression(W, n, m)[c]; P["S_loco"][c] = synth_blend(sim, W, m)[c]; P["M"][c] = m
            for k in res:
                res[k].append(score(P[k], B, nB))
        V8c[n_s] = {k: round(float(np.mean(v)), 4) for k, v in res.items()} | {"S_win_vs_R": round(float(np.mean(np.array(res["S_loco"]) < np.array(res["R"]))), 3)}
        print("V8c", n_s, V8c[n_s], flush=True)

    # 판정
    m_ = lambda n_s, k: V7[n_s][k]["mean"]  # noqa: E731
    cross = next((n_s for n_s in NS if m_(n_s, "D") <= m_(n_s, "S")), None)
    small = [n_s for n_s in NS if n_s <= 175]
    verdict = {
        "V7a": {"crossover_n": cross, "pass": cross is not None and cross <= 400},
        "V7b": {"worst_gap_BS_minus_D": round(max(m_(n_s, "BS") - m_(n_s, "D") for n_s in NS), 4), "pass": all(m_(n_s, "BS") <= m_(n_s, "D") + 0.005 for n_s in NS)},
        "V7c": {"gaps_BS_minus_BP": [round(m_(n_s, "BS") - m_(n_s, "BP"), 4) for n_s in small], "pass": all(m_(n_s, "BS") < m_(n_s, "BP") for n_s in small)},
        "V7d": {"gaps_BS_minus_R": [round(m_(n_s, "BS") - m_(n_s, "R"), 4) for n_s in small], "pass": all(m_(n_s, "BS") < m_(n_s, "R") for n_s in small)},
        "V7e": {"gap_at_max": round(abs(m_(NS[-1], "BS") - m_(NS[-1], "D")), 4), "pass": abs(m_(NS[-1], "BS") - m_(NS[-1], "D")) < 0.005},
        "V8": {"S_loco": V8["weighted"]["S_loco"], "R": V8["weighted"]["R"], "pass": V8["weighted"]["S_loco"] < V8["weighted"]["R"]},
        "V7'a": {"worst_gap_BSR_minus_best_real": round(max(m_(n_s, "BSR") - min(m_(n_s, k) for k in ["D", "R", "M", "BP", "BR"]) for n_s in NS), 4),
                 "pass": all(m_(n_s, "BSR") <= min(m_(n_s, k) for k in ["D", "R", "M", "BP", "BR"]) + 0.002 for n_s in NS)},
        "V7'b": {"gaps_BSR_minus_BR": [round(m_(n_s, "BSR") - m_(n_s, "BR"), 4) for n_s in NS if n_s <= 400], "pass": all(m_(n_s, "BSR") < m_(n_s, "BR") for n_s in NS if n_s <= 400)},
        "V8b": {**V8b["weighted"], "pass": V8b["weighted"]["S_loco"] < min(V8b["weighted"]["neighbor"], V8b["weighted"]["linear_logit"])},
        "V8c": {"n175": V8c[175], "pass": V8c[175]["S_loco"] < V8c[175]["R"]},
    }
    for k, v in verdict.items():
        print(k, "✅" if v["pass"] else "❌", v)
    out = {"ts": datetime.now().isoformat(timespec="seconds"), "seconds": round(time.time() - t0), "cost_usd": 0.0, "reps": REPS, "lam": LAM,
           "floor_A_direct": round(floor, 4), "raw_sim": round(score(sim, B, nB), 4), "V7": {str(k): v for k, v in V7.items()}, "V8": V8, "V8b": V8b, "V8c": {str(k): v for k, v in V8c.items()}, "verdict": verdict}
    (ROOT / "data/validation_learning.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"done {out['seconds']}s")


if __name__ == "__main__":
    main()
