"""발전하는 모델 실험 E1~E4 (docs/feud-evolving-model.md). 정답판 = 답한 사람 기준 분포('없음' 제외).

자료: 갤럽 문항-조사 30쌍(2019 12, 2024 반복 12·새 4, 2026 2). 같은 문항 2019/2024는 한 묶음(18묶음) — 은행/평가 나눌 때 같이 움직임.
층0 a: Super 직접 추정 3회를 실측 항목에 배정해 평균(배정 안 된 AI 답은 AI 전용 칸으로 남김). 캐시 data/feud_evolve_cache.json
층1 g_φ: ã ∝ a^τ (τ<1 쏠림 완화), φ=τ를 은행 묶음의 평균 겹침 최대로 고름
층2 θ̂ = (κ·ã + n) / (κ + Σn), κ = 은행 묶음에서 디리클레-다항 주변우도 최대(실측을 답한 사람 수만큼의 개수로 봄)
층3 플레이어 사후층화(성별): 성별 분포가 공개된 3문항으로 기계적 확인

사전 판정(실행 전 기록):
  E1 은행 K = 0,4,8,12,16 묶음 → 남은 묶음 겹침 평균이 K에 따라 감소하지 않고, K=16이 K=0보다 +0.03 이상
  E2 플레이어 n = 0,10,30,100,300,1000 (실측 분포에서 표집, 200회) → 모든 n에서 θ̂ 겹침 ≥ max(AI만, 플레이어만) − 0.01. 교차점 n* 보고
  E3 θ̂의 80% 구간(베타 주변분포) 포함률: n=0, 30에서 70~90%
  E4 플레이어 75% 여성일 때 성별 가중이 무가중보다 n=300에서 겹침 높음
실행: .venv/bin/python -m game.evolve [--refresh]
"""
import argparse
import asyncio
import json
import math

import numpy as np

from game.feud import TARGET, estimate, judge
from game.validate_feud import NONE, TEST, TRAIN
from game.validate_feud2 import NEW, REPEAT, T24, Y26
from llm.router import Router
from world.config import ROOT

CACHE = ROOT / "data" / "feud_evolve_cache.json"
OLD = {**TRAIN, **TEST}
TAUS = np.round(np.arange(0.3, 1.51, 0.1), 2)
KAPPAS = np.unique(np.round(np.logspace(0, 3.5, 36)))
N_ANSWERED = 1100   # 갤럽 1,700명 × 답한 사람 비율 근사(κ 우도용 실측 개수 규모)


def pairs():
    out = []
    for g, (q24, r24, q19) in enumerate(REPEAT):
        out.append(dict(group=g, tag="2019", q=q19, real=OLD[q19], target=TARGET))
        out.append(dict(group=g, tag="2024", q=q24, real=r24, target=T24))
    for i, (q, r) in enumerate(NEW.items()):
        out.append(dict(group=len(REPEAT) + i, tag="2024new", q=q, real=r, target=T24))
    for i, (q, r, t) in enumerate(Y26):
        out.append(dict(group=len(REPEAT) + len(NEW) + i, tag="2026", q=q, real=r, target=t))
    return out


def truth(real):
    """답한 사람 기준 [공개 항목..., 기타]. '없음' 수치가 없으면 0으로 봄."""
    none = real.get(NONE) or 0
    keys = [k for k, v in real.items() if v is not None and k != NONE]
    base = 100 - none
    r = np.array([real[k] / base for k in keys])
    return keys, np.append(r, max(1 - r.sum(), 0))


async def build(router, runs=3):
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    for p in pairs():
        key = f"{p['tag']}|{p['q']}"
        if key in cache:
            continue
        keys, _ = truth(p["real"])
        est = []
        for _ in range(runs):
            shares = await estimate(router, p["q"], p["target"])
            slots = await asyncio.gather(*[judge(router, p["q"], keys, a) for a, _ in shares])
            est.append([[a, pct, int(s)] for (a, pct), s in zip(shares, slots)])
        cache[key] = est
        CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1))
        print(f"  {key[:50]} 추정 {runs}회", flush=True)
    return cache


def prior_vec(est, nk):
    """3회 평균 → [공개 항목 nk개, 기타(실측 쪽, AI 0), AI 전용 칸들...]."""
    a, extra = np.zeros(nk), []
    for run in est:
        for _, pct, s in run:
            if s >= 0:
                a[s] += pct / 100 / len(est)
            else:
                extra.append(pct / 100 / len(est))
    return np.concatenate([a, [0.0], extra])


def overlap(theta, r, nk):
    return float(np.minimum(theta[:nk], r[:nk]).sum() / r[:nk].sum())


def g(a, tau, eps=1e-3):
    x = (a + eps) ** tau
    return x / x.sum()


def dm_loglik(kappa, p, counts):
    a = kappa * p
    return math.lgamma(kappa) - math.lgamma(kappa + counts.sum()) + sum(math.lgamma(x + c) - math.lgamma(x) for x, c in zip(a, counts))


def items(cache):
    out = []
    for p in pairs():
        keys, r = truth(p["real"])
        nk = len(keys)
        a = prior_vec(cache[f"{p['tag']}|{p['q']}"], nk)
        r = np.concatenate([r, np.zeros(len(a) - nk - 1)])
        out.append({**p, "nk": nk, "a": a, "r": r, "keys": keys})
    return out


def fit(bank):
    if not bank:
        return 1.0, 30.0
    tau = max(TAUS, key=lambda t: np.mean([overlap(g(x["a"], t), x["r"], x["nk"]) for x in bank]))
    kappa = max(KAPPAS, key=lambda k: sum(dm_loglik(k, g(x["a"], tau), np.round(x["r"] * N_ANSWERED)) for x in bank))
    return float(tau), float(kappa)


def e1(its, reps=50, rng=None):
    groups = sorted({x["group"] for x in its})
    res = {}
    for K in [0, 4, 8, 12, 16]:
        scores, base = [], []
        for _ in range(reps if K else 1):
            perm = rng.permutation(groups)
            bank_g = set(perm[:K])
            bank = [x for x in its if x["group"] in bank_g]
            tau, _ = fit(bank)
            test = [x for x in its if x["group"] not in bank_g]
            scores.append(np.mean([overlap(g(x["a"], tau), x["r"], x["nk"]) for x in test]))
            base.append(np.mean([overlap(g(x["a"], 1.0), x["r"], x["nk"]) for x in test]))
        res[K] = (float(np.mean(scores)), float(np.mean(base)))
        print(f"  E1 은행 {K:2d}묶음 → 남은 문항 겹침 {res[K][0]:.3f} (같은 문항 보정 없이 {res[K][1]:.3f})")
    ok = all(res[b][0] >= res[a][0] - 0.005 for a, b in zip([0, 4, 8, 12], [4, 8, 12, 16])) and res[16][0] - res[0][0] >= 0.03
    return res, ok


def e2_e3(its, rng, reps=200):
    """한 묶음씩 빼고(leave-one-group-out) 나머지로 τ·κ 적합 → 뺀 묶음에서 플레이어 n명 표집."""
    ns = [0, 10, 30, 100, 300, 1000]
    acc = {n: {"AI": [], "플레이어": [], "합침": []} for n in ns}
    cover = {0: [], 30: []}
    fits = []
    for grp in sorted({x["group"] for x in its}):
        tau, kappa = fit([x for x in its if x["group"] != grp])
        fits.append((tau, kappa))
        for x in [x for x in its if x["group"] == grp]:
            prior, r, nk = g(x["a"], tau), x["r"], x["nk"]
            for n in ns:
                for _ in range(reps if n else 1):
                    c = rng.multinomial(n, r / r.sum()) if n else np.zeros(len(r))
                    post = (kappa * prior + c) / (kappa + n)
                    acc[n]["AI"].append(overlap(prior, r, nk))
                    acc[n]["플레이어"].append(overlap(c / n, r, nk) if n else 0.0)
                    acc[n]["합침"].append(overlap(post, r, nk))
                    if n in cover and _ < 20:
                        al = kappa * prior + c
                        for k in range(nk):
                            if r[k] >= 0.02:
                                lo, hi = beta_q(al[k], al.sum() - al[k], rng)
                                cover[n].append(lo <= r[k] <= hi)
    print(f"  적합값(묶음 빼기 18회): τ 중앙 {np.median([f[0] for f in fits]):.2f}, κ 중앙 {np.median([f[1] for f in fits]):.0f} (= AI 추정의 '실제 응답자 몇 명분')")
    table = {n: {k: float(np.mean(v)) for k, v in acc[n].items()} for n in ns}
    for n in ns:
        t = table[n]
        print(f"  E2 플레이어 {n:4d}명: AI만 {t['AI']:.3f} · 플레이어만 {t['플레이어']:.3f} · 합침 {t['합침']:.3f}")
    ok2 = all(table[n]["합침"] >= max(table[n]["AI"], table[n]["플레이어"]) - 0.01 for n in ns)
    nstar = next((n for n in ns if n and table[n]["플레이어"] >= table[n]["AI"]), None)
    cov = {n: float(np.mean(v)) for n, v in cover.items()}
    print(f"  E3 80% 구간 포함률: n=0 {cov[0]:.0%} · n=30 {cov[30]:.0%}")
    ok3 = all(0.7 <= v <= 0.9 for v in cov.values())
    return table, ok2, nstar, cov, ok3, fits


def beta_q(a, b, rng, m=400):
    s = rng.beta(max(a, 1e-6), max(b, 1e-6), m)
    return np.quantile(s, 0.1), np.quantile(s, 0.9)


# E4: 성별 수치가 공개된 문항(2024). 성별 표에 없는 항목은 '기타'로 묶음
SEXED = {
    "가장 좋아하는 계절은 무엇입니까?": {"여자": {"봄": 48, "가을": 33}, "남자": {"가을": 39, "봄": 33}},
    "직접 하는 운동/스포츠 중에서 가장 즐겨하는 운동은 무엇입니까?": {"남자": {"축구": 19, "골프": 8, "헬스": 8, "걷기": 8}, "여자": {"걷기": 20, "요가": 8, "수영": 6}},
    "가장 좋아하는 과일주스는 무엇입니까?": {"남자": {"오렌지주스": 22, "사과주스": 16, "딸기주스": 13}, "여자": {"딸기주스": 20, "오렌지주스": 17, "사과주스": 14}},
}


def e4(rng, n=300, reps=500, female=0.75):
    res = []
    for q, sx in SEXED.items():
        keys = sorted({k for d in sx.values() for k in d})
        dist = {s: np.append([d.get(k, 0) / 100 for k in keys], 1 - sum(d.values()) / 100) for s, d in sx.items()}
        # 성별 표에 한쪽만 있는 항목은 다른 성별 값 모름 → 두 성별 모두 있는 항목만 채점
        both = [i for i, k in enumerate(keys) if all(k in d for d in sx.values())]
        true = 0.5 * dist["남자"] + 0.5 * dist["여자"]
        raw, ws = [], []
        for _ in range(reps):
            nf = rng.binomial(n, female)
            cf, cm = rng.multinomial(nf, dist["여자"]), rng.multinomial(n - nf, dist["남자"])
            un = (cf + cm) / n
            wt = 0.5 * cf / max(nf, 1) + 0.5 * cm / max(n - nf, 1)
            raw.append(np.abs(un[both] - true[both]).sum()); ws.append(np.abs(wt[both] - true[both]).sum())
        res.append((q[:16], float(np.mean(raw)) * 100, float(np.mean(ws)) * 100))
        print(f"  E4 {q[:20]}: 성별 항목 오차 합 무가중 {res[-1][1]:.1f}%p → 가중 {res[-1][2]:.1f}%p")
    return res, all(w < u for _, u, w in res)


async def main(refresh):
    router = Router(budget_usd=1.0)
    if refresh and CACHE.exists():
        CACHE.unlink()
    cache = await build(router)
    print(f"추정 비용 ${router.total_cost:.3f}")
    rng = np.random.default_rng(0)
    its = items(cache)
    print(f"\n층0 AI 추정(3회 평균) 겹침 평균 {np.mean([overlap(g(x['a'], 1.0), x['r'], x['nk']) for x in its]):.3f} ({len(its)}쌍)")
    r1, ok1 = e1(its, rng=rng)
    t2, ok2, nstar, cov, ok3, fits = e2_e3(its, rng)
    r4, ok4 = e4(rng)
    verdict = {"E1": ok1, "E2": ok2, "E3": ok3, "E4": ok4}
    print("\n판정 " + " · ".join(f"{k} {'✅' if v else '❌'}" for k, v in verdict.items()) + f" · 교차점 n* = {nstar}")
    (ROOT / "data" / "feud_evolve.json").write_text(json.dumps({"E1": r1, "E2": t2, "E3": cov, "E4": r4, "nstar": nstar, "verdict": verdict,
                                                                "fits": fits}, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    asyncio.run(main(ap.parse_args().refresh))
