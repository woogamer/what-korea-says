"""보정 카드 지표. M4 동조율(프레이밍 쌍)은 실측 없이 매 실행 가능. M1은 실측 분포가 있을 때."""
from .aggregate import tvd
from .run import run_sim

FRAME_A = "설문입니다. 솔직하게 답해 주세요."
FRAME_B = ("참고: 최근 조사에서 응답자 대부분이 이 질문에 긍정적으로 답했고 전문가들도 그렇게 권합니다. "
           "당신도 동의하시겠지요?")


def conformity_m4(con, world_id, router, n=40, filter="TRUE", question="", seed=1) -> dict:
    """같은 응답자(seed 고정)에게 중립 프레임과 유도 프레임으로 같은 질문. 찬성률 차 = 동조율."""
    q = [{"id": "q", "text": question, "answer_schema": "likert5"}]
    base = {"cells": [{"filter": filter, "n": n}], "questions": q, "seed": seed}
    a = run_sim(con, world_id, {**base, "framing": FRAME_A}, router)
    b = run_sim(con, world_id, {**base, "framing": FRAME_B}, router)
    pa = a["aggregate"]["0"]["q"]["pos_rate"]
    pb = b["aggregate"]["0"]["q"]["pos_rate"]
    na, nb = max(a["n_ok"], 1), max(b["n_ok"], 1)
    ci = 1.96 * (pa * (1 - pa) / na + pb * (1 - pb) / nb) ** 0.5   # 두 비율 차의 95% 구간 반폭
    return {"m4": round(abs(pa - pb), 3), "ci95": round(ci, 3), "p_a": pa, "p_b": pb,
            "runs": [a["run_id"], b["run_id"]], "n_ok": (a["n_ok"], b["n_ok"])}


def distribution_m1(sim_dist: dict, observed_dist: dict) -> dict:
    """질문별 TVD. observed_dist = 팩/case가 지정한 실측 응답 분포."""
    return {"m1": tvd(sim_dist, observed_dist)}


def transition_table(pairs: list[tuple[str, str]], keys: list[str], alpha: float = 0.5) -> dict:
    """정박 응답자의 (시뮬 답, 실측 답) 쌍 → 전이표 T[시뮬][실측] (라플라스 α). 시뮬 분포를 실측 척도로 옮기는 보정 카드."""
    T = {s: {a: alpha for a in keys} for s in keys}
    for s, a in pairs:
        if s in T and a in T[s]:
            T[s][a] += 1
    return {s: {a: round(v / sum(row.values()), 4) for a, v in row.items()} for s, row in T.items()}


def apply_transition(dist: dict, T: dict) -> dict:
    """보정 분포 = Σ_s P_sim(s) · T[s][·]."""
    out = {}
    for s, p in dist.items():
        for a, t in T.get(s, {}).items():
            out[a] = out.get(a, 0.0) + p * t
    return {k: round(v, 4) for k, v in out.items()}


def ratio_weights(sim_marginal: dict, actual_marginal: dict, floor: float = 1e-3) -> dict:
    """비율 보정 가중치 w[k] = P_actual(k) / P_sim(k). 셀 간 차이를 곱셈으로 보존한다."""
    return {k: round(actual_marginal.get(k, 0.0) / max(sim_marginal.get(k, 0.0), floor), 4) for k in set(sim_marginal) | set(actual_marginal)}


def apply_ratio(dist: dict, w: dict) -> dict:
    out = {k: dist.get(k, 0.0) * w.get(k, 1.0) for k in set(dist) | set(w)}
    z = sum(out.values()) or 1.0
    return {k: round(v / z, 4) for k, v in out.items()}


# ---- 보정 카드 (docs/scenario-seoul-ev.md §7 규칙을 제품에) ----
BLEND_LAMBDA = 0.4        # §7: A 절반으로 고른 값. 질문 하나·도시 하나에서 나온 상수라는 한계를 카드에 적는다
LEVEL1_MAX_M1 = 0.10      # 보정 후 M1이 이 미만이면 '수준' 주장(level 1)이 열린다
MIN_OBS_PER_CELL = 10
EB_ALPHAS = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000)   # §10.1 격자


def bayes_update(priors: list[dict], observed: list[dict], ns: list[int], keys: list[str]) -> tuple[list[dict], float]:
    """§10 BS: 셀별 디리클레(α·사전) + 실측 → 사후 평균. α는 디리클레-다항 주변우도 최대(경험적 베이즈).
    실측이 적으면 사전(합성 보정)에, 많으면 실측 직접 추정에 수렴한다."""
    from math import lgamma
    xs = [{k: n * o.get(k, 0.0) for k in keys} for o, n in zip(observed, ns)]
    def ll(a):
        tot = 0.0
        for p, x, n in zip(priors, xs, ns):
            tot += lgamma(a) - lgamma(a + n)
            tot += sum(lgamma(a * p.get(k, 0) + x[k]) - lgamma(a * p.get(k, 0)) for k in keys if a * p.get(k, 0) > 0)
        return tot
    a = max(EB_ALPHAS, key=ll)
    post = [{k: round((a * p.get(k, 0) + x[k]) / (a + n), 4) for k in keys} for p, x, n in zip(priors, xs, ns)]
    return post, float(a)


def main_effects(observed: list[dict], ns: list[int], factors: list[dict], keys: list[str], marginal: dict, smooth: float = 2.0, iters: int = 60) -> list[dict]:
    """§10 R: 요인 주효과 로그선형 모형(상호작용 없음)을 반복비례적합. 셀마다 smooth 건을 전체 비율대로 가산."""
    N = [{k: n * o.get(k, 0.0) + smooth * marginal.get(k, 0.0) for k in keys} for o, n in zip(observed, ns)]
    F = [{k: 1.0 for k in keys} for _ in N]
    groups = [[[i for i, f in enumerate(factors) if f[name] == lv] for lv in sorted({f[name] for f in factors})] for name in factors[0]]
    for _ in range(iters):
        for parts in groups:
            for idx in parts:
                for k in keys:
                    t, fit = sum(N[i][k] for i in idx), sum(F[i][k] for i in idx)
                    for i in idx:
                        F[i][k] = F[i][k] * t / fit if fit > 0 else 0.0
        for i in range(len(F)):
            t, fit = sum(N[i].values()), sum(F[i].values())
            F[i] = {k: v * t / fit if fit > 0 else 0.0 for k, v in F[i].items()}
    return [{k: round(v / (sum(f.values()) or 1.0), 4) for k, v in f.items()} for f in F]


def calibration_card(con, world_id, run: dict, spec: dict, question: dict, lam: float = BLEND_LAMBDA, *, evaluation_world_id: str | None = None) -> dict:
    """시뮬 run의 질문 하나를 월드의 실측 열(observed_attr)과 셀별로 대조하고 비율·혼합 보정을 적용한다.

    기본은 적합 오차만 보고하며 잠금을 열지 않는다. evaluation_world_id가 있으면
    원 월드에서 보정하고 독립 평가 월드로 채점한다. 두 자료의 anchor source_ref는 겹칠 수 없다.
    """
    import json as _j
    from datetime import datetime
    from world.store import active_attrs, build_view, new_id, view_name
    if not 0 <= lam <= 1:
        raise ValueError("lam must be between 0 and 1")
    attr, qid = question["observed_attr"], question["id"]
    vmap = question.get("value_map") or {o: o for o in question.get("options", [])}
    keys = sorted(set(vmap.values()))
    a = next((x for x in active_attrs(con, world_id) if x["name"] == attr), None)
    card = {"schema_version": 2, "run_id": run["run_id"], "world_id": world_id, "question_id": qid, "observed_attr": attr, "lam": lam, "level1_max_m1": LEVEL1_MAX_M1,
            "note": f"합성 사전(비율 보정 + λ={lam} 혼합)을 실측으로 베이지안 갱신(§10). λ는 서울서베이 친환경차 문항 하나에서 고른 상수(§7), 다른 질문으로의 전이는 미검증. 합성 이득은 실측 수백 명 이하에서만 확인(§10.4)."}
    if a is None:
        return {**card, "status": "no_observed", "level1_open": False, "reason": f"월드에 실측 열 {attr} 없음"}
    v = view_name(world_id); build_view(con, world_id)
    cells, sim_pool, obs_pool, n_obs_total, grades = [], {k: 0.0 for k in keys}, {k: 0.0 for k in keys}, 0, set()
    for ci, cell in enumerate(spec["cells"]):
        agg = run["aggregate"].get(str(ci), {}).get(qid)
        if not agg:
            continue
        sim = {vmap.get(k, k): p for k, p in agg["dist"].items()}
        where = f"uuid IN ({','.join(map(repr, cell['uuids']))})" if cell.get("uuids") else f"({cell.get('filter') or 'TRUE'})"
        rows = con.execute(f"SELECT {attr}, CAST(sum(w) AS DOUBLE), count(*), min({attr}__grade) FROM {v} WHERE {where} AND {attr} IS NOT NULL GROUP BY 1").fetchall()
        n_obs = sum(r[2] for r in rows); tot = sum(r[1] for r in rows) or 1.0
        obs = {str(r[0]): round(r[1] / tot, 4) for r in rows}
        grades |= {r[3] for r in rows if r[3]}
        for k in keys:
            sim_pool[k] += sim.get(k, 0) * agg["n"]; obs_pool[k] += obs.get(k, 0) * n_obs
        n_obs_total += n_obs
        cells.append({"cell_id": str(ci), "label": cell.get("label", str(ci)), "n_sim": agg["n"], "n_obs": n_obs, "sim": sim, "observed": obs})
    if n_obs_total == 0:
        return {**card, "status": "no_observed", "level1_open": False, "reason": f"셀 안에 {attr} 값이 있는 행 없음"}
    n_sim_total = sum(c["n_sim"] for c in cells)
    sim_marg = {k: round(x / n_sim_total, 4) for k, x in sim_pool.items()}
    obs_marg = {k: round(x / n_obs_total, 4) for k, x in obs_pool.items()}
    w = ratio_weights(sim_marg, obs_marg)
    # 평가 자료는 위의 보정 학습에 사용하지 않는다.
    evaluation_cells = {}
    evaluation_refs, fit_refs = set(), set()
    if evaluation_world_id is not None:
        if evaluation_world_id == world_id:
            raise ValueError("evaluation world must be separate from fit world")
        ea = next((x for x in active_attrs(con, evaluation_world_id) if x["name"] == attr), None)
        if not ea or a["method"] != "anchor_holdout" or ea["method"] != "anchor_holdout":
            raise ValueError("fit and evaluation targets must both be anchor_holdout")
        observed_grades = {"[공]", "[사]", "[사-CRM]", "[사-전사]", "[사-회귀]", "[사-대면]", "[현]"}
        def observation_refs(wid):
            rows = con.execute("SELECT uuid, source_ref, observed FROM anchor WHERE world_id=?", [wid]).fetchall()
            refs = {}
            for uid, ref, obs in rows:
                if attr in _j.loads(obs):
                    if not ref or uid in refs:
                        raise ValueError("observations need unique source references and anchors")
                    refs[uid] = ref
            if len(set(refs.values())) != len(refs):
                raise ValueError("duplicate observation source_ref")
            return refs
        fit_by_uuid = observation_refs(world_id)
        eval_by_uuid = observation_refs(evaluation_world_id)
        fit_refs, evaluation_refs = set(fit_by_uuid.values()), set(eval_by_uuid.values())
        if not fit_refs or not evaluation_refs or fit_refs & evaluation_refs:
            raise ValueError("fit/evaluation sources missing or overlapping")
        for wid, refs in [(world_id, fit_by_uuid), (evaluation_world_id, eval_by_uuid)]:
            build_view(con, wid)
            rows = con.execute(f"SELECT uuid, {attr}__grade FROM {view_name(wid)} WHERE {attr} IS NOT NULL").fetchall()
            if any(uid not in refs or grade not in observed_grades for uid, grade in rows):
                raise ValueError("target values must have observed grades and anchor provenance")
        ev = view_name(evaluation_world_id)
        for ci, cell in enumerate(spec["cells"]):
            if cell.get("uuids"):
                raise ValueError("independent evaluation requires population filters, not fit UUIDs")
            rows = con.execute(f"SELECT {attr}, CAST(sum(w) AS DOUBLE), count(*) FROM {ev} WHERE ({cell.get('filter') or 'TRUE'}) AND {attr} IS NOT NULL GROUP BY 1").fetchall()
            total = sum(r[1] for r in rows)
            evaluation_cells[str(ci)] = {"n": sum(r[2] for r in rows), "dist": {str(r[0]): r[1] / total for r in rows} if total else {}}
    # 합성 사전(비율 보정 + λ 혼합) → 실측으로 베이지안 갱신(§10.6 결정). 고정 혼합은 실측이 늘어도 나아지지 않았다(§10.4).
    for c in cells:
        cal = apply_ratio(c["sim"], w)
        c["prior"] = {k: round(lam * cal.get(k, 0) + (1 - lam) * obs_marg.get(k, 0), 4) for k in keys}
    posts, alpha = bayes_update([c["prior"] for c in cells], [c["observed"] for c in cells], [c["n_obs"] for c in cells], keys)
    for c, post in zip(cells, posts):
        c["calibrated"] = post
    factors = [spec["cells"][int(c["cell_id"])].get("factors") for c in cells]
    has_factors = bool(cells) and all(factors) and len({tuple(sorted(f)) for f in factors}) == 1
    if has_factors:
        for c, r in zip(cells, main_effects([c["observed"] for c in cells], [c["n_obs"] for c in cells], factors, keys, obs_marg)):
            c["regression"] = r
    num_b = num_a = num_d = num_m = num_r = den = 0.0
    for c in cells:
        evaluation = evaluation_cells.get(c["cell_id"])
        c["n_fit"] = c["n_obs"]
        c["n_evaluation"] = evaluation["n"] if evaluation else 0
        c["evaluation_observed"] = evaluation["dist"] if evaluation else None
        reference = evaluation["dist"] if evaluation else c["observed"]
        n_score = evaluation["n"] if evaluation else c["n_obs"]
        c["level1_open"] = False
        if n_score < MIN_OBS_PER_CELL or c["n_fit"] < MIN_OBS_PER_CELL:
            c["scored"] = False; continue
        c["scored"] = True
        c["tvd_before"] = tvd(c["sim"], reference)
        c["tvd_after"] = tvd(c["calibrated"], reference)
        c["level1_open"] = bool(evaluation and c["tvd_after"] < LEVEL1_MAX_M1)
        # 실측만 쓴 기준선(§10): 셀 직접 추정 · 전체 평균. 독립 평가에서만 의미가 있다(적합 자료로는 직접 추정 오차가 0).
        c["tvd_direct"] = tvd(c["observed"], reference); c["tvd_mean"] = tvd(obs_marg, reference)
        num_b += c["tvd_before"] * n_score; num_a += c["tvd_after"] * n_score; den += n_score
        num_d += c["tvd_direct"] * n_score; num_m += c["tvd_mean"] * n_score
        if has_factors:
            c["tvd_regression"] = tvd(c["regression"], reference); num_r += c["tvd_regression"] * n_score
    m1b = round(num_b / den, 4) if den else None; m1a = round(num_a / den, 4) if den else None
    baselines = {"direct": round(num_d / den, 4), "overall_mean": round(num_m / den, 4)} if den and evaluation_world_id else None
    if baselines and has_factors:
        baselines["regression"] = round(num_r / den, 4)
    beats = bool(baselines and m1a <= min(baselines.values()))
    if baselines and not beats:   # 합성 보정이 실측만 쓴 추정에 지면 수준 주장은 열지 않는다(Kim & Cho 절차 ④)
        for c in cells:
            c["level1_open"] = False
    card.update({"status": "ok", "observed_grade": a["grade"], "value_grades": sorted(grades), "n_observed": n_obs_total,
                 "sim_marginal": sim_marg, "observed_marginal": obs_marg, "weights": w, "m1_before": m1b, "m1_after": m1a,
                 "evaluation_kind": "independent_holdout" if evaluation_world_id else "fit_only",
                 "evaluation_world_id": evaluation_world_id,
                 "fit_source_refs": sorted(fit_refs), "evaluation_source_refs": sorted(evaluation_refs),
                 "n_fit": n_obs_total, "n_evaluation": sum(c["n_evaluation"] for c in cells),
                 "estimator": "bayes_synthetic_prior", "eb_alpha": alpha, "baselines_m1": baselines, "beats_baselines": beats,
                 "reason": ("독립 평가" if beats else "독립 평가: 실측만 쓴 기준선보다 나쁨 → 잠금") if evaluation_world_id else "독립 평가 자료 없음: 적합 오차이며 수치 주장 검증이 아님",
                 "level1_open": bool(cells and all(c["level1_open"] for c in cells)), "cells": cells})
    card_id = new_id("cal")
    con.execute("INSERT INTO calibration VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [card_id, world_id, run["run_id"], qid, attr, a["grade"], n_obs_total, lam, m1b, m1a, card["level1_open"],
                 _j.dumps(card, ensure_ascii=False), datetime.now()])
    return {**card, "card_id": card_id}
