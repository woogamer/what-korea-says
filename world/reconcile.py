"""정합성 검사. 각 함수는 {"check", "passed", "detail"} 를 돌려준다."""
import math

from .store import build_view, row_weight, view_name


def total(con, world_id, expr, expected, tol=0.05, filter=None):
    """Σ(w × 행가중치 × 1[expr]) 이 expected ± tol 안인가. 예: 공개 통계의 모수 총량."""
    v = view_name(world_id); build_view(con, world_id)
    where = filter or "TRUE"
    got = con.execute(f"SELECT CAST(COALESCE(sum(w),0) AS DOUBLE) FROM {v} WHERE ({where}) AND ({expr})").fetchone()[0]
    got *= row_weight(con)
    rel = abs(got - expected) / expected
    return {"check": "total", "passed": rel <= tol,
            "detail": {"expr": expr, "estimated": round(got), "expected": expected, "rel_err": round(rel, 4)}}


def denominator(funnel, claims, tol=0.05):
    """퍼널 절대 수와 전환율 주장이 같은 분모에서 성립하는가.

    funnel = {"inquiry":100, "visit":14, "buy":8}; claims = [{"from":"visit","to":"buy","rate":0.96}]
    """
    problems = []
    for c in claims:
        implied = funnel[c["to"]] / funnel[c["from"]]
        if abs(implied - c["rate"]) > tol:
            problems.append({**c, "implied": round(implied, 3)})
    return {"check": "denominator", "passed": not problems, "detail": {"conflicts": problems}}


def variance(con, world_id, attr, filter=None, min_norm_entropy=0.1, min_coverage=0.5):
    """축으로 쓸 열이 모집단 안에서 실제로 갈리는가 (정규화 엔트로피) + 커버리지."""
    v = view_name(world_id); build_view(con, world_id)
    where = filter or "TRUE"
    rows = con.execute(f"SELECT {attr}, CAST(sum(w) AS DOUBLE) FROM {v} WHERE {where} GROUP BY 1").fetchall()
    tot = sum(r[1] for r in rows)
    covered = sum(r[1] for r in rows if r[0] is not None)
    dist = [r[1] / covered for r in rows if r[0] is not None and r[1] > 0]
    k = len(dist)
    H = -sum(p * math.log(p) for p in dist) / math.log(k) if k > 1 else 0.0
    cov = covered / tot if tot else 0.0
    return {"check": "variance", "passed": H >= min_norm_entropy and cov >= min_coverage,
            "detail": {"attr": attr, "norm_entropy": round(H, 3), "coverage": round(cov, 3), "k": k}}


def anchor_bias(con, world_id, cols, filter=None, max_tvd=0.3):
    """앵커(실측)의 분포 vs 같은 filter 모집단 분포의 TVD. 접촉표본 편향 크기."""
    v = view_name(world_id); build_view(con, world_id)
    where = filter or "TRUE"
    out, worst = {}, 0.0
    for c in cols:
        pop = dict(con.execute(f"SELECT {c}, CAST(sum(w) AS DOUBLE) FROM {v} WHERE {where} GROUP BY 1").fetchall())
        anc = dict(con.execute(
            f"SELECT v.{c}, count(*) FROM anchor a JOIN {v} v USING (uuid) "
            f"WHERE a.world_id=? AND ({where}) GROUP BY 1", [world_id]).fetchall())
        if not anc:
            return {"check": "anchor_bias", "passed": True, "detail": {"note": "no anchors"}}
        sp, sa = sum(pop.values()), sum(anc.values())
        tvd = 0.5 * sum(abs(pop.get(k, 0) / sp - anc.get(k, 0) / sa) for k in set(pop) | set(anc))
        out[c] = round(tvd, 3); worst = max(worst, tvd)
    return {"check": "anchor_bias", "passed": worst <= max_tvd, "detail": {"tvd": out, "n_anchor": sa}}


CHECKS = {"total": total, "denominator": denominator, "variance": variance, "anchor_bias": anchor_bias}


def run_check(con, world_id, spec):
    """op params["checks"] 항목 실행. spec = {"type": "total", ...kwargs}"""
    spec = dict(spec); kind = spec.pop("type")
    fn = CHECKS[kind]
    return fn(**spec) if kind == "denominator" else fn(con, world_id, **spec)


def l0_consistency(con, world_id, reference: dict, filter=None, max_tvd=0.10) -> dict:
    """L0 정합성 카드: 시장 범위 안 페르소나 DB의 축별 분포 vs 참조 분포(공식 통계·조사 가중치). 축마다 TVD.

    reference = {col: {value: share}}. 페르소나 DB가 '국가 통계에 정합'하다는 주장을 우리 수치로 확인·반증하는 자리.
    """
    from .store import build_view, view_name
    v = view_name(world_id); build_view(con, world_id)
    where = filter or "TRUE"
    axes = []
    for col, ref in reference.items():
        rows = con.execute(f"SELECT {col}, CAST(sum(w) AS DOUBLE) FROM {v} WHERE ({where}) AND {col} IS NOT NULL GROUP BY 1").fetchall()
        tot = sum(r[1] for r in rows) or 1.0
        l0 = {str(r[0]): round(r[1] / tot, 4) for r in rows}
        keys = set(l0) | set(ref)
        t = round(0.5 * sum(abs(l0.get(k, 0) - ref.get(k, 0)) for k in keys), 4)
        worst = max(keys, key=lambda k: abs(l0.get(k, 0) - ref.get(k, 0)))
        axes.append({"col": col, "tvd": t, "passed": t < max_tvd, "l0": l0, "reference": ref,
                     "worst_value": worst, "l0_share": l0.get(worst, 0), "ref_share": ref.get(worst, 0)})
    return {"passed": all(a["passed"] for a in axes), "max_tvd": max_tvd, "axes": axes, "filter": where,
            "reweight_suggestion": [a["col"] for a in axes if not a["passed"]],   # ops.reweight targets={col: reference[col]} 로 맞출 수 있는 축
            "reweight_targets": {a["col"]: a["reference"] for a in axes if not a["passed"]}}
