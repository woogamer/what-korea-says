"""query 단계: DB만 읽는다. LLM 없음. 수치는 전부 여기서 나온다."""
import json

from world import reconcile
from world.segment import segment
from world.store import active_attrs, attr_grade

MAX_K = 12  # 범주가 이보다 많은 열은 축으로 부적합(직업 1,385개 등)
L0_AXIS_CANDIDATES = ["sex", "age_band", "marital_status", "family_type", "housing_type",
                      "education_level", "occupation", "province"]


def eligibility(ctx) -> dict:
    """축 후보 자격 심사: 월드 안에서 값이 실제로 갈리는 열만(정규화 엔트로피·커버리지)."""
    con, w = ctx["con"], ctx["world_id"]
    scope = ctx["case"].world.market_scope
    cols = L0_AXIS_CANDIDATES + [a["name"] for a in active_attrs(con, w)]
    rows = []
    for c in cols:
        r = reconcile.variance(con, w, c, filter=scope)
        d = r["detail"]
        rows.append({"attr": c, "eligible": r["passed"] and d["k"] <= MAX_K, "norm_entropy": d["norm_entropy"],
                     "coverage": d["coverage"], "k": d["k"], "grade": attr_grade(con, w, c)})
    return {"market_scope": scope, "attrs": rows, "eligible": [r["attr"] for r in rows if r["eligible"]]}


def cells(ctx) -> dict:
    """채택된 축으로 세분시장 셀 표. 규모[도]·관측 n·앵커 n·시뮬 가능."""
    con, w = ctx["con"], ctx["world_id"]
    axes = ctx["decisions"]["g1_adopt_axes"]["axes"]
    df = segment(con, w, axes, filter=ctx["case"].world.market_scope)
    recs = df.to_dict("records")
    for r in recs:
        r["cell"] = " × ".join(str(r[a]) for a in axes)
    return {"axes": axes, "cells": recs, "grade": df.grade.iloc[0] if len(df) else "[미]",
            "largest": max(recs, key=lambda r: r["size_est"])["cell"] if recs else None}


def beliefs(ctx) -> dict:
    con, w = ctx["con"], ctx["world_id"]
    rows = con.execute("SELECT belief_id, proposition, grade, source_ref, n, sample_condition FROM belief "
                       "WHERE world_id=? AND status='committed'", [w]).fetchall()
    out = [dict(zip(["id", "proposition", "grade", "source_ref", "n", "sample_condition"], r)) for r in rows]
    out += [{"id": e.id, "proposition": e.proposition, "grade": e.grade, "source_ref": e.source_ref, "dimension": e.dimension}
            for e in ctx["case"].environment]
    return {"beliefs": out}


def drivers(ctx) -> dict:
    """시나리오: 환경 동인(개인 속성 아님). case.environment 그대로."""
    c = ctx["case"]
    return {"drivers": [e.model_dump() for e in c.environment], "controllable": c.controllable,
            "horizon": "향후 10년", "decision": c.decision}


def target_cell_rows(ctx) -> dict:
    """게이트 뒤: 타겟 셀의 규모·앵커만 DB에서 다시 읽는다(게이트 앞 LLM 출력 불필요)."""
    d = ctx["decisions"]
    axes = d["g1_adopt_axes"]["axes"]
    con, w = ctx["con"], ctx["world_id"]
    df = segment(con, w, axes, filter=ctx["case"].world.market_scope)
    df["cell"] = df[axes].astype(str).agg(" × ".join, axis=1)
    row = df[df.cell == d["g2_target"]["cell"]]
    return {"axes": axes, "target": d["g2_target"]["cell"], "row": row.to_dict("records")[0] if len(row) else None,
            "total_size": int(df.size_est.sum())}


QUERIES = {f.__name__: f for f in [eligibility, cells, beliefs, drivers, target_cell_rows]}


# ---- 툴 연결: 타겟 셀 → 페르소나 → FGI 스펙 ----

def cell_filter(con, world_id, axes: list[str], cell: str, market_scope: str | None) -> str:
    """'a × b' 셀 라벨 → SQL 필터. [미]는 NULL 값."""
    vals = cell.split(" × ")
    terms = [f"COALESCE(CAST({a} AS VARCHAR), '[미]') = '{v}'" for a, v in zip(axes, vals)]
    if market_scope:
        terms.append(f"({market_scope})")
    return " AND ".join(terms)


def target_filter(ctx) -> str:
    d = ctx["decisions"]
    return cell_filter(ctx["con"], ctx["world_id"], d["g1_adopt_axes"]["axes"], d["g2_target"]["cell"], ctx["case"].world.market_scope)


def target_sample(ctx, n: int = 3, seed: int = 7) -> dict:
    """페르소나 = 표적 셀에서 층화(성별×연령대) 샘플한 실제 L0 행. 앵커(실측) 행 우선."""
    from sim.render import sample_rows
    con, w = ctx["con"], ctx["world_id"]
    f = target_filter(ctx)
    from world.store import build_view, view_name
    v = view_name(w); build_view(con, w)
    df = con.execute(f"""
        WITH r AS (
          SELECT v.*, (an.uuid IS NOT NULL) AS anchored,
                 ROW_NUMBER() OVER (PARTITION BY sex, age_band ORDER BY (an.uuid IS NOT NULL) DESC, hash(v.uuid || '{seed}')) AS rn,
                 count(*) OVER (PARTITION BY sex, age_band) AS grp
          FROM {v} v LEFT JOIN (SELECT DISTINCT uuid FROM anchor WHERE world_id='{w}') an USING (uuid)
          WHERE {f})
        SELECT * FROM r WHERE rn = 1 ORDER BY anchored DESC, grp DESC LIMIT {n}""").df()
    keep = ["uuid", "sex", "age", "age_band", "province", "district", "occupation", "family_type", "housing_type",
            "education_level", "persona", "family_persona", "hobbies_and_interests", "anchored"]
    from world.store import active_attrs
    l1 = [a["name"] for a in active_attrs(con, w)]
    rows = []
    for r in df.to_dict("records"):
        row = {k: (r[k][:300] if isinstance(r[k], str) else r[k]) for k in keep}
        row["attrs"] = {c: r.get(c) for c in l1 if r.get(c) is not None}
        row["grades"] = {c: r.get(f"{c}__grade") for c in l1 if r.get(c) is not None}
        rows.append(row)
    return {"filter": f, "target": ctx["decisions"]["g2_target"]["cell"], "rows": rows,
            "n_in_cell": int(con.execute(f"SELECT count(*) FROM {v} WHERE {f}").fetchone()[0]),
            "provenance": {"decisions": ctx.get("decision_ids", {})}}


def fgi_spec(ctx, n_per_cell: int = 20) -> dict:
    """FGI 스펙: 타겟 셀 + 대조 셀(시장 범위 안의 나머지). 질문 = 채택된 니즈 문장(likert5) + 이유. LLM 없음."""
    f = target_filter(ctx)
    scope = ctx["case"].world.market_scope or "TRUE"
    needs = ctx["decisions"]["g1_needs"]["needs"]
    questions = [{"id": f"n{i+1}", "text": f"다음 진술에 얼마나 동의합니까: \"{t}\"", "answer_schema": "likert5"} for i, t in enumerate(needs)]
    questions += _calibration_questions(ctx)   # 팩이 선언한 보정 질문(실측 열과 대조 가능한 문항)을 같은 응답자에게 함께 묻는다
    return {"cells": [{"filter": f, "n": n_per_cell, "label": ctx["decisions"]["g2_target"]["cell"]},
                      {"filter": f"({scope}) AND NOT ({f})", "n": n_per_cell, "label": "대조: 나머지"}],
            "questions": questions, "seed": 11, "temperature": 0.8,
            "provenance": {"decisions": ctx.get("decision_ids", {})}}


def _calibration_questions(ctx) -> list[dict]:
    from pathlib import Path
    from world.domain import load_pack
    pack_path = ctx["case"].world.pack
    if not pack_path:
        return []
    pack = load_pack(Path(__file__).resolve().parent.parent / pack_path)
    return [dict(q) for q in pack.get("calibration_questions", [])]


def fgi_run(ctx) -> dict:
    """sim 단계: 응답자마다 독립 요청. 스펙의 질문에는 분석자 가설이 들어가지만 응답자 렌더링에는 들어가지 않는다."""
    from sim.run import run_sim
    spec = ctx["outputs"]["s0_spec"]
    r = run_sim(ctx["con"], ctx["world_id"], {k: v for k, v in spec.items() if k != "provenance"}, ctx["router"])
    # 최빈값에서 먼 응답의 이유를 뽑아 모더레이터 입력으로
    import json as _j
    rows = ctx["con"].execute("SELECT uuid, question_id, answer FROM sim_response WHERE run_id=?", [r["run_id"]]).fetchall()
    by_q = {}
    for u, q, a in rows:
        a = _j.loads(a); by_q.setdefault((str(a["cell"]), q), []).append((u, a["value"], a["reason"]))
    outliers = []
    for (cell, q), lst in by_q.items():
        vals = [v for _, v, _ in lst]
        mode = max(set(vals), key=vals.count)
        for u, v, reason in lst:
            if isinstance(v, str) or abs(int(v) - int(mode)) >= 2:
                if isinstance(v, str) and v == mode:
                    continue
                outliers.append({"cell": spec["cells"][int(cell)]["label"], "q": q, "value": v, "mode": mode, "reason": reason})
    labels = {str(i): c["label"] for i, c in enumerate(spec["cells"])}
    from sim.calibrate import calibration_card
    cards = [calibration_card(ctx["con"], ctx["world_id"], r, spec, q, evaluation_world_id=ctx.get("evaluation_world_id"))
             for q in spec["questions"] if q.get("observed_attr")]
    return {"run_id": r["run_id"], "n_requested": r["n_requested"], "n_ok": r["n_ok"], "cost_usd": round(r["cost_usd"], 4),
            "aggregate": {labels[k]: v for k, v in r["aggregate"].items()}, "outliers": outliers[:12],
            "questions": spec["questions"], "cell_labels": labels, "calibration": cards, "grade": "[시]", "provenance": spec["provenance"]}


QUERIES.update({f.__name__: f for f in [target_sample, fgi_spec, fgi_run]})


def plan_context(ctx) -> dict:
    """조사 계획 입력: case 의 결정·선택지·정보 필요 + 채택 가설 + 타겟. LLM 없음."""
    c = ctx["case"]
    return {"decision": c.decision, "options": [o.model_dump() for o in c.options],
            "information_needs": [n.model_dump() for n in c.information_needs],
            "existing_rqs": [r.model_dump(by_alias=True) for r in c.research_questions],
            "target": ctx["decisions"]["g2_target"]["cell"], "target_population": c.target_population,
            "hypotheses": ctx["decisions"]["g1_hypotheses"]["hypotheses"]}


QUERIES["plan_context"] = plan_context


# ---- 근거 수집 에이전트 (tools/manifests/evidence_agent.yaml) ----

def _pack(ctx):
    from pathlib import Path
    from world.domain import load_pack
    return load_pack(Path(__file__).resolve().parent.parent / ctx["case"].world.pack)


def agent_collect(ctx) -> dict:
    """팩의 agent_attributes 마다 에이전트 루프. 월드에는 쓰지 않는다(게이트 뒤 agent_apply)."""
    from evidence.agent import collect
    scope = ctx["case"].world.market_scope or "TRUE"
    reqs = _pack(ctx).get("agent_attributes", [])
    cands = [collect(ctx["con"], ctx["world_id"], ctx["router"], r, scope) for r in reqs]
    return {"candidates": cands, "grade": "[미]",
            "summary": [{"attr": c["attr"], "passed": c["checks"]["passed"], "evidence_level": c["checks"].get("evidence_level"), "coverage": c["checks"]["coverage"],
                         "url": c["table"]["url"], "year": c["table"]["year"], "rows": len(c["checks"]["rows"]), "problems": c["checks"]["problems"],
                         "warnings": c["checks"].get("warnings", []), "conflicts": c["conflicts"]} for c in cands]}


def agent_apply(ctx) -> dict:
    from evidence.agent import apply
    approved = set(ctx["decisions"]["g1_approve"]["approved"])
    cands = {c["attr"]: c for c in ctx["outputs"]["s0_collect"]["candidates"]}
    results = [{"attr": a, **apply(ctx["con"], ctx["world_id"], cands[a])} for a in approved if a in cands]
    return {"applied": results, "grade": "[도]" if results else "[미]", "provenance": {"decisions": ctx.get("decision_ids", {})}}


def l0_card(ctx) -> dict:
    """L0 정합성 카드: 팩이 reference_distributions 를 주면 시장 범위 안 페르소나 분포와 대조."""
    from world.reconcile import l0_consistency
    ref = _pack(ctx).get("reference_distributions")
    if not ref:
        return {"status": "no_reference", "grade": "[미]"}
    r = l0_consistency(ctx["con"], ctx["world_id"], ref["axes"], filter=ctx["case"].world.market_scope)
    return {**r, "source_ref": ref.get("source_ref"), "grade": "[공]"}


QUERIES.update({f.__name__: f for f in [agent_collect, agent_apply, l0_card]})
