"""C6: 새 근거로 재판단. v1 → fork v2 → 조사 문서 ingest(모순 ≥2 검출) → 승인·커밋(anchor·reweight) → 같은 FGI 스펙 재실행 → diff + M2 + 판단 v2."""
import json
from pathlib import Path

from sim.aggregate import tvd
from sim.run import run_sim
from world import ops
from world.diff import diff

from .ingest import commit, ingest

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DOC = ROOT / "cases/samples/ev_survey_v1.md"


def verify_rejudge(con, case, world_v1, router, chain, pack, doc_path=SAMPLE_DOC) -> dict:
    from tools.runner import run_tool
    from tools.checks import auto_stp_decisions
    problems, metrics = [], {}
    stp1, fgi1 = chain["stp"], chain["fgi"]
    # 1) fork → v2
    _, r = ops.run(con, world_v1, "fork", {"name": f"{case.name} v2 (조사 반영)"})
    v2 = r["new_world_id"]
    # 2) ingest: 모순 검출
    existing = [dict(zip(["id", "proposition", "grade"], x)) for x in con.execute(
        "SELECT belief_id, proposition, grade FROM belief WHERE world_id=?", [v2]).fetchall()]
    ing = ingest(con, v2, doc_path.read_text(encoding="utf-8"), router, existing)
    n_conf = len(ing["conflicts"])
    metrics.update(n_props=len(ing["propositions"]), n_conflicts=n_conf, n_llm_conflicts=len(ing["llm_conflicts"]))
    kinds = {c["type"] for c in ing["conflicts"]}
    if n_conf < 2 or not {"denominator", "sample_condition_missing"} <= kinds:
        problems.append(f"심어둔 모순 2종 미검출: {sorted(kinds)} ({n_conf}건)")
    # 3) 승인(모순 항목 제외) → 커밋
    approved = [p["text"] for p in ing["propositions"] if p["text"] not in ing["flagged"]]
    committed_before = con.execute("SELECT count(*) FROM belief WHERE world_id=? AND status='committed'", [v2]).fetchone()[0]
    target_f = fgi1["outputs"]["s0_spec"]["cells"][0]["filter"]
    cm = commit(con, v2, ing, approved, reweight_filter=target_f, source_ref=doc_path.name)
    metrics.update(n_beliefs=len(cm["beliefs"]), ops=[(o["op"], o["status"]) for o in cm["ops"]])
    if any(o["status"] != "committed" for o in cm["ops"]):
        problems.append(f"연산 실패 {cm['ops']}")
    flagged_committed = con.execute("SELECT count(*) FROM belief WHERE world_id=? AND status='committed' AND proposition IN (SELECT UNNEST(?::VARCHAR[]))",
                                    [v2, ing["flagged"]]).fetchone()[0] if ing["flagged"] else 0
    if flagged_committed:
        problems.append("모순 명제가 승인 없이 committed 됨")
    # 4) diff v1 vs v2
    axes = stp1["decisions"]["g1_adopt_axes"]["axes"]
    d = diff(con, world_v1, v2, axes, filter=case.world.market_scope)
    n_cells_changed = sum(1 for c in d["cells"] if c["size_delta"] != 0 or c["anchor_n_b"] != c["anchor_n_a"])  # 규모 또는 관측 n 변화
    metrics.update(anchors_v2=d["anchors"][v2], cells_changed=n_cells_changed)
    if d["anchors"][v2] == 0 or n_cells_changed == 0:
        problems.append(f"diff 에 변화 없음: anchors={d['anchors']} attrs={d['attrs']} ops={cm['ops']} cells={[c['size_delta'] for c in d['cells']]}")
    # 5) 같은 FGI 스펙 v2 재실행 → 분포 변화 + M2
    spec = {k: v for k, v in fgi1["outputs"]["s0_spec"].items() if k != "provenance"}
    run2 = run_sim(con, v2, spec, router)
    agg1 = fgi1["outputs"]["s1_run"]["aggregate"]
    label0 = spec["cells"][0]["label"]
    tv = {q: tvd(agg1[label0][q]["dist"], run2["aggregate"]["0"][q]["dist"]) for q in agg1[label0] if q in run2["aggregate"].get("0", {})}
    metrics.update(tvd_target=tv)
    if not tv or max(tv.values()) == 0:
        problems.append("v2 재실행 분포가 v1 과 동일")
    metrics["m2"] = _m2(con, v2, run2["run_id"], spec["questions"][0]["id"])
    # 6) 판단 v2: 같은 STP 를 v2 에서 → 이전 판단을 supersedes
    prev = stp1["decision_ids"]["g2_target"]
    prev_cell = stp1["decisions"]["g2_target"]["cell"]

    def rejudge(view):
        new = view["s2_cells"]["largest"]
        keep = new == prev_cell
        return {"cell": new, "supersedes": prev,
                "rationale": (f"유지: 조사 반영(앵커 {d['anchors'][v2]}·가중치) 후에도 규모 최대 셀이 {new} 로 같음" if keep
                              else f"수정: 조사 반영 후 규모 최대 셀이 {prev_cell} → {new}")}
    dec = auto_stp_decisions(pack); dec["g2_target"] = rejudge
    stp2 = run_tool(con, case, v2, "stp", router, decisions=dec)
    if stp2["status"] != "done":
        problems.append(f"v2 STP {stp2['status']}: {stp2.get('error')}")
    else:
        row = con.execute("SELECT chosen, rationale FROM decision WHERE decision_id=?", [stp2["decision_ids"]["g2_target"]]).fetchone()
        ch = json.loads(row[0])
        if ch.get("supersedes") != prev or not (row[1].startswith("유지") or row[1].startswith("수정")):
            problems.append("판단 v2 에 supersedes/유지·수정 이유 없음")
        metrics["decision_v2"] = row[1][:60]
    ev = (f"모순 {n_conf}건 검출({', '.join(sorted(kinds))}) → 승인 {len(approved)}/{len(ing['propositions'])} → 앵커 {d['anchors'][v2]}·가중치 → "
          f"셀 규모 변화 {n_cells_changed}칸, FGI 재실행 TVD {tv}, M2 {metrics['m2']}, 판단: {metrics.get('decision_v2','')}")
    return {"passed": not problems, "evidence": ev if not problems else "; ".join(problems) + " | " + ev,
            "metrics": {k: v for k, v in metrics.items() if k in ("n_conflicts", "cells_changed", "anchors_v2", "m2")}, "v2": v2}


def _m2(con, world_id, run_id, qid) -> dict | None:
    """앵커 응답자: 관측 purchase_intent(≥4) vs 시뮬 답(≥4) 적중률."""
    rows = con.execute(
        "SELECT a.observed, r.answer FROM anchor a JOIN sim_response r ON a.uuid=r.uuid AND r.run_id=? WHERE a.world_id=? AND r.question_id=?",
        [run_id, world_id, qid]).fetchall()
    if not rows:
        return None
    hit = 0
    for obs, ans in rows:
        o = json.loads(obs).get("purchase_intent"); s = json.loads(ans)["value"]
        if o is not None and (int(o) >= 4) == (int(s) >= 4):
            hit += 1
    return {"n": len(rows), "hit_rate": round(hit / len(rows), 2)}
