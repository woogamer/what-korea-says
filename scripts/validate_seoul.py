"""검증 시나리오 1 실행: docs/scenario-seoul-ev.md 의 V1~V5. 결과를 문서 §4 와 data/validation_seoul.json 에 쓴다.

  .venv/bin/python -m scripts.validate_seoul            # 전체 (LLM 비용 ≈ $0.15)
  .venv/bin/python -m scripts.validate_seoul --only V1
"""
import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / ".env")

from case import load_case  # noqa: E402
from evidence.seoul_survey import anchor_records, cell_distribution, load  # noqa: E402
from llm import Router  # noqa: E402
from sim.aggregate import tvd  # noqa: E402
from sim.calibrate import conformity_m4  # noqa: E402
from sim.run import run_sim  # noqa: E402
from world import ops  # noqa: E402
from world.diff import diff  # noqa: E402
from world.domain import build_world, load_pack  # noqa: E402
from world.store import connect, view_name, build_view  # noqa: E402

SCOPE = "province = '서울' AND age >= 20"
BANDS = ["19-29", "30-39", "40-49", "50-59", "60-69", "70+"]
OPTIONS = ["전기차", "하이브리드", "수소차", "없음"]
OPT2KEY = dict(zip(OPTIONS, ["ev", "hybrid", "hydrogen", "none"]))
Q = "2년 안에 친환경 자동차를 살 계획이 있습니까?"


def housing_group(h):
    return "아파트" if h == "아파트" else "비아파트"


def survey_cells(df):
    d = df.copy(); d["hg"] = d.housing_type.map(housing_group)
    return cell_distribution(d, "ev_plan", ["age_band", "hg"])


def sim_spec(n_per_cell, seed):
    cells = [{"filter": f"{SCOPE} AND age_band = '{b}' AND housing_type {'=' if hg == '아파트' else '<>'} '아파트'", "n": n_per_cell, "label": f"{b} × {hg}"}
             for b in BANDS for hg in ("아파트", "비아파트")]
    return {"cells": cells, "questions": [{"id": "q", "text": Q, "answer_schema": "choice", "options": OPTIONS}], "seed": seed, "temperature": 0.8}


def sim_cell_dists(run):
    out = {}
    for ci, agg in run["aggregate"].items():
        d = agg["q"]["dist"]
        out[int(ci)] = {OPT2KEY.get(k, k): v for k, v in d.items()}
    return out


def weighted_tvd(sim_dists, spec, B):
    num = den = 0.0; rows = []
    for ci, cell in enumerate(spec["cells"]):
        b = B.get(cell["label"])
        if not b or ci not in sim_dists:
            continue
        t = tvd(sim_dists[ci], b["dist"]); rows.append({"cell": cell["label"], "tvd": t, "n_survey": b["n"],
                                                       "sim_ev": sim_dists[ci].get("ev", 0), "survey_ev": b["dist"].get("ev", 0),
                                                       "sim_dist": sim_dists[ci], "survey_dist": b["dist"]})
        num += t * b["n"]; den += b["n"]
    return round(num / den, 4) if den else None, rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--only"); ap.add_argument("--n", type=int, default=50); ap.add_argument("--anchors", type=int, default=600)
    a = ap.parse_args(); only = set(a.only.split(",")) if a.only else None
    t0 = time.time()
    df = load(); A, B = df[df.split == "A"], df[df.split == "B"]
    Bcells = survey_cells(B)
    con = connect(":memory:")
    pack = load_pack(ROOT / "domains/ev_charging_seoul.yaml"); case = load_case(ROOT / "cases/ev_charging_seoul.yaml")
    v1, res = build_world(con, pack, "seoul v1")
    print(f"world v1 {v1} {[r['status'] for r in res]}  L0 Seoul adults:", con.execute(f"SELECT count(*) FROM {view_name(v1)} WHERE {SCOPE}").fetchone()[0])
    rt = Router(con=con, run_id="validate_seoul", budget_usd=1.0)
    R = {"ts": datetime.now().isoformat(timespec="seconds"), "results": {}}
    prev = ROOT / "data/validation_seoul.json"
    if only and prev.exists():   # 부분 실행이면 이전 결과와 병합
        R["results"] = json.loads(prev.read_text()).get("results", {})

    def run(vid, fn):
        if only and vid not in only:
            return
        t = time.time()
        try:
            R["results"][vid] = fn(); R["results"][vid]["seconds"] = round(time.time() - t)
        except Exception as e:
            import traceback; R["results"][vid] = {"passed": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-800:]}
        print(vid, json.dumps({k: v for k, v in R["results"][vid].items() if k not in ("rows", "trace")}, ensure_ascii=False)[:400], flush=True)

    # V1: 통계 임퓨트 vs B
    def v1_stat():
        w, _ = build_world(con, {**pack, "attributes": pack["attributes"] + pack["validation_attributes"]}, "seoul v1+ev_plan")
        v = view_name(w); build_view(con, w)
        rows = con.execute(f"SELECT age_band, CASE WHEN housing_type='아파트' THEN '아파트' ELSE '비아파트' END hg, ev_plan, count(*) n FROM {v} WHERE {SCOPE} AND ev_plan IS NOT NULL GROUP BY 1,2,3").fetchall()
        per = {}
        for b, hg, val, n in rows:
            per.setdefault(f"{b} × {hg}", {})[val] = n
        num = den = 0.0; detail = []
        for label, d in per.items():
            tot = sum(d.values()); pd_ = {k: v / tot for k, v in d.items()}
            if label in Bcells:
                t = tvd(pd_, Bcells[label]["dist"]); n = Bcells[label]["n"]; num += t * n; den += n
                detail.append({"cell": label, "tvd": t, "imputed_ev": round(pd_.get("ev", 0), 3), "survey_ev": Bcells[label]["dist"].get("ev", 0)})
        m = round(num / den, 4)
        return {"passed": m < 0.10, "tvd_weighted": m, "threshold": 0.10, "rows": detail}
    run("V1", v1_stat)

    # V2: 시뮬 기준선 (조사 정보 없는 v1)
    spec = sim_spec(a.n, seed=21)
    state = {}

    def v2_baseline():
        r = run_sim(con, v1, spec, rt); state["run1"] = r
        m, rows = weighted_tvd(sim_cell_dists(r), spec, Bcells)
        sim_all = {k: 0 for k in OPT2KEY.values()}
        return {"passed": True, "m1_tvd": m, "n_ok": r["n_ok"], "n_requested": r["n_requested"], "cost_usd": round(r["cost_usd"], 4), "rows": rows}
    run("V2", v2_baseline)

    # V3: 정박 후 (v2 월드)
    def v3_anchor():
        _, f = ops.run(con, v1, "fork", {"name": "seoul v2 (정박)"}); v2 = f["new_world_id"]; state["v2"] = v2
        recs = anchor_records(A, a.anchors, seed=3)
        targets = {r["source_ref"]: r.pop("_target") for r in recs}
        _, ar = ops.run(con, v2, "anchor", {"records": recs, "grade": "[사]", "seed": 3})
        uuid2target = {}
        for rec, an in zip(recs, ar["anchors"]):
            uuid2target[an["uuid"]] = targets[rec["source_ref"]]
        r = run_sim(con, v2, spec, rt); state["run2"] = r
        m, rows = weighted_tvd(sim_cell_dists(r), spec, Bcells)
        # M2: 앵커 응답자의 시뮬 답 vs 보류한 실측 구매계획
        ans = con.execute("SELECT uuid, answer FROM sim_response WHERE run_id=?", [r["run_id"]]).fetchall()
        hit = {"ev": [0, 0], "none": [0, 0]}   # target → [n, sim_said_ev]
        for u, s in ans:
            t = uuid2target.get(u)
            if t in hit:
                hit[t][0] += 1; hit[t][1] += (OPT2KEY.get(json.loads(s)["value"]) == "ev")
        p_ev = hit["ev"][1] / hit["ev"][0] if hit["ev"][0] else None
        p_none = hit["none"][1] / hit["none"][0] if hit["none"][0] else None
        lift = round(p_ev / p_none, 2) if p_ev is not None and p_none else None
        base = R["results"].get("V2", {}).get("m1_tvd")
        delta = round(m - base, 4) if base is not None and m is not None else None
        return {"passed": (delta is not None and delta < 0) and (lift is not None and lift > 1.2), "m1_tvd": m, "delta_m1": delta,
                "m2": {"n_anchor_ev": hit["ev"][0], "p_sim_ev_given_ev": p_ev and round(p_ev, 3), "n_anchor_none": hit["none"][0],
                       "p_sim_ev_given_none": p_none and round(p_none, 3), "lift": lift},
                "n_anchored_respondents": sum(1 for u, _ in ans if u in uuid2target), "cost_usd": round(r["cost_usd"], 4), "rows": rows}
    run("V3", v3_anchor)

    # V4: 재판단
    def v4_rejudge():
        from tools.runner import run_tool
        from tools.checks import auto_stp_decisions
        v2 = state.get("v2")
        if not v2:
            _, f = ops.run(con, v1, "fork", {"name": "seoul v2"}); v2 = f["new_world_id"]
            ops.run(con, v2, "anchor", {"records": [dict(r, **{}) for r in anchor_records(A, 100, seed=3)], "grade": "[사]"})
        s1 = run_tool(con, case, v1, "stp", rt, decisions=auto_stp_decisions(pack))
        prev = s1["decision_ids"]["g2_target"]; prev_cell = s1["decisions"]["g2_target"]["cell"]
        dec = auto_stp_decisions(pack)
        dec["g2_target"] = lambda view: {"cell": view["s2_cells"]["largest"], "supersedes": prev,
                                        "rationale": ("유지: 정박 후에도 규모 최대 셀 동일 " if view["s2_cells"]["largest"] == prev_cell else "수정: 규모 최대 셀 변경 ") + f"{prev_cell} → {view['s2_cells']['largest']}"}
        s2 = run_tool(con, case, v2, "stp", rt, decisions=dec)
        d = diff(con, v1, v2, pack["axes_default"], filter=SCOPE)
        changed = sum(1 for c in d["cells"] if c["size_delta"] != 0 or c["anchor_n_b"] != c["anchor_n_a"])
        ok = s1["status"] == "done" and s2["status"] == "done" and changed > 0
        return {"passed": ok, "v1_target": prev_cell, "v2_target": s2["decisions"].get("g2_target", {}).get("cell"),
                "rationale": s2["decisions"].get("g2_target", {}).get("rationale"), "cells_changed": changed, "anchors": d["anchors"],
                "status": [s1["status"], s2["status"], s2.get("error")]}
    run("V4", v4_rejudge)

    # V5: 동조율
    def v5_m4():
        r = conformity_m4(con, v1, rt, n=300, filter=SCOPE, question=Q, seed=5)
        return {"passed": r["m4"] < 0.10, **{k: r[k] for k in ("m4", "ci95", "p_a", "p_b", "n_ok")}}
    run("V5", v5_m4)

    R["cost_usd"] = round(rt.total_cost, 4); R["seconds"] = round(time.time() - t0)
    (ROOT / "data/validation_seoul.json").write_text(json.dumps(R, ensure_ascii=False, indent=1))
    write_doc(R)
    print(f"done cost=${rt.total_cost:.4f} {R['seconds']}s")


def write_doc(R):
    p = ROOT / "docs/scenario-seoul-ev.md"; s = p.read_text(encoding="utf-8")
    lines = [f"## 4. 결과 (실행 {R['ts']}, LLM 비용 ${R.get('cost_usd', 0)}, {R.get('seconds', 0)}초)", "",
             "| ID | 판정 | 수치 |", "|---|---|---|"]
    for vid, r in R["results"].items():
        m = {k: v for k, v in r.items() if k not in ("rows", "passed", "trace", "seconds")}
        lines.append(f"| {vid} | {'✅' if r.get('passed') else '❌'} | {json.dumps(m, ensure_ascii=False)} |")
    for vid in ("V1", "V2", "V3"):
        rows = R["results"].get(vid, {}).get("rows")
        if rows:
            lines += ["", f"**{vid} 셀별**", "", "| 셀 | TVD | 시뮬/임퓨트 EV | 실측 EV | 실측 n |", "|---|---|---|---|---|"]
            for x in rows:
                lines.append(f"| {x['cell']} | {x['tvd']} | {x.get('sim_ev', x.get('imputed_ev'))} | {x['survey_ev']} | {x.get('n_survey', '')} |")
    s = re.sub(r"## 4\. 결과.*?(?=\n## 5\.|\Z)", "\n".join(lines) + "\n", s, flags=re.S)
    p.write_text(s, encoding="utf-8")


if __name__ == "__main__":
    main()
