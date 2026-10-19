"""분포 보정 (docs/scenario-seoul-ev.md §7). 시뮬의 "없음" 과대는 모델 사전 편향 → A의 정박 응답자에서 보정표를 배우고 B로 채점.

  응답자: A에서 정박한 페르소나 N명(셀 12개). 조건 E0(단일 질문) · E4(2단 질문: 차 구매 계획 → 종류).
  보정: 전이표 T[시뮬→실측] / 비율 w=P_A/P_sim. 각각 (g) 전체 A로 학습 → 같은 셀에 적용, (loco) 그 셀 응답자를 빼고 학습 → 전이 검증.
  혼합: λ·비율보정(loco) + (1-λ)·A 주변분포, λ는 A 셀 분포로 고르고 B로 채점(B는 선택에 안 씀).
  기준선: A 전체 주변분포(시뮬 없음) · A 셀별(=V1). 결과 → data/calibration.json(응답자별 답 포함) + 문서 §7.
"""
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

from evidence.seoul_survey import anchor_records, load  # noqa: E402
from llm import Router  # noqa: E402
from scripts.validate_seoul import OPT2KEY, OPTIONS, Q, survey_cells  # noqa: E402
from sim.aggregate import tvd  # noqa: E402
from sim.calibrate import apply_ratio, apply_transition, ratio_weights, transition_table  # noqa: E402
from sim.run import run_sim  # noqa: E402
from world import ops  # noqa: E402
from world.domain import build_world, load_pack  # noqa: E402
from world.store import build_view, connect, view_name  # noqa: E402

KEYS = list(OPT2KEY.values())
Q2_OPTIONS = ["전기차", "하이브리드", "수소차", "휘발유·경유·LPG차"]
TWO_STAGE = [{"id": "q1", "text": "2년 안에 자동차를 새로 살(또는 바꿀) 계획이 있습니까?", "answer_schema": "yes_no"},
             {"id": "q2", "text": "자동차를 산다면 어떤 종류를 고르겠습니까?", "answer_schema": "choice", "options": Q2_OPTIONS}]


def dist(vals):
    n = len(vals) or 1
    return {k: round(sum(v == k for v in vals) / n, 4) for k in KEYS}


def wtvd(cell_dists, Bcells):
    num = den = 0.0
    for lab, d in cell_dists.items():
        b = Bcells.get(lab)
        if b:
            num += tvd(d, b["dist"]) * b["n"]; den += b["n"]
    return round(num / den, 4)


def corr(x, y):
    n = len(x); mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y)); sx = sum((a - mx) ** 2 for a in x) ** 0.5; sy = sum((b - my) ** 2 for b in y) ** 0.5
    return round(sxy / (sx * sy), 3) if sx and sy else None


def blend(cal, A_marg, lam):
    return {k: round(lam * cal.get(k, 0) + (1 - lam) * A_marg.get(k, 0), 4) for k in A_marg}


def evaluate(ans, cells, targets, Bcells, Acells, A_marg):
    """ans: uuid→시뮬 답. 조건 하나의 원시·보정 M1과 셀별 EV 비율. 혼합 λ는 A 셀 분포로 선택."""
    labels = sorted(cells)
    raw = {lab: dist([ans[u] for u in cells[lab] if u in ans]) for lab in labels}
    pairs = {lab: [(ans[u], targets[u]) for u in cells[lab] if u in ans] for lab in labels}
    allp = [p for ps in pairs.values() for p in ps]
    sim_marg = dist([s for s, _ in allp]); act_marg = dist([a for _, a in allp])
    Tg = transition_table(allp, KEYS); wg = ratio_weights(sim_marg, act_marg)
    out = {"raw": raw, "transition_g": {}, "transition_loco": {}, "ratio_g": {}, "ratio_loco": {}, "a_marginal": {}}
    for lab in labels:
        rest = [p for l2, ps in pairs.items() if l2 != lab for p in ps]
        Tl = transition_table(rest, KEYS); wl = ratio_weights(dist([s for s, _ in rest]), dist([a for _, a in rest]))
        out["transition_g"][lab] = apply_transition(raw[lab], Tg); out["transition_loco"][lab] = apply_transition(raw[lab], Tl)
        out["ratio_g"][lab] = apply_ratio(raw[lab], wg); out["ratio_loco"][lab] = apply_ratio(raw[lab], wl)
        out["a_marginal"][lab] = A_marg
    lams = [i / 10 for i in range(11)]
    lam = min(lams, key=lambda l: wtvd({lab: blend(out["ratio_loco"][lab], A_marg, l) for lab in labels}, Acells))
    out["blend_loco"] = {lab: blend(out["ratio_loco"][lab], A_marg, lam) for lab in labels}
    m1 = {k: wtvd(v, Bcells) for k, v in out.items()}
    b_ev = [Bcells[lab]["dist"].get("ev", 0) for lab in labels]
    ev_corr = {k: corr([v[lab].get("ev", 0) for lab in labels], b_ev) for k, v in out.items() if k != "a_marginal"}
    return {"m1": m1, "blend_lambda": lam, "ev_corr_with_B": ev_corr, "overall_sim": sim_marg, "overall_actual_A": act_marg, "T_global": Tg, "w_global": wg,
            "cells": {lab: {"n": len(pairs[lab]), "raw": raw[lab], "transition_loco": out["transition_loco"][lab], "ratio_loco": out["ratio_loco"][lab],
                            "blend_loco": out["blend_loco"][lab], "survey_B": Bcells[lab]["dist"]} for lab in labels},
            "answers": ans}


def main(n_anchors=1200):
    t0 = time.time()
    df = load(); A, B = df[df.split == "A"], df[df.split == "B"]
    Bcells = survey_cells(B); Acells = survey_cells(A)
    A_marg = A.groupby("ev_plan")["w"].sum(); A_marg = {k: round(v, 4) for k, v in (A_marg / A_marg.sum()).items()}
    con = connect(":memory:")
    pack = load_pack(ROOT / "domains/ev_charging_seoul.yaml")
    v1, _ = build_world(con, pack, "seoul v1")
    rt = Router(con=con, run_id="calib", budget_usd=1.0)
    recs = anchor_records(A, n_anchors, seed=3)
    tgt = {r["source_ref"]: r.pop("_target") for r in recs}
    _, f = ops.run(con, v1, "fork", {"name": "v2 (정박 매핑용)"}); v2 = f["new_world_id"]
    _, ar = ops.run(con, v2, "anchor", {"records": recs, "grade": "[사]", "seed": 3})
    targets = {an["uuid"]: tgt[r["source_ref"]] for r, an in zip(recs, ar["anchors"])}
    v = view_name(v1); build_view(con, v1)
    ids = ",".join(repr(u) for u in targets)
    cells = {}
    for u, b, hg in con.execute(f"SELECT uuid, age_band, CASE WHEN housing_type='아파트' THEN '아파트' ELSE '비아파트' END FROM {v} WHERE uuid IN ({ids})").fetchall():
        cells.setdefault(f"{b} × {hg}", []).append(u)
    print(f"anchored {len(targets)} · cells {len(cells)} · {time.time()-t0:.0f}s", flush=True)
    base = {"cells": [{"uuids": us, "n": len(us), "label": lab} for lab, us in sorted(cells.items())], "seed": 31, "temperature": 0.8}
    R = {"ts": datetime.now().isoformat(timespec="seconds"), "n_respondents": len(targets), "A_marginal": A_marg, "targets": targets, "cells": cells,
         "v1_stat_impute_m1": wtvd({lab: c["dist"] for lab, c in Acells.items()}, Bcells), "conditions": {}}

    # E0 단일 질문
    r = run_sim(con, v1, {**base, "questions": [{"id": "q", "text": Q, "answer_schema": "choice", "options": OPTIONS}]}, rt)
    ans = {u: OPT2KEY[json.loads(a)["value"]] for u, a in con.execute("SELECT uuid, answer FROM sim_response WHERE run_id=?", [r["run_id"]]).fetchall()}
    R["conditions"]["E0 단일 질문"] = {"run_id": r["run_id"], "n_ok": r["n_ok"], "cost_usd": round(r["cost_usd"], 4), **evaluate(ans, cells, targets, Bcells, Acells, A_marg)}
    print("E0", R["conditions"]["E0 단일 질문"]["m1"], flush=True)

    # E4 2단 질문: 차 구매 계획(yes/no) → 종류
    r = run_sim(con, v1, {**base, "questions": TWO_STAGE}, rt)
    raw = {}
    for u, qid, a in con.execute("SELECT uuid, question_id, answer FROM sim_response WHERE run_id=?", [r["run_id"]]).fetchall():
        raw.setdefault(u, {})[qid] = json.loads(a)["value"]
    ans2 = {u: ("none" if d.get("q1") != "yes" else {"전기차": "ev", "하이브리드": "hybrid", "수소차": "hydrogen"}.get(d.get("q2"), "none")) for u, d in raw.items()}
    R["conditions"]["E4 2단 질문"] = {"run_id": r["run_id"], "n_ok": r["n_ok"], "cost_usd": round(r["cost_usd"], 4),
                                    "p_buy_any": round(sum(d.get("q1") == "yes" for d in raw.values()) / max(len(raw), 1), 4),
                                    "q2_given_buy": dist([{"전기차": "ev", "하이브리드": "hybrid", "수소차": "hydrogen"}.get(d.get("q2"), "none") for d in raw.values() if d.get("q1") == "yes"]),
                                    **evaluate(ans2, cells, targets, Bcells, Acells, A_marg)}
    print("E4", R["conditions"]["E4 2단 질문"]["m1"], flush=True)
    R["cost_usd"] = round(rt.total_cost, 4); R["seconds"] = round(time.time() - t0)
    (ROOT / "data/calibration.json").write_text(json.dumps(R, ensure_ascii=False, indent=1))
    print(f"done ${rt.total_cost:.3f} {R['seconds']}s")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1200)
