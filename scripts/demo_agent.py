"""근거 수집 에이전트 데모·검증: 서울 v1의 빈 열(commute_car)을 에이전트가 공개 통계로 채우고, 서울서베이 B의 같은 변수와 대조한다.

  1) 팩에 reference_distributions(서울서베이 B 가중 주변분포: 성·연령대·주거·학력·혼인)를 넣는다 → L0 정합성 카드 재료
  2) evidence_agent 툴 실행(게이트: 검사 통과·모순 없음인 후보만 자동 승인 — 개발 루프)
  3) 검증: 임퓨트된 commute_car 의 연령대별 분포 vs B(commute_mode ∈ {car, car_mixed}) TVD
  .venv/bin/python -m scripts.demo_agent   (Tavily 수 회 + LLM ≈ $0.05)
"""
import json
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / ".env")

from case import load_case  # noqa: E402
from evidence.seoul_survey import SOURCE_URL, load  # noqa: E402
from llm import Router  # noqa: E402
from sim.aggregate import tvd  # noqa: E402
from tools.runner import run_tool  # noqa: E402
from world.domain import build_world, load_pack  # noqa: E402
from world.store import build_view, connect, view_name  # noqa: E402

PACK = ROOT / "domains/ev_charging_seoul.yaml"
REF_COLS = ["sex", "age_band", "housing_type", "education_level", "marital_status"]


def ensure_reference(B):
    pack = yaml.safe_load(PACK.read_text(encoding="utf-8"))
    if "reference_distributions" in pack:
        return
    axes = {}
    for c in REF_COLS:
        g = B.dropna(subset=[c]).groupby(c)["w"].sum(); axes[c] = {str(k): round(float(v), 4) for k, v in (g / g.sum()).items()}
    pack["reference_distributions"] = {"source_ref": SOURCE_URL, "sample_condition": "서울서베이 2024 가구원 split B(50%), 만 20세+, 가중치 wtb2", "axes": axes}
    PACK.write_text(yaml.safe_dump(pack, allow_unicode=True, sort_keys=False, width=200), encoding="utf-8")
    print("pack: reference_distributions 추가", list(axes))


def main():
    t0 = time.time()
    df = load(); B = df[df.split == "B"]
    ensure_reference(B)
    con = connect(":memory:")
    pack = load_pack(PACK); case = load_case(ROOT / "cases/ev_charging_seoul.yaml")
    v1, _ = build_world(con, pack, "서울 v1")
    rt = Router(con=con, run_id="demo_agent", budget_usd=0.5)
    approve = lambda view: {"approved": [s["attr"] for s in view["s0_collect"]["summary"] if s["passed"] and not s["conflicts"]],
                            "rationale": "러너 자동 결정(개발 루프): 검사 통과·모순 없음 후보만"}
    res = run_tool(con, case, v1, "evidence_agent", rt, decisions={"g1_approve": approve})
    out = res["outputs"]
    print("status", res["status"], res.get("error"))
    for s in out["s0_collect"]["summary"]:
        print("candidate", json.dumps(s, ensure_ascii=False)[:600])
    for step in out["s0_collect"]["candidates"][0]["trace"]:
        print("  trace", json.dumps(step, ensure_ascii=False)[:300])
    print("applied", out.get("s1_apply", {}).get("applied"))
    # 검증: 임퓨트 vs B
    v = view_name(v1); build_view(con, v1)
    rows = con.execute(f"SELECT age_band, commute_car, CAST(sum(w) AS DOUBLE) FROM {v} WHERE province='서울' AND age>=20 AND commute_car IS NOT NULL GROUP BY 1,2").fetchall() if out.get("s1_apply", {}).get("applied") else []
    imp = {}
    for b, val, w in rows:
        imp.setdefault(b, {})[val] = w
    Bc = B.dropna(subset=["commute_mode"]).copy(); Bc["car"] = Bc.commute_mode.isin(["car", "car_mixed"]).map({True: "car", False: "other"})
    num = den = 0.0; per = {}
    for b, g in Bc.groupby("age_band"):
        ref = (g.groupby("car")["w"].sum() / g.w.sum()).to_dict()
        if b in imp:
            tot = sum(imp[b].values()); d = {k: v / tot for k, v in imp[b].items()}
            per[b] = {"imputed_car": round(d.get("car", 0), 3), "survey_car": round(ref.get("car", 0), 3), "tvd": tvd(d, ref)}
            num += per[b]["tvd"] * len(g); den += len(g)
    m = round(num / den, 4) if den else None
    print("validation vs B: weighted TVD", m, json.dumps(per, ensure_ascii=False))
    card = out.get("s2_l0_card", {})
    print("L0 card:", [(a["col"], a["tvd"], a["worst_value"], a["l0_share"], a["ref_share"]) for a in card.get("axes", [])])
    c0 = out["s0_collect"]["candidates"][0]
    print("table:", json.dumps({k: c0["table"][k] for k in ("rows", "quote", "note", "sample_condition", "n")}, ensure_ascii=False)[:1200])
    R = {"status": res["status"], "summary": out["s0_collect"]["summary"], "trace": c0["trace"], "table": c0["table"], "checks": c0["checks"],
         "applied": out.get("s1_apply", {}).get("applied"), "validation_tvd": m, "validation_cells": per, "l0_card": card,
         "cost_usd": round(rt.total_cost, 4), "seconds": round(time.time() - t0)}
    (ROOT / "data/demo_agent.json").write_text(json.dumps(R, ensure_ascii=False, indent=1))
    print(f"${rt.total_cost:.3f} {R['seconds']}s")


if __name__ == "__main__":
    main()
