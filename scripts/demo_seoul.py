"""데모 보고서: 서울 전기차 case를 실측이 쌓이는 순서로 세 번 돌려 한 보고서에 (§10 베이지안 보정 카드).

  v1 실측 없음 → 카드 'no_observed', 수준 주장 잠김.
  v2 서울서베이 A 150명 정박(구매 계획은 holdout) · v3 A 450명 추가(누적 600명).
  카드는 별도 평가 월드(서울서베이 B 1,000명, 적합에 안 씀)로 채점하고, 실측만 쓴 기준선에 지면 잠근다.
  .venv/bin/python -m scripts.demo_seoul   (LLM ≈ $0.3)
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / ".env")

from case import load_case  # noqa: E402
from evidence.seoul_survey import anchor_records, load  # noqa: E402
from llm import Router  # noqa: E402
from report.render import render_case_report, verify_claims  # noqa: E402
from tools.checks import run_chain  # noqa: E402
from world import ops  # noqa: E402
from world.domain import build_world, load_pack  # noqa: E402
from world.store import connect  # noqa: E402


def main(n_anchors=(150, 600), n_eval=1000):
    t0 = time.time()
    con = connect(":memory:")
    pack = load_pack(ROOT / "domains/ev_charging_seoul.yaml"); case = load_case(ROOT / "cases/ev_charging_seoul.yaml")
    v1, _ = build_world(con, pack, "서울 v1 (실측 없음)")
    rt = Router(con=con, run_id="demo_seoul", budget_usd=1.0)
    # 0) 근거 수집 에이전트: 빈 열을 공개 통계로 채우고 L0 정합성 카드 (보고서 첫 섹션)
    from tools.runner import run_tool
    approve = lambda view: {"approved": [s["attr"] for s in view["s0_collect"]["summary"] if s["passed"] and not s["conflicts"]],
                            "rationale": "러너 자동 결정(개발 루프): 검사 통과·모순 없음 후보만"}
    ag = run_tool(con, case, v1, "evidence_agent", rt, decisions={"g1_approve": approve})
    print("agent:", ag["status"], [(x["attr"], x["passed"], x.get("evidence_level")) for x in ag["outputs"]["s0_collect"]["summary"]], f"${rt.total_cost:.3f}", flush=True)
    c1 = run_chain(con, case, v1, rt, pack)
    if c1.get("error"):
        raise SystemExit(f"v1 chain: {c1['error']}")
    cards1 = c1["fgi"]["outputs"]["s1_run"]["calibration"]
    print("v1 cards:", [(c["question_id"], c["status"], c.get("reason")) for c in cards1], f"${rt.total_cost:.3f}", flush=True)

    df = load(); A, B = df[df.split == "A"], df[df.split == "B"]
    def records(frame, n, seed):
        recs = anchor_records(frame, n, seed=seed)
        for r in recs:
            r["observed"]["ev_plan"] = r.pop("_target")    # 채점 전용(holdout): 월드에 남고 응답자에게는 안 보임
        return recs
    _, f = ops.run(con, v1, "fork", {"name": "평가 (서울서베이 B 1,000명, 적합에 안 씀)"}); ev = f["new_world_id"]
    ops.run(con, ev, "anchor", {"records": records(B, n_eval, 5), "grade": "[사]", "seed": 5, "holdout": ["ev_plan"]})
    recs = records(A, n_anchors[-1], 3)
    prev, done = v1, 0
    for i, n in enumerate(n_anchors, start=2):
        _, f = ops.run(con, prev, "fork", {"name": f"서울 v{i} (서울서베이 A {n}명 정박)"}); vi = f["new_world_id"]
        ops.run(con, vi, "anchor", {"records": recs[done:n], "grade": "[사]", "seed": 3, "holdout": ["ev_plan"]})
        ch = run_chain(con, case, vi, rt, pack, evaluation_world_id=ev)
        if ch.get("error"):
            raise SystemExit(f"v{i} chain: {ch['error']}")
        for c in ch["fgi"]["outputs"]["s1_run"]["calibration"]:
            print(f"v{i} card:", c["question_id"], c["status"], "n_fit", c.get("n_fit"), "n_eval", c.get("n_evaluation"), "α", c.get("eb_alpha"),
                  "M1", c.get("m1_before"), "→", c.get("m1_after"), "기준선", c.get("baselines_m1"), "open", c.get("level1_open"), f"${rt.total_cost:.3f}", flush=True)
        prev, done = vi, n
    url = render_case_report(con, case)
    v = verify_claims(con, case)
    print("report", url, v, f"${rt.total_cost:.3f} {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
