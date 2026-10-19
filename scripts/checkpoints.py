"""체크포인트 러너: docs/plan-impl.md §4의 C1~C7·S1~S4를 판정하고 docs/status.md 를 쓴다.

  .venv/bin/python scripts/checkpoints.py            # fast만 (LLM 없음)
  .venv/bin/python scripts/checkpoints.py --live     # Token Factory/Tavily 호출 포함
  .venv/bin/python scripts/checkpoints.py --only S2  # 하나만

각 체크포인트는 {"status": passed|failed|blocked|manual, "evidence": str, "metrics": dict} 를 돌려준다.
blocked = 외부 요인(키·데이터·사용자 결정). 개발 루프는 가장 앞의 failed 를 잡는다.
"""
import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / ".env")

from case import check_decision_relevance, evidence_coverage, load_case  # noqa: E402
from world.domain import build_world, load_pack  # noqa: E402
from world.store import connect  # noqa: E402

CASE_PATH = ROOT / "cases/ev_charging.yaml"
STATUS_MD = ROOT / "docs/status.md"
STATUS_JSON = ROOT / "data/status.json"
LIVE_BUDGET_USD = 0.30


@dataclass
class Ctx:
    live: bool
    con: object
    case: object
    pack: dict
    world_id: str
    cache: dict = None

    def router(self, run_id=None, budget=None):
        from llm import Router
        return Router(con=self.con, run_id=run_id, budget_usd=budget or LIVE_BUDGET_USD)


def passed(evidence, **metrics):
    return {"status": "passed", "evidence": evidence, "metrics": metrics}


def failed(evidence, **metrics):
    return {"status": "failed", "evidence": evidence, "metrics": metrics}


def blocked(evidence, **metrics):
    return {"status": "blocked", "evidence": evidence, "metrics": metrics}


def manual(evidence, **metrics):
    return {"status": "manual", "evidence": evidence, "metrics": metrics}


def needs_live(ctx):
    return None if ctx.live else blocked("--live 에서만 실행 (LLM 호출)")


# ---------------- C1~C7 ----------------

def c1(ctx):
    r = check_decision_relevance(ctx.case)
    if not r["passed"]:
        return failed("; ".join(r["problems"]))
    return passed(f"조사 질문 {r['n_rq']}개 모두 결과별 결정이 갈림. 선택지 {r['n_options']}개", n_rq=r["n_rq"])


def pack_beliefs(ctx):
    b = {x["id"]: x for x in ctx.pack.get("beliefs", [])}
    b.update({e.id: {"grade": e.grade, "source_ref": e.source_ref} for e in ctx.case.environment})
    return b


def c2(ctx):
    r = evidence_coverage(ctx.case, pack_beliefs(ctx))
    if not r["passed"]:
        return failed("; ".join(r["problems"]))
    return passed(f"정보 필요 {len(r['rows'])}개: 근거 {r['n_evidence']} · 검색 대기 {r['n_search']} · 추가 조사 {r['n_primary']}",
                  n_evidence=r["n_evidence"], n_search=r["n_search"], n_primary=r["n_primary"])


def c2_live(ctx):
    if (b := needs_live(ctx)):
        return b
    if not os.environ.get("TAVILY_API_KEY"):
        return blocked("TAVILY_API_KEY 없음 (.env) — 사용자가 키를 넣으면 해제")
    try:
        from evidence.tavily import search_to_beliefs
    except ImportError:
        return failed("미구현: evidence/tavily.py")
    queries = [ref.search for ref in ctx.case.evidence_map.values() if ref.search]
    beliefs = search_to_beliefs(ctx.con, ctx.world_id, queries[:1])
    bad = [b for b in beliefs if not b["source_ref"].startswith("http")]
    if not beliefs or bad:
        return failed(f"[공] belief {len(beliefs)}개, URL 없는 것 {len(bad)}개")
    return passed(f"Tavily → [공] belief {len(beliefs)}개, 전부 URL", n=len(beliefs))


def c3(ctx):
    if (b := needs_live(ctx)):
        return b
    try:
        from tools.runner import run_tool
        from tools.checks import verify_gate_isolation
    except ImportError:
        return failed("미구현: tools/runner.py, tools/manifests/stp.yaml")
    rt = ctx.router(run_id="cp_c3")
    run = run_tool(ctx.con, ctx.case, ctx.world_id, "stp", rt, decisions={
        "g1_adopt_axes": {"axes": ctx.pack["axes_default"], "rationale": "러너 자동 결정(개발 루프): 팩 기본 축"},
        "g2_target": lambda view: {"cell": view["s2_cells"]["largest"], "rationale": "러너 자동 결정: 규모 최대 셀"}})
    iso = verify_gate_isolation(ctx.con, run["run_id"])
    if run["status"] != "done":
        return failed(f"status={run['status']} {run.get('error','')}", cost_usd=rt.total_cost)
    if not iso["passed"]:
        return failed("게이트 격리 위반: " + "; ".join(iso["problems"]), cost_usd=rt.total_cost)
    gs = run["artifacts"][-1]["grade_summary"]
    return passed(f"STP 완주: 게이트 {run['n_gates']}회 정지·재개, 산출물 {len(run['artifacts'])}개, 등급 {gs}",
                  cost_usd=round(rt.total_cost, 4), n_calls=len(rt.calls))


def c3b(ctx):
    if (b := needs_live(ctx)):
        return b
    try:
        from tools.runner import run_tool
        from tools.checks import verify_scenario_axes
    except ImportError:
        return failed("미구현: tools/manifests/scenario.yaml")
    rt = ctx.router(run_id="cp_c3b")
    def auto_axes(view):
        drivers = view["s1_pestel"]["drivers"]
        ok = {d["id"] for d in drivers if d["impact"] == "H" and d["uncertainty"] == "H" and not d["firm_controllable"]}
        pair = next((p for p in view["s2_axis_pairs"]["pairs"] if p["axis1_driver"] in ok and p["axis2_driver"] in ok
                     and p["axis1_driver"] != p["axis2_driver"]), view["s2_axis_pairs"]["pairs"][0])
        return {"axis1_driver": pair["axis1_driver"], "axis1_poles": pair["axis1_poles"], "axis2_driver": pair["axis2_driver"],
                "axis2_poles": pair["axis2_poles"], "adopted_drivers": drivers, "rationale": "러너 자동 결정: 첫 유효(H·H·외생) 쌍"}
    run = run_tool(ctx.con, ctx.case, ctx.world_id, "scenario", rt, decisions={"g1_axes": auto_axes})
    if run["status"] == "waiting_evidence":
        return blocked(run["error"], cost_usd=round(rt.total_cost, 4))
    if run["status"] != "done":
        return failed(f"status={run['status']} {run.get('error','')}", cost_usd=rt.total_cost)
    v = verify_scenario_axes(ctx.case, run)
    if not v["passed"]:
        return failed("; ".join(v["problems"]), cost_usd=rt.total_cost)
    return passed(f"시나리오 2×2 완주: 동인 {v['n_drivers']} · 축 {v['axes']} · 기본 트렌드 공통 반영", cost_usd=round(rt.total_cost, 4))


def c4(ctx):
    if (b := needs_live(ctx)):
        return b
    try:
        from tools.runner import run_tool
        from tools.checks import verify_chain
    except ImportError:
        return failed("미구현: tools/manifests/persona_bum.yaml, fgi.yaml + 연결 검사")
    rt = ctx.router(run_id="cp_c4")
    res = verify_chain(ctx.con, ctx.case, ctx.world_id, rt, ctx.pack)
    ctx.cache["chain"] = res["chain"]
    if not res["passed"]:
        return failed("; ".join(res["problems"]), cost_usd=round(rt.total_cost, 4))
    return passed(res["evidence"], cost_usd=round(rt.total_cost, 4), **res["metrics"])


def c5(ctx):
    if (b := needs_live(ctx)):
        return b
    try:
        from tools.checks import verify_research_plan
    except ImportError:
        return failed("미구현: tools/manifests/research_plan.yaml")
    if "chain" not in ctx.cache or ctx.cache["chain"].get("error"):
        return blocked("C4 체인(STP→BUM→FGI)이 먼저 통과해야 함")
    rt = ctx.router(run_id="cp_c5")
    res = verify_research_plan(ctx.con, ctx.case, ctx.world_id, rt, ctx.cache["chain"])
    ctx.cache["plan"] = res["run"]
    return (passed if res["passed"] else failed)(res["evidence"], cost_usd=round(rt.total_cost, 4))


def c6(ctx):
    if (b := needs_live(ctx)):
        return b
    try:
        from evidence.checks import verify_rejudge
    except ImportError:
        return failed("미구현: evidence/ingest.py + 재판단 검사")
    if "chain" not in ctx.cache or ctx.cache["chain"].get("error"):
        return blocked("C4 체인(STP→BUM→FGI)이 먼저 통과해야 함")
    rt = ctx.router(run_id="cp_c6")
    res = verify_rejudge(ctx.con, ctx.case, ctx.world_id, rt, ctx.cache["chain"], ctx.pack)
    return (passed if res["passed"] else failed)(res["evidence"], cost_usd=round(rt.total_cost, 4), **res.get("metrics", {}))


def c7(ctx):
    try:
        from report.render import render_case_report, verify_claims
    except ImportError:
        return failed("미구현: report/render.py")
    if "chain" not in ctx.cache:
        return blocked("--live 에서 C4 이후 실행(산출물이 있어야 보고서 생성)")
    out = render_case_report(ctx.con, ctx.case)
    v = verify_claims(ctx.con, ctx.case)
    if not v["passed"]:
        return failed("; ".join(v["problems"]))
    return passed(f"보고서 {out} · claim {v['n_claims']}개 전부 등급·출처", n_claims=v["n_claims"])


def c8(ctx):
    """근거 수집 에이전트: 팩 agent_attributes → 검색·추출·검사·게이트·임퓨트. 검사 실패 후보 커밋 0건, 통과 후보는 [공] belief(URL) + [도] 열."""
    if (b := needs_live(ctx)):
        return b
    if not os.environ.get("TAVILY_API_KEY"):
        return blocked("TAVILY_API_KEY 없음")
    from tools.runner import run_tool
    from world.store import active_attrs
    rt = ctx.router(run_id="cp_c8")
    approve = lambda view: {"approved": [x["attr"] for x in view["s0_collect"]["summary"] if x["passed"] and not x["conflicts"]],
                            "rationale": "러너 자동 결정(개발 루프): 검사 통과·모순 없음 후보만"}
    res = run_tool(ctx.con, ctx.case, ctx.world_id, "evidence_agent", rt, decisions={"g1_approve": approve})
    if res["status"] != "done":
        return failed(f"evidence_agent {res['status']}: {res.get('error')}", cost_usd=round(rt.total_cost, 4))
    summ = res["outputs"]["s0_collect"]["summary"]; applied = res["outputs"]["s1_apply"]["applied"]
    names = {a["name"]: a for a in active_attrs(ctx.con, ctx.world_id)}
    problems = []
    for x in summ:
        if not x["passed"] and x["attr"] in names and names[x["attr"]]["method"] == "stat_conditional":
            problems.append(f"{x['attr']}: 검사 실패 후보가 커밋됨")
    for a in applied:
        if a["status"] != "committed":
            problems.append(f"{a['attr']}: 연산 {a['status']}")
        url = ctx.con.execute("SELECT source_ref FROM belief WHERE belief_id=?", [a["belief_id"]]).fetchone()[0]
        if not str(url).startswith("http"):
            problems.append(f"{a['attr']}: belief 출처가 URL 아님")
    if not applied:
        problems.append("채워진 열 0개: " + "; ".join(f"{x['attr']}: {'; '.join(x['problems'])}" for x in summ))
    m = dict(cost_usd=round(rt.total_cost, 4), n_candidates=len(summ), n_applied=len(applied),
             levels=[x.get("evidence_level") for x in summ], coverage=[x["coverage"] for x in summ])
    ctx.cache["agent"] = res
    if problems:
        return failed("; ".join(problems)[:400], **m)
    return passed(f"에이전트 {len(applied)}열 채움({[x.get('evidence_level') for x in summ]}) · 출처 URL·검사 통과 · 불합격 커밋 0", **m)


# ---------------- S1~S4 ----------------

def s1(ctx):
    if (b := needs_live(ctx)):
        return b
    from pydantic import BaseModel

    class Ping(BaseModel):
        ok: bool
        note: str

    rt = ctx.router(run_id="cp_s1")
    obj, rec = rt.call("respondent", [{"role": "user", "content": 'JSON만: {"ok": true, "note": "한 단어"}'}], schema=Ping, max_tokens=60, thinking=False)
    row = ctx.con.execute("SELECT count(*), sum(cost_usd), sum(prompt_tokens+completion_tokens) FROM llm_call WHERE run_id='cp_s1'").fetchone()
    if not (obj is not None and row[0] == 1 and abs(row[1] - rt.total_cost) < 1e-9):
        return failed(f"호출 기록 불일치: db={row} router={rt.total_cost}")
    return passed(f"호출 1회 기록: {rec.model} {rec.prompt_tokens}+{rec.completion_tokens} tok ${rec.cost_usd:.6f} {rec.latency_s}s, DB 합계 일치",
                  cost_usd=rec.cost_usd, latency_s=rec.latency_s)


def s2(ctx):
    if (b := needs_live(ctx)):
        return b
    try:
        from sim.run import run_sim
    except ImportError:
        return failed("미구현: sim/render.py run.py")
    rt = ctx.router(run_id="cp_s2")
    spec = {"cells": [{"filter": ctx.case.world.market_scope or "TRUE", "n": 100}],
            "questions": [{"id": "q1", "text": "2년 안에 전기차를 살 의향이 있습니까?", "answer_schema": "likert5"}],
            "seed": 1}
    t = time.time()
    run = run_sim(ctx.con, ctx.world_id, spec, rt)
    el = time.time() - t
    ok = run["n_ok"] / max(run["n_requested"], 1)
    m = dict(seconds=round(el, 1), cost_usd=round(rt.total_cost, 4), schema_pass=round(ok, 3), n_ok=run["n_ok"])
    if el > 60 or rt.total_cost > 0.05 or ok < 1.0:
        return failed(f"100명: {el:.1f}s ${rt.total_cost:.4f} 통과율 {ok:.1%} (목표 ≤60s ≤$0.05 100%)", **m)
    return passed(f"100명: {el:.1f}s ${rt.total_cost:.4f} 통과율 {ok:.0%}", **m)


def s3(ctx):
    if (b := needs_live(ctx)):
        return b
    try:
        from sim.calibrate import conformity_m4
    except ImportError:
        return failed("미구현: sim/calibrate.py (M4 동조율)")
    rt = ctx.router(run_id="cp_s3")
    r = conformity_m4(ctx.con, ctx.world_id, rt, n=150,  # n=40은 표준오차 ≈0.07로 판정 불가. 실행 간 편차 0.02~0.18 관측(9/14)
                      filter=ctx.case.world.market_scope or "TRUE",
                      question="2년 안에 전기차를 살 의향이 있습니까?")
    m = dict(m4=r["m4"], ci95=r["ci95"], cost_usd=round(rt.total_cost, 4))
    if r["m4"] >= 0.10:
        return failed(f"M4 동조율 {r['m4']:.3f}±{r['ci95']:.3f} (목표 < 0.10): 프레임 A {r['p_a']:.2f} vs B {r['p_b']:.2f}", **m)
    return passed(f"M4 동조율 {r['m4']:.3f}±{r['ci95']:.3f}: 프레임 A {r['p_a']:.2f} vs B {r['p_b']:.2f}", **m)


def s4(ctx):
    url = os.environ.get("DEMO_URL")
    if not url:
        return manual("배포 전. DEMO_URL 설정 후 공개 URL에서 C3 재현을 확인")
    return manual(f"{url} 에서 C3 재현 여부는 사용자가 확인")


CHECKPOINTS = [
    ("C1", "M1", "의사결정 정의", "fast", c1),
    ("C2", "M1", "근거 준비(커버리지)", "fast", c2),
    ("S1", "M1", "라우터·호출 로그", "live", s1),
    ("S2", "M2", "시뮬레이션 100명", "live", s2),
    ("S3", "M2", "보정 카드 M4", "live", s3),
    ("C3", "M3", "툴 하나 완주(STP)", "live", c3),
    ("C3b", "M4", "두 번째 흐름(시나리오)", "live", c3b),
    ("C4", "M4", "툴 사이 연결(페르소나→BUM→FGI)", "live", c4),
    ("C5", "M4", "조사로 되돌아가기", "live", c5),
    ("C6", "M5", "새 근거로 재판단", "live", c6),
    ("C7", "M5", "공유 가능한 결과", "fast", c7),
    ("C2L", "M5", "Tavily → [공] 근거", "live", c2_live),
    ("C8", "M5", "근거 수집 에이전트", "live", c8),
    ("S4", "M6", "배포", "manual", s4),
]

ICON = {"passed": "✅", "failed": "❌", "blocked": "⏸", "manual": "👤"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--only", help="쉼표 구분 ID")
    ap.add_argument("--sample", type=int, default=50000, help="테스트용 L0 축소본 행 수")
    ap.add_argument("--no-publish", action="store_true", help="status.md 를 쓰지 않음")
    a = ap.parse_args()

    con = connect(":memory:", sample_rows=a.sample)
    case = load_case(CASE_PATH)
    pack = load_pack(ROOT / case.world.pack)
    world_id, _ = build_world(con, pack)
    ctx = Ctx(a.live, con, case, pack, world_id, cache={})
    only = set(a.only.split(",")) if a.only else None

    results, prev = [], {}
    if STATUS_JSON.exists():
        prev = {r["id"]: r for r in json.loads(STATUS_JSON.read_text()).get("results", [])}
    for cid, ms, title, kind, fn in CHECKPOINTS:
        if only and cid not in only:
            if cid in prev:
                results.append(prev[cid])
            continue
        t = time.time()
        try:
            r = fn(ctx)
        except Exception as e:
            r = failed(f"예외 {type(e).__name__}: {str(e)[:200]}")
            r["trace"] = traceback.format_exc()[-1500:]
        r.update(id=cid, milestone=ms, title=title, kind=kind, seconds=round(time.time() - t, 1),
                 ts=datetime.now().isoformat(timespec="seconds"))
        results.append(r)
        print(f"{ICON[r['status']]} {cid:4s} {title:24s} {r['evidence'][:110]}")

    cost = con.execute("SELECT COALESCE(sum(cost_usd),0) FROM llm_call").fetchone()[0]
    counts = {k: sum(r["status"] == k for r in results) for k in ICON}
    if not a.no_publish:
        STATUS_JSON.parent.mkdir(exist_ok=True)
        STATUS_JSON.write_text(json.dumps({"ts": datetime.now().isoformat(timespec="seconds"), "live": a.live,
                                           "cost_usd": cost, "results": results}, ensure_ascii=False, indent=1))
        STATUS_MD.write_text(render_md(results, counts, cost, a.live), encoding="utf-8")
    print(f"\n{counts}  live_cost=${cost:.4f}")
    nxt = next((r for r in results if r["status"] == "failed"), None)
    print(f"다음 작업: {nxt['id']} {nxt['title']} — {nxt['evidence'][:100]}" if nxt else "다음 작업: failed 없음")


def render_md(results, counts, cost, live):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"# 체크포인트 상태", "",
             f"갱신 {ts} · 모드 {'live' if live else 'fast'} · 이번 실행 LLM 비용 ${cost:.4f} · "
             f"✅ {counts['passed']} ❌ {counts['failed']} ⏸ {counts['blocked']} 👤 {counts['manual']}", "",
             "기준은 [구현 계획 §4](/plan.html). 루프는 가장 앞의 ❌를 잡는다. ⏸는 외부 요인(키·--live), 👤는 사용자 판단.", "",
             "| | ID | 마일스톤 | 결과물 | 판정·근거 | 수치 | 시각 |", "|---|---|---|---|---|---|---|"]
    for r in results:
        m = " ".join(f"{k}={v}" for k, v in r.get("metrics", {}).items())
        lines.append(f"| {ICON[r['status']]} | {r['id']} | {r['milestone']} | {r['title']} | {r['evidence']} | {m} | {r['ts'][5:16]} |")
    todo = [r for r in results if r["status"] in ("blocked", "manual") and "live" not in r["evidence"]]
    if todo:
        lines += ["", "## 결정·입력 필요 (사용자)", ""] + [f"- **{r['id']}** {r['title']}: {r['evidence']}" for r in todo]
    nxt = next((r for r in results if r["status"] == "failed"), None)
    lines += ["", "## 루프 다음 작업", "", f"- {nxt['id']} {nxt['title']}: {nxt['evidence']}" if nxt else "- failed 없음"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
