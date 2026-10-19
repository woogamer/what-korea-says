"""시뮬레이션 실행: 스펙 → 셀별 응답자 샘플 → 응답자마다 독립 요청(Nano) → sim_response 저장 → 집계.

spec = {cells:[{filter, n, label?}], questions:[{id, text, answer_schema, options?}], seed, framing?}
응답자 간 컨텍스트 공유 없음: 요청마다 messages 는 [system(응답자 렌더링), user(질문)] 뿐이다.
"""
import asyncio
import json
from datetime import datetime

from llm import ROLES
from world.store import new_id

from .aggregate import aggregate
from .render import render_respondent, sample_rows
from .schemas import response_model


def question_block(questions: list[dict], framing: str | None = None) -> str:
    lines = [framing.strip()] if framing else []
    for q in questions:
        t = q.get("answer_schema", "text")
        hint = {"likert5": "1~5 정수(1 전혀 아니다, 5 매우 그렇다)", "yes_no": "yes 또는 no",
                "number": "숫자", "text": "한 문장"}.get(t, "")
        if t == "choice":
            hint = "다음 중 하나: " + " / ".join(q["options"])
        lines.append(f"{q['id']}: {q['text']} ({hint})")
    lines.append("각 질문에 답하고, <id>_reason 에 당신의 상황에 근거한 한 문장 이유를 쓰세요. JSON만 출력.")
    return "\n".join(lines)


def run_sim(con, world_id, spec: dict, router, concurrency: int = 32) -> dict:
    run_id = new_id("sim")
    seed = int(spec.get("seed", 0))
    questions = spec["questions"]
    Model = response_model(questions)
    con.execute("INSERT INTO sim_run VALUES (?,?,?,?,?,?,?,?)",
                [run_id, world_id, json.dumps(spec, ensure_ascii=False), "running", 0, 0.0, None, datetime.now()])
    respondents = []  # (cell_idx, row)
    for ci, cell in enumerate(spec["cells"]):
        if cell.get("uuids"):
            from world.store import build_view, view_name
            v = view_name(world_id); build_view(con, world_id)
            ids = ",".join(repr(u) for u in cell["uuids"])
            rows = con.execute(f"SELECT * FROM {v} WHERE uuid IN ({ids})").df().to_dict("records")
        else:
            rows = sample_rows(con, world_id, cell.get("filter"), cell["n"], seed)
        for row in rows:
            respondents.append((ci, row))
    max_tokens = (1500 if ROLES["respondent"][1] else 120) + 90 * len(questions)
    users = [question_block(questions, c.get("framing", spec.get("framing"))) for c in spec["cells"]]
    jobs = [{"role": "respondent", "schema": Model, "max_tokens": max_tokens,
             "temperature": float(spec.get("temperature", 0.8)), "tag": f"{run_id}:{ci}", "seed": seed + i,
             "messages": [{"role": "system", "content": render_respondent(con, world_id, row)},
                          {"role": "user", "content": users[ci]}]}
            for i, (ci, row) in enumerate(respondents)]
    prev_run = router.run_id
    router.run_id = run_id
    try:
        results = asyncio.run(router.gather(jobs, concurrency=concurrency))
    finally:
        router.run_id = prev_run
    n_ok, rows = 0, []
    for (ci, row), (obj, rec) in zip(respondents, results):
        if obj is None:
            continue
        n_ok += 1
        d = obj.model_dump()
        for q in questions:
            rows.append([run_id, row["uuid"], q["id"],
                         json.dumps({"value": d[q["id"]], "reason": d.get(f"{q['id']}_reason"), "cell": ci}, ensure_ascii=False),
                         rec.model, rec.prompt_tokens + rec.completion_tokens])
    if rows:
        con.executemany("INSERT INTO sim_response VALUES (?,?,?,?,?,?)", rows)
    cost = sum(r.cost_usd for _, r in results)
    agg = aggregate(con, run_id, spec)
    con.execute("UPDATE sim_run SET status='done', n_respondents=?, cost_usd=?, calibration=? WHERE run_id=?",
                [n_ok, cost, json.dumps(agg, ensure_ascii=False), run_id])
    return {"run_id": run_id, "n_requested": len(respondents), "n_ok": n_ok, "cost_usd": cost,
            "errors": [r.error for _, r in results if not r.ok][:5], "aggregate": agg}
