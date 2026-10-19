"""셀별·질문별 분포와 찬성률. 수치는 sim_response 집계에서만 나온다."""
import json
from collections import Counter

from .schemas import positive


def aggregate(con, run_id: str, spec: dict) -> dict:
    rows = con.execute("SELECT question_id, answer FROM sim_response WHERE run_id=?", [run_id]).fetchall()
    qtype = {q["id"]: q.get("answer_schema", "text") for q in spec["questions"]}
    out = {}
    for qid, ans in rows:
        a = json.loads(ans)
        cell = out.setdefault(str(a["cell"]), {}).setdefault(qid, {"n": 0, "dist": Counter(), "pos": 0.0})
        cell["n"] += 1
        cell["dist"][str(a["value"])] += 1
        p = positive(a["value"], qtype[qid])
        if p is not None:
            cell["pos"] += p
    for cell in out.values():
        for q in cell.values():
            n = q["n"]
            q["dist"] = {k: round(v / n, 3) for k, v in sorted(q["dist"].items())}
            q["pos_rate"] = round(q["pos"] / n, 3) if n else None
            del q["pos"]
    return out


def tvd(p: dict, q: dict) -> float:
    keys = set(p) | set(q)
    return round(0.5 * sum(abs(p.get(k, 0) - q.get(k, 0)) for k in keys), 4)
