"""Evidence Ingest: 문서 → [Super] 명제 추출 → [규칙+Ultra] 모순 검사 → ★승인 → 커밋(belief + anchor + reweight).

승인 없이는 belief 가 committed 로 못 간다. 응답자 표는 LLM이 옮겨 적지 않고 규칙으로 파싱한다.
"""
import re
from collections import Counter

from pydantic import BaseModel, Field

from world import ops, reconcile
from world.store import add_belief


class Prop(BaseModel):
    text: str = Field(description="문서에 적힌 사실 한 문장")
    kind: str = Field(description="count(절대 수) / rate(전환율) / share(비율) / average / other")
    stage_from: str | None = Field(default=None, description="rate 일 때 분모 단계 이름(문서 표기 그대로)")
    stage_to: str | None = Field(default=None, description="count 는 그 단계 이름, rate 는 분자 단계 이름")
    value: float | None = Field(default=None, description="수치. 비율은 0~1")
    n: int | None = Field(default=None, description="표본 수(있으면)")
    sample_condition: str | None = Field(default=None, description="표본이 누구인지·편향 방향. 문서에 없으면 null")
    grade_candidate: str = Field(description="[사] 실측 / [공] 공개통계(URL 있을 때만) / [현] / [가]")
    source_span: str = Field(description="원문 발췌")


class Props(BaseModel):
    propositions: list[Prop]


class Conflicts(BaseModel):
    conflicts: list[str] = Field(description="서로 성립할 수 없는 명제 쌍과 이유. 없으면 빈 목록")


def extract(router, text: str) -> list[Prop]:
    msgs = [{"role": "system", "content": "너는 조사 문서에서 명제를 구조화하는 분석가다. 문서에 있는 수치만 옮긴다. 표(| 로 시작하는 줄)는 무시한다. JSON만 출력."},
            {"role": "user", "content": f"문서:\n{text}\n\n수치가 있는 문장을 전부 명제로 옮겨라. 퍼널의 절대 수는 kind=count 에 stage_to=단계 이름, 전환율은 kind=rate 에 stage_from/stage_to. "
                                        "표본 조건은 문서에 명시된 경우에만 적고, 없으면 null 로 둔다(만들지 말 것)."}]
    obj, rec = router.call("extract", msgs, schema=Props, max_tokens=9000, tag="ingest:extract")
    return obj.propositions if obj else []


def parse_table(text: str) -> list[dict]:
    rows, header = [], None
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if header is None:
            header = cells; continue
        if set(cells[0]) <= set("-: "):
            continue
        rows.append(dict(zip(header, cells)))
    return rows


def rule_conflicts(props: list[Prop]) -> list[dict]:
    """분모 검사(퍼널 절대 수 vs 전환율) + 표본조건 누락([사]인데 sample_condition 없음)."""
    out = []
    counts = {p.stage_to: p.value for p in props if p.kind == "count" and p.stage_to and p.value is not None}

    def find(name):
        for k, v in counts.items():
            if name and (name in k or k in name):
                return v
        return None
    for p in props:
        if p.kind == "rate" and p.stage_from and p.stage_to and p.value is not None:
            a, b = find(p.stage_from), find(p.stage_to)
            if a and b:
                r = reconcile.denominator({"from": a, "to": b}, [{"from": "from", "to": "to", "rate": p.value}])
                if not r["passed"]:
                    out.append({"type": "denominator", "proposition": p.text,
                                "detail": f"{p.stage_from} {a:.0f} → {p.stage_to} {b:.0f} 이면 전환율 {b/a:.0%}, 주장 {p.value:.0%}"})
        if p.grade_candidate.startswith("[사]") and not p.sample_condition:
            out.append({"type": "sample_condition_missing", "proposition": p.text, "detail": "실측인데 표본 조건(누구를, 어떤 편향으로) 없음"})
    return out


def llm_conflicts(router, props: list[Prop], existing: list[dict]) -> list[str]:
    msgs = [{"role": "system", "content": "너는 회의적인 검토자다. 같은 분모에서 동시에 성립할 수 없는 수치, 서로 모순되는 주장만 찾는다. JSON만 출력."},
            {"role": "user", "content": f"새 명제: {[p.model_dump() for p in props]}\n기존 belief: {existing}\n\n모순 쌍과 이유를 적어라. 없으면 빈 목록."}]
    obj, rec = router.call("conflict", msgs, schema=Conflicts, max_tokens=3000, tag="ingest:conflict")
    return obj.conflicts if obj else []


def ingest(con, world_id, text: str, router, existing_beliefs: list[dict] | None = None, source_grade: str = "[사]") -> dict:
    """추출 → 모순 → 제안. 커밋은 하지 않는다(승인 게이트).

    source_grade: 문서 종류는 분석자가 선언한다(설문·CRM = [사], 공개 통계 = [공]). LLM 의 grade_candidate 는 [공]+URL 일 때만 우선.
    """
    props = extract(router, text)
    for p in props:
        if not (p.grade_candidate == "[공]" and "http" in p.source_span):
            p.grade_candidate = source_grade
    rules = rule_conflicts(props)
    llm = llm_conflicts(router, props, existing_beliefs or [])
    flagged = {c["proposition"] for c in rules}
    table = parse_table(text)
    return {"propositions": [p.model_dump() for p in props], "conflicts": rules, "llm_conflicts": llm,
            "flagged": sorted(flagged), "table_rows": table,
            "proposed_ops": _propose_ops(table)}


def _propose_ops(table: list[dict]) -> list[dict]:
    if not table:
        return []
    records = [{"source_ref": r["id"], "source_type": "survey",
                "fields": {"sex": r["sex"], "age": int(r["age"]), "province": r["province"], "housing_type": r["housing_type"]},
                "observed": {k: r[k] for k in r if k not in ("id", "sex", "age", "province", "housing_type")}}
               for r in table]
    band = lambda a: "19-29" if a < 30 else "30-39" if a < 40 else "40-49" if a < 50 else "50-59" if a < 60 else "60-69" if a < 70 else "70+"
    dist = Counter(band(int(r["age"])) for r in table)
    n = sum(dist.values())
    return [{"op": "anchor", "params": {"records": records, "grade": "[사]"}},
            {"op": "reweight", "params": {"targets": {"age_band": {k: round(v / n, 4) for k, v in dist.items()}},
                                          "note": "접점 도달층 가중치(응답자 연령 분포). filter 는 커밋 시 타겟 셀"}}]


def commit(con, world_id, ingested: dict, approved: list[str], reweight_filter: str | None = None, source_ref: str = "survey") -> dict:
    """승인된 명제만 belief committed. 제안 연산 실행."""
    beliefs = []
    for p in ingested["propositions"]:
        if p["text"] not in approved:
            continue
        g = p["grade_candidate"] if p["grade_candidate"] in ("[사]", "[현]", "[가]") else "[사]"
        beliefs.append(add_belief(con, world_id, p["text"], g, source_ref, variables={"kind": p["kind"], "value": p["value"]},
                                  n=p["n"], sample_condition=p["sample_condition"]))
    results = []
    for o in ingested["proposed_ops"]:
        params = dict(o["params"]); params.pop("note", None)
        if o["op"] == "reweight" and reweight_filter:
            params["filter"] = reweight_filter
        oid, r = ops.run(con, world_id, o["op"], params, actor="ingest")
        results.append({"op": o["op"], "op_id": oid, "status": con.execute("SELECT status FROM op_log WHERE op_id=?", [oid]).fetchone()[0],
                        "n": len(params.get("records", [])) or r.get("n_reweighted")})
    return {"beliefs": beliefs, "ops": results}
