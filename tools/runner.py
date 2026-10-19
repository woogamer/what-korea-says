"""매니페스트 실행기. 단계는 query(DB) / model(LLM, 출력 스키마 필수) / gate(결정 대기) / sim(응답자 독립 병렬) 넷.

툴 연결: run_tool(..., inherit=이전 run 의 result) 로 이전 툴의 결정(decisions·decision_ids)을 상속한다. 매니페스트 requires 가 그 목록.

상태: running → waiting_decision → running → done. 결정은 decision 테이블에.
게이트 뒤 model 단계의 in 에는 게이트 앞 model 단계가 올 수 없다(매니페스트 검증). 결정값은 decisions 로 들어간다.
모든 model 단계는 prompt_snapshot 을 artifact 에 남겨 격리를 사후 검증할 수 있다.
"""
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import yaml

from world.store import new_id

from .prompts import PROMPTS
from .queries import QUERIES
from .schemas import SCHEMAS
from .validation import NeedsEvidence

MANIFESTS = Path(__file__).parent / "manifests"

DDL = """
CREATE TABLE IF NOT EXISTS tool_run (
  run_id TEXT PRIMARY KEY, case_name TEXT, world_id TEXT, tool TEXT, status TEXT,
  step_idx INTEGER, state JSON, error TEXT, ts TIMESTAMP);
"""


def load_manifest(tool: str) -> dict:
    m = yaml.safe_load((MANIFESTS / f"{tool}.yaml").read_text(encoding="utf-8"))
    validate_manifest(m)
    return m


def validate_manifest(m: dict):
    """게이트 격리 규칙을 매니페스트 수준에서 강제한다."""
    model_before_gate, seen_gate = set(), False
    for s in m["steps"]:
        if s["kind"] == "gate":
            seen_gate = True
            model_before_gate |= {x["id"] for x in m["steps"][:m["steps"].index(s)] if x["kind"] == "model"}
        elif s["kind"] in ("sim", "agent"):
            if "fn" not in s:
                raise ValueError(f"{s['id']}: {s['kind']} 단계는 fn 필수")
        elif s["kind"] == "model":
            if "out_schema" not in s:
                raise ValueError(f"{s['id']}: model 단계는 out_schema 필수")
            schema = SCHEMAS[s["out_schema"]]
            bad = set(s.get("forbid_fields", [])) & set(schema.model_fields)
            if bad:
                raise ValueError(f"{s['id']}: 스키마에 금지 필드 {bad}")
            if seen_gate:
                leak = set(s.get("in", [])) & model_before_gate
                if leak:
                    raise ValueError(f"{s['id']}: 게이트 뒤 단계가 게이트 앞 LLM 출력 {leak} 을 입력으로 받음")


def _grade_summary(obj) -> dict:
    c = Counter()

    def walk(x):
        if isinstance(x, dict):
            if "grade" in x and isinstance(x["grade"], str):
                c[x["grade"]] += 1
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(obj)
    return dict(c)


def _forbidden_keys(obj, forbid) -> list:
    hits = []

    def walk(x, path=""):
        if isinstance(x, dict):
            for k, v in x.items():
                if k in forbid:
                    hits.append(f"{path}.{k}")
                walk(v, f"{path}.{k}")
        elif isinstance(x, list):
            for i, v in enumerate(x):
                walk(v, f"{path}[{i}]")
    walk(obj)
    return hits


class ToolRun:
    def __init__(self, con, case, world_id, tool, router, inherit: dict | None = None, evaluation_world_id: str | None = None):
        con.execute(DDL)
        self.con, self.case, self.world_id, self.router = con, case, world_id, router
        self.evaluation_world_id = evaluation_world_id   # 보정 카드 독립 평가용 월드(적합에 안 씀)
        self.m = load_manifest(tool)
        self.tool = tool
        self.run_id = new_id("run")
        self.outputs, self.decisions, self.artifacts = {}, {}, []
        self.decision_ids = {}
        self.inherited_from = None
        if inherit:
            missing = [g for g in self.m.get("requires", []) if g not in inherit.get("decisions", {})]
            if missing:
                raise ValueError(f"{tool}: 상속할 결정 없음 {missing}")
            self.decisions = dict(inherit["decisions"])
            self.decision_ids = dict(inherit.get("decision_ids", {}))
            self.inherited_from = inherit["run_id"]
        elif self.m.get("requires"):
            raise ValueError(f"{tool}: requires {self.m['requires']} — 이전 툴 결과(inherit) 필요")
        self.step_idx, self.status, self.error = 0, "running", None
        self.n_gates = 0
        con.execute("INSERT INTO tool_run VALUES (?,?,?,?,?,?,?,?,?)",
                    [self.run_id, case.name, world_id, tool, "running", 0, "{}", None, datetime.now()])

    # ---- 입력 조립 ----
    def _inputs(self, step):
        inp = {}
        for name in step.get("in", []):
            if name == "case":
                inp["case"] = self.case.model_dump()
            elif name == "decisions":
                inp["decisions"] = self.decisions
            elif name in self.decisions:
                inp.setdefault("decisions", {})[name] = self.decisions[name]
            elif name in self.outputs:
                inp[name] = self.outputs[name]
            else:
                raise KeyError(f"{step['id']}: 입력 {name} 없음")
        return inp

    def _ctx(self):
        return {"con": self.con, "world_id": self.world_id, "case": self.case, "decisions": self.decisions,
                "decision_ids": self.decision_ids, "outputs": self.outputs, "router": self.router, "run_id": self.run_id,
                "evaluation_world_id": self.evaluation_world_id}

    def _save_artifact(self, step_id, content, grade_summary=None, kind="step"):
        aid = new_id("art")
        if isinstance(content, dict) and self.decision_ids:
            content = {**content, "provenance": {**content.get("provenance", {}), "decisions": self.decision_ids,
                                                 "inherited_from": self.inherited_from}}
        self.con.execute("INSERT INTO artifact VALUES (?,?,?,?,?,?,?,?)",
                         [aid, self.world_id, self.tool, step_id, json.dumps(content, ensure_ascii=False),
                          json.dumps(grade_summary or {}, ensure_ascii=False), self.run_id, datetime.now()])
        self.artifacts.append({"artifact_id": aid, "step": step_id, "kind": kind, "grade_summary": grade_summary or {}})
        return aid

    # ---- 실행 ----
    def run(self, decisions=None):
        """decisions: {gate_id: dict | callable(gate_view)->dict}. 없으면 게이트에서 멈춘다."""
        decisions = decisions or {}
        steps = self.m["steps"]
        while self.step_idx < len(steps):
            s = steps[self.step_idx]
            try:
                if s["kind"] in ("query", "sim", "agent"):   # agent: 검색·추출·검사 루프(LLM 호출 포함), 산출물은 후보+추적
                    out = QUERIES[s["fn"]](self._ctx())
                    self.outputs[s["id"]] = out
                    self._save_artifact(s["id"], out, {out["grade"]: 1} if "grade" in out else {}, kind=s["kind"])
                elif s["kind"] == "model":
                    self._model_step(s)
                elif s["kind"] == "gate":
                    view = {k: self.outputs[k] for k in s.get("show", []) if k in self.outputs}
                    d = decisions.get(s["id"])
                    if callable(d):
                        d = d(view)
                    if d is None:
                        self.status = "waiting_decision"
                        self._persist()
                        return self.result()
                    self.decide(s, d, view)
                    self.n_gates += 1
            except NeedsEvidence as e:
                self.status, self.error = "waiting_evidence", str(e)
                self._persist()
                return self.result()
            except Exception as e:
                self.status, self.error = "error", f"{s['id']}: {type(e).__name__}: {str(e)[:300]}"
                self._persist()
                return self.result()
            self.step_idx += 1
        self.status = "done"
        self._persist()
        return self.result()

    def _model_step(self, s):
        inp = self._inputs(s)
        if s["out_schema"] == "AxisPairs":
            from .validation import require_scenario_axes
            require_scenario_axes(inp["s1_pestel"]["drivers"])
        msgs = PROMPTS[s["prompt"]](inp)
        schema = SCHEMAS[s["out_schema"]]
        from .validation import validate_output, ground_findings
        obj, rec = self.router.call(s["role"], msgs, schema=schema, max_tokens=s.get("max_tokens", 6000), retries=2, tag=f"{self.run_id}:{s['id']}",
                                    validator=lambda result: validate_output(result, inp))
        if obj is None:
            self._save_artifact(s["id"], {"validation_error": rec.error, "raw_output": rec.raw}, kind="error")
            raise RuntimeError(f"LLM 출력 실패: {rec.error}")
        out = obj.model_dump()
        if s["out_schema"] == "FGIFindings":
            ground_findings(out, inp["s1_run"])
        hits = _forbidden_keys(out, set(s.get("forbid_fields", [])))
        if hits:
            raise RuntimeError(f"금지 필드 출력 {hits}")
        self.outputs[s["id"]] = out
        self._save_artifact(s["id"], {"output": out, "prompt_snapshot": msgs, "inputs": sorted(inp),
                                      "model": rec.model, "cost_usd": rec.cost_usd}, _grade_summary(out))

    def decide(self, step, chosen: dict, view: dict):
        schema = SCHEMAS[step["decision_schema"]]
        d = schema.model_validate(chosen).model_dump()
        if step["decision_schema"] == "AdoptScenarioAxes":
            from .validation import validate_axes
            drivers = {x["id"]: x for x in d["adopted_drivers"]}
            validate_axes(d, drivers)
            shown = {x["id"]: x for x in view.get("s1_pestel", {}).get("drivers", [])}
            if any(shown.get(k) != v for k, v in drivers.items()):
                raise ValueError("adopted drivers must match the displayed candidates")
        aid = self._save_artifact(step["id"], {"shown": view}, kind="gate")
        did = new_id("d")
        self.con.execute("INSERT INTO decision VALUES (?,?,?,?,?,?,?)",
                         [did, aid, step["id"], json.dumps(d, ensure_ascii=False), d.get("rationale", ""),
                          "analyst", datetime.now()])
        self.decisions[step["id"]] = d
        self.decision_ids[step["id"]] = did

    def resume(self, decision: dict):
        s = self.m["steps"][self.step_idx]
        assert s["kind"] == "gate" and self.status == "waiting_decision"
        self.status = "running"
        return self.run({s["id"]: decision})

    def _persist(self):
        self.con.execute("UPDATE tool_run SET status=?, step_idx=?, state=?, error=? WHERE run_id=?",
                         [self.status, self.step_idx, json.dumps({"outputs": self.outputs, "decisions": self.decisions}, ensure_ascii=False),
                          self.error, self.run_id])

    def result(self):
        return {"run_id": self.run_id, "tool": self.tool, "status": self.status, "error": self.error,
                "step_idx": self.step_idx, "n_gates": self.n_gates, "outputs": self.outputs,
                "decisions": self.decisions, "decision_ids": self.decision_ids, "artifacts": self.artifacts,
                "inherited_from": self.inherited_from}


def run_tool(con, case, world_id, tool, router, decisions=None, inherit=None, evaluation_world_id=None) -> dict:
    return ToolRun(con, case, world_id, tool, router, inherit, evaluation_world_id).run(decisions)
