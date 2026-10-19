import copy
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from llm.router import Router, extract_json
from report.render import claim_lock, render_case_report
from tools.schemas import Finding, ResearchPlan
from tools.validation import ground_findings, validate_axes, validate_output


@pytest.mark.parametrize("text", ['a } b', 'a { b', 'quote " slash \\ brace }'])
def test_json_parser_handles_string_braces(text):
    raw = json.dumps({"text": text, "nested": {"value": 1}})
    assert json.loads(extract_json("prefix " + raw + " suffix"))["text"] == text


@pytest.mark.parametrize("text", ["80 percent", "80퍼센트", "80%", "절반 이상"])
def test_finding_numbers_and_grade_are_not_model_authority(text):
    finding = Finding(kind="consensus", text=text, cells=["target"], question_id="q", grade="[공]")
    assert finding.grade == "[시]" and finding.level == 1


def test_unknown_and_empty_schema_fields_rejected():
    base = dict(kind="consensus", text="hypothesis", cells=["target"], question_id="q")
    with pytest.raises(ValidationError):
        Finding(**base, statistics=[{"value": .8}])
    with pytest.raises(ValidationError):
        Finding(**{**base, "question_id": "   "})
    assert Finding.model_json_schema()["properties"]["question_id"]["minLength"] == 1


def simulation_and_card():
    card = {"schema_version": 2, "card_id": "cal", "run_id": "sim", "question_id": "q",
            "evaluation_kind": "independent_holdout", "level1_open": True,
            "cells": [{"label": "target", "level1_open": True, "calibrated": {"yes": .3, "no": .7}}]}
    run = {"run_id": "sim", "questions": [{"id": "q", "answer_schema": "choice"}],
           "aggregate": {"target": {"q": {"n": 20, "dist": {"yes": .9, "no": .1}}}}, "calibration": [card]}
    finding = Finding(kind="consensus", text="99% invented", cells=["target"], question_id="q",
                      metric="share", answer="yes").model_dump()
    return run, card, finding


def test_numeric_text_comes_from_matching_calibrated_aggregate():
    run, card, finding = simulation_and_card()
    ground_findings({"findings": [finding]}, run)
    assert "30.0%" in finding["text"] and "99%" not in finding["text"]
    assert finding["provenance"] == ["sim", "q"]
    assert claim_lock(finding, [card]) is None
    for key, value in [("question_id", "other"), ("run_id", "other"), ("evaluation_kind", "fit_only")]:
        assert claim_lock(finding, [{**card, key: value}])
    bad = copy.deepcopy(finding)
    bad["statistics"][0]["value"] = .99
    assert claim_lock(bad, [card])
    bad_card = copy.deepcopy(card)
    bad_card["cells"][0]["level1_open"] = False
    assert claim_lock(finding, [bad_card])


def test_unrelated_card_cannot_unlock_raw_simulation():
    run, card, finding = simulation_and_card()
    card["question_id"] = "other"
    ground_findings({"findings": [finding]}, run)
    assert "90.0%" in finding["text"]
    assert finding["statistics"][0]["source"] == "raw_simulation"
    assert claim_lock(finding, [card])


def test_report_escapes_unlocked_claims(monkeypatch, tmp_path):
    import report.render as report
    payload = '<script>alert("x")</script>'
    monkeypatch.setattr(report, "load_runs", lambda *_: [{
        "run_id": "r", "world_id": "w", "tool": "test", "status": "done", "decisions": [],
        "artifacts": [{"step": "x", "grade_summary": {}, "content": {
            "output": {"text": payload, "grade": "[시]", "provenance": ["r"]}}}]}])
    case = SimpleNamespace(name="safe", title="test", decision="test", options=[], research_questions=[])
    render_case_report(None, case, tmp_path)
    html = (tmp_path / "safe.html").read_text()
    assert payload not in html and "&lt;script&gt;" in html


def plan():
    return {"items": [{"hypothesis": "H", "information_need": "need", "questions": [{"text": "Q?", "type": "open"}],
                       "analysis": "qual_coding", "implications": [{"if": "yes", "then": "a"}, {"if": "no", "then": "b"}]}],
            "target_population": "target", "sample_size": 8, "sample_rationale": "exploratory interviews", "method": "qualitative"}


def test_context_validator_retries_unknown_option_inside_router(monkeypatch):
    good = plan()
    bad = copy.deepcopy(good)
    bad["items"][0]["implications"][1]["then"] = "invented"
    replies = iter([bad, good])
    seen = []
    async def create(**kwargs):
        seen.append(kwargs["messages"])
        return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=10),
                               choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(next(replies))))])
    router = Router(api_key="test-not-a-real-key")
    monkeypatch.setattr(router, "_aclient", lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    context = {"s0_context": {"options": [{"id": "a"}, {"id": "b"}], "hypotheses": ["H"]}}
    result, record = router.call("synthesize", [{"role": "user", "content": "test"}], schema=ResearchPlan,
                                 validator=lambda obj: validate_output(obj, context))
    assert result is not None and record.ok and record.attempts == 2
    assert "distinct option IDs" in seen[1][-1]["content"]


def test_finding_cell_ids_resolve_to_recorded_labels():
    from tools.schemas import FGIFindings
    run, _, finding = simulation_and_card()
    run["aggregate"]["contrast"] = run["aggregate"]["target"]
    findings = FGIFindings(findings=[{**finding, "cells": ["0", "1"]} for _ in range(3)],
                           hypotheses_to_test=["H"], limits="Exploratory")
    validate_output(findings, {"s1_run": run})
    assert all(f.cells == ["target", "contrast"] for f in findings.findings)
    findings.findings[0].cells = ["invented"]
    with pytest.raises(ValueError, match="exact aggregate labels"):
        validate_output(findings, {"s1_run": run})


def test_router_closes_client_before_loop_shutdown(monkeypatch):
    import asyncio
    closed = []
    async def close():
        closed.append(True)
    async def create(**kwargs):
        return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
                               choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])
    client = SimpleNamespace(close=close, chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    router = Router(api_key="test-not-a-real-key")
    def get_client():
        router._aclients[id(asyncio.get_running_loop())] = client
        return client
    monkeypatch.setattr(router, "_aclient", get_client)
    assert router.call("synthesize", [{"role": "user", "content": "test"}])[0] == "ok"
    assert closed == [True] and not router._aclients


def test_scenario_rejects_empty_unknown_or_same_axis():
    drivers = {k: {"impact": "H", "uncertainty": "H", "firm_controllable": False} for k in ["a", "b"]}
    validate_axes({"axis1_driver": "a", "axis2_driver": "b"}, drivers)
    for other in ["", "missing", "a"]:
        with pytest.raises(ValueError):
            validate_axes({"axis1_driver": "a", "axis2_driver": other}, drivers)


def test_scenario_waits_for_evidence_instead_of_inventing_second_axis():
    from tools.validation import NeedsEvidence, require_scenario_axes
    with pytest.raises(NeedsEvidence, match="2개 미만"):
        require_scenario_axes([{"id": "a", "impact": "H", "uncertainty": "H", "firm_controllable": False},
                               {"id": "b", "impact": "H", "uncertainty": "L", "firm_controllable": False}])


def test_external_energy_price_is_not_company_selling_price():
    from tools.checks import verify_scenario_axes
    drivers = [{"id": key, "text": text, "impact": "H", "uncertainty": "H", "firm_controllable": False}
               for key, text in [("energy", "전력 가격 변동"), ("policy", "정책 변화")]]
    run = {"decisions": {"g1_axes": {"axis1_driver": "energy", "axis2_driver": "policy", "adopted_drivers": drivers}},
           "outputs": {"s3_matrix": {"scenarios": [{"name": str(i), "common_trends": []} for i in range(4)]}}}
    assert verify_scenario_axes(SimpleNamespace(controllable=["가격"]), run)["passed"]
    drivers[0]["firm_controllable"] = True
    assert not verify_scenario_axes(SimpleNamespace(controllable=["가격"]), run)["passed"]


def test_independent_calibration_never_fits_evaluation_labels(con, world):
    from sim.calibrate import calibration_card
    from world import ops
    from world.store import create_world
    def anchor(wid, prefix, n_yes):
        records = [{"source_ref": f"{prefix}:{i}", "fields": {}, "observed": {"plan": "yes" if i < n_yes else "no"}} for i in range(20)]
        ops.run(con, wid, "anchor", {"records": records, "grade": "[사]", "holdout": ["plan"]})
    anchor(world, "train", 6)
    evaluation = create_world(con, "evaluation")
    anchor(evaluation, "test", 6)
    run = {"run_id": "sim", "aggregate": {"0": {"q": {"n": 20, "dist": {"yes": .1, "no": .9}}}}}
    spec = {"cells": [{"filter": "TRUE", "label": "all"}]}
    q = {"id": "q", "observed_attr": "plan", "options": ["yes", "no"]}
    card = calibration_card(con, world, run, spec, q, evaluation_world_id=evaluation)
    assert card["evaluation_kind"] == "independent_holdout" and card["level1_open"]
    assert not set(card["fit_source_refs"]) & set(card["evaluation_source_refs"])
    other = create_world(con, "different labels")
    anchor(other, "test2", 18)
    changed = calibration_card(con, world, run, spec, q, evaluation_world_id=other)
    assert changed["weights"] == card["weights"]
    assert changed["cells"][0]["calibrated"] == card["cells"][0]["calibrated"]
    assert changed["m1_after"] > .5 and not changed["level1_open"]
    with pytest.raises(ValueError, match="separate"):
        calibration_card(con, world, run, spec, q, evaluation_world_id=world)
    overlap = create_world(con, "overlap")
    anchor(overlap, "train", 6)
    with pytest.raises(ValueError, match="overlapping"):
        calibration_card(con, world, run, spec, q, evaluation_world_id=overlap)


def test_calibration_locks_when_observed_only_baseline_wins(con, world):
    """§10: 합성 보정이 실측 직접 추정보다 나쁘면 독립 평가를 통과해도 수준 주장은 잠긴다."""
    from sim.calibrate import calibration_card
    from world import ops
    from world.store import create_world
    def anchor(wid, prefix):   # 남자 yes 80%, 여자 yes 10%
        records = [{"source_ref": f"{prefix}:{s}:{i}", "fields": {"sex": s}, "observed": {"plan": "yes" if i < k else "no"}}
                   for s, k in [("남자", 16), ("여자", 2)] for i in range(20)]
        ops.run(con, wid, "anchor", {"records": records, "grade": "[사]", "holdout": ["plan"]})
    anchor(world, "train")
    evaluation = create_world(con, "evaluation")
    anchor(evaluation, "test")
    sim = {"n": 20, "dist": {"yes": .5, "no": .5}}   # 셀 차이를 전혀 모르는 시뮬
    run = {"run_id": "sim", "aggregate": {"0": {"q": sim}, "1": {"q": sim}}}
    spec = {"cells": [{"filter": "sex = '남자'", "label": "m"}, {"filter": "sex = '여자'", "label": "f"}]}
    q = {"id": "q", "observed_attr": "plan", "options": ["yes", "no"]}
    card = calibration_card(con, world, run, spec, q, evaluation_world_id=evaluation)
    assert card["baselines_m1"]["direct"] == 0 and card["m1_after"] > 0
    assert not card["beats_baselines"] and not card["level1_open"]
    assert not any(c["level1_open"] for c in card["cells"])


def test_calibration_card_bayes_and_regression_baseline(con, world):
    """§10: 카드 추정치는 합성 사전의 베이지안 사후, 셀에 factors가 있으면 주효과 회귀 기준선도 채점한다."""
    from sim.calibrate import calibration_card
    from world import ops
    from world.store import create_world
    plan = {("남자", "19-29"): 14, ("남자", "30-39"): 10, ("여자", "19-29"): 6, ("여자", "30-39"): 2}
    def anchor(wid, prefix):
        records = [{"source_ref": f"{prefix}:{s}:{b}:{i}", "fields": {"sex": s, "age": 25 if b == "19-29" else 35},
                    "observed": {"plan": "yes" if i < k else "no"}} for (s, b), k in plan.items() for i in range(20)]
        ops.run(con, wid, "anchor", {"records": records, "grade": "[사]", "holdout": ["plan"]})
    anchor(world, "train")
    evaluation = create_world(con, "evaluation")
    anchor(evaluation, "test")
    sim = {"n": 20, "dist": {"yes": .5, "no": .5}}
    run = {"run_id": "sim", "aggregate": {str(i): {"q": sim} for i in range(4)}}
    spec = {"cells": [{"filter": f"sex = '{s}' AND age_band = '{b}'", "label": f"{s} {b}", "factors": {"sex": s, "age": b}} for s, b in plan]}
    q = {"id": "q", "observed_attr": "plan", "options": ["yes", "no"]}
    card = calibration_card(con, world, run, spec, q, evaluation_world_id=evaluation)
    assert card["estimator"] == "bayes_synthetic_prior" and card["eb_alpha"] >= 1
    assert set(card["baselines_m1"]) == {"direct", "overall_mean", "regression"}
    for c in card["cells"]:   # 사후는 사전과 실측 사이
        lo, hi = sorted([c["prior"]["yes"], c["observed"]["yes"]])
        assert lo - 1e-4 <= c["calibrated"]["yes"] <= hi + 1e-4
    assert not card["beats_baselines"] and not card["level1_open"]   # 셀 차이를 모르는 시뮬은 실측 기준선에 진다


def test_share_of_unchosen_option_is_zero_not_rejected():
    """아무도 고르지 않은 선택지는 분포 키가 없다. 선택지에 있으면 0%로 접지하고, 없는 답은 거부한다."""
    from tools.schemas import FGIFindings
    run = {"run_id": "sim", "questions": [{"id": "q", "answer_schema": "choice", "options": ["yes", "no", "maybe"]}],
           "aggregate": {"target": {"q": {"n": 20, "dist": {"yes": .9, "no": .1}}}}, "calibration": []}
    ok = Finding(kind="consensus", text="maybe 0%", cells=["target"], question_id="q", metric="share", answer="maybe")
    validate_output(FGIFindings.model_construct(findings=[ok]), {"s1_run": run})
    finding = ok.model_dump()
    ground_findings({"findings": [finding]}, run)
    assert finding["statistics"][0]["value"] == 0.0
    bad = Finding(kind="consensus", text="ev 30%", cells=["target"], question_id="q", metric="share", answer="ev")
    with pytest.raises(ValueError, match="one of"):
        validate_output(FGIFindings.model_construct(findings=[bad]), {"s1_run": run})


def test_truncated_output_retries_fresh_with_more_tokens(monkeypatch):
    """토큰 한도에서 잘린 출력은 대화에 붙이지 않고, 원래 메시지로 한도를 늘려 다시 요청한다."""
    replies = iter([('{"a": "cut', "length"), (json.dumps({"x": 1}), "stop")])
    seen = []
    async def create(**kwargs):
        seen.append((kwargs["messages"], kwargs["max_tokens"]))
        text, reason = next(replies)
        return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=10),
                               choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=reason)])
    router = Router(api_key="test-not-a-real-key")
    monkeypatch.setattr(router, "_aclient", lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    result, record = router.call("synthesize", [{"role": "user", "content": "test"}], max_tokens=1000)
    assert record.ok and record.attempts == 2 and result == json.dumps({"x": 1})
    assert seen[1][0] == seen[0][0] and seen[1][1] == 1500
