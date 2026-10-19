"""Context-dependent checks run inside the model retry boundary, before artifacts are saved."""


class NeedsEvidence(ValueError):
    """A valid input cannot support the requested analysis yet."""


def eligible_drivers(drivers):
    return [d for d in drivers if d["impact"] == "H" and d["uncertainty"] == "H" and not d["firm_controllable"]]


def require_scenario_axes(drivers):
    eligible = eligible_drivers(drivers)
    if len({d["id"] for d in eligible}) < 2:
        raise NeedsEvidence("시나리오 축에 필요한 외생 H/H 동인이 2개 미만입니다. 동인 근거·평가를 보완해야 합니다.")
    return eligible


def validate_output(obj, inputs):
    from .schemas import AxisPairs, FGIFindings, ResearchPlan
    if isinstance(obj, AxisPairs):
        drivers = {d["id"]: d for d in inputs["s1_pestel"]["drivers"]}
        for pair in obj.pairs:
            validate_axes(pair.model_dump(), drivers)
    elif isinstance(obj, ResearchPlan):
        context = inputs["s0_context"]
        options = {o["id"] for o in context["options"]}
        for item in obj.items:
            if item.hypothesis not in context["hypotheses"]:
                raise ValueError("hypothesis must match an adopted hypothesis exactly")
            choices = {i.then for i in item.implications}
            if len(choices) < 2 or not choices <= options:
                raise ValueError(f"implications must use distinct option IDs from {sorted(options)}")
    elif isinstance(obj, FGIFindings):
        run = inputs["s1_run"]
        questions = {q["id"]: q for q in run["questions"]}
        aliases = dict(run.get("cell_labels") or {str(i): label for i, label in enumerate(run["aggregate"])})
        labels = list(aliases.values())
        # fgi_spec has an explicit target first and contrast second. Resolve only
        # these unambiguous presentation aliases, never guess an unknown segment.
        if len(labels) == 2:
            aliases.update({"target": labels[0], "타겟": labels[0], "타겟 셀": labels[0],
                            "contrast": labels[1], "대조": labels[1], "대조 셀": labels[1]})
        for finding in obj.findings:
            finding.cells = [cell if cell in run["aggregate"] else aliases.get(cell, cell) for cell in finding.cells]
            if finding.question_id not in questions:
                raise ValueError(f"question_id must be one of {list(questions)}; got {finding.question_id!r}")
            for cell in finding.cells:
                agg = run["aggregate"].get(cell, {}).get(finding.question_id)
                if not agg:
                    raise ValueError(f"cells must use exact aggregate labels {list(run['aggregate'])}; got {cell!r} for {finding.question_id}")
                if finding.level == 1:
                    allowed = set(agg["dist"]) | set(questions[finding.question_id].get("options") or [])   # 아무도 안 고른 선택지 = 0%
                    if finding.metric == "share" and finding.answer not in allowed:
                        raise ValueError(f"share requires answer to be one of {sorted(allowed)}; got {finding.answer!r}")
                    if finding.metric == "pos_rate" and questions[finding.question_id]["answer_schema"] not in ("likert5", "yes_no"):
                        raise ValueError("choice questions require metric=share and an answer key")
            finding.grade = "[시]"
            finding.provenance = [run["run_id"], finding.question_id]


def validate_axes(pair, drivers):
    axes = [pair["axis1_driver"], pair["axis2_driver"]]
    if axes[0] == axes[1]:
        raise ValueError("scenario axes must be distinct")
    for axis in axes:
        d = drivers.get(axis)
        if not d or d["firm_controllable"] or d["impact"] != "H" or d["uncertainty"] != "H":
            raise ValueError(f"axis {axis!r} must reference an external H/H driver")


def ground_findings(out, run):
    """Numeric prose is generated from aggregates; free-form prose remains an unvalidated hypothesis."""
    questions = {q["id"]: q for q in run["questions"]}
    for finding in out["findings"]:
        finding["grade"] = "[시]"
        finding["provenance"] = [run["run_id"], finding["question_id"]]
        if finding["level"] != 1:
            continue
        qid = finding["question_id"]
        stats, texts = [], []
        for cell in finding["cells"]:
            agg = run["aggregate"][cell][qid]
            answer = finding.get("answer")
            value = agg["dist"].get(answer, 0.0) if finding["metric"] == "share" else agg["pos_rate"]
            source, card_id = "raw_simulation", None
            # A calibration for another question or another cell can never ground this number.
            if finding["metric"] == "share":
                key = questions[qid].get("value_map", {}).get(answer, answer)
                for card in run.get("calibration", []):
                    if card.get("question_id") != qid or card.get("run_id") != run["run_id"]:
                        continue
                    c = next((c for c in card.get("cells", []) if c["label"] == cell), None)
                    if card.get("evaluation_kind") == "independent_holdout" and c and c.get("level1_open") and key in c["calibrated"]:
                        value, source, card_id = c["calibrated"][key], "calibrated", card["card_id"]
                        break
            else:
                key = None
            stats.append({"cell": cell, "question_id": qid, "run_id": run["run_id"],
                          "metric": finding["metric"], "answer": key, "value": value,
                          "source": source, "card_id": card_id})
            label = answer if finding["metric"] == "share" else "찬성"
            texts.append(f"{cell}: {label} {value:.1%} ({'보정 분포' if source == 'calibrated' else '가상 응답 집계'}, n={agg['n']})")
        finding["statistics"] = stats
        finding["text"] = f"질문 {qid} · " + "; ".join(texts)
