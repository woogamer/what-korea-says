"""체크포인트용 사후 검증. 산출물·결정·프롬프트 스냅샷을 DB에서 읽어 판정한다."""
import json


def verify_gate_isolation(con, run_id: str) -> dict:
    """게이트 뒤 model 단계 프롬프트에 게이트 앞 LLM 출력 텍스트가 섞이지 않았는가 + 결정이 기록됐는가."""
    rows = con.execute("SELECT step, content, grade_summary FROM artifact WHERE op_id=? ORDER BY ts, rowid", [run_id]).fetchall()
    problems, pre_gate_texts, seen_gate = [], [], False
    gates = con.execute("SELECT count(*) FROM decision d JOIN artifact a USING (artifact_id) WHERE a.op_id=?", [run_id]).fetchone()[0]
    for step, content, gs in rows:
        c = json.loads(content)
        if "shown" in c:                      # gate artifact
            seen_gate = True
            continue
        if "prompt_snapshot" in c:            # model step
            if seen_gate:
                prompt = json.dumps(c["prompt_snapshot"], ensure_ascii=False)
                for sid, txt in pre_gate_texts:
                    if len(txt) > 40 and txt in prompt:
                        problems.append(f"{step} 프롬프트에 {sid} 출력 포함")
            else:
                out = c["output"]
                for v in _strings(out):
                    pre_gate_texts.append((step, v))
            if not json.loads(gs):
                pass
    if gates == 0:
        problems.append("decision 기록 없음")
    return {"passed": not problems, "problems": problems, "n_decisions": gates}


def _strings(x):
    if isinstance(x, str):
        yield x
    elif isinstance(x, dict):
        for v in x.values():
            yield from _strings(v)
    elif isinstance(x, list):
        for v in x:
            yield from _strings(v)


def verify_scenario_axes(case, run: dict) -> dict:
    d = run["decisions"].get("g1_axes", {})
    matrix = run["outputs"].get("s3_matrix", {})
    drivers = {x["id"]: x for x in d.get("adopted_drivers", [])}
    problems = []
    a1, a2 = d.get("axis1_driver"), d.get("axis2_driver")
    if a1 == a2:
        problems.append("두 축이 같은 동인")
    for a in (a1, a2):
        drv = drivers.get(a)
        if drv is None:
            problems.append(f"축 {a} 이 채택 동인 표에 없음"); continue
        # An external electricity price can contain "가격" without being a
        # company's controllable selling price. Use the adopted classification.
        if drv["firm_controllable"]:
            problems.append(f"축 {a} 이 외생 변수가 아님")
        if not (drv["impact"] == "H" and drv["uncertainty"] == "H"):
            problems.append(f"축 {a} 이 핵심 불확실성(H·H)이 아님: {drv['impact']}/{drv['uncertainty']}")
    base = {i for i, x in drivers.items() if x["impact"] == "H" and x["uncertainty"] == "L"}
    for s in matrix.get("scenarios", []):
        missing = base - set(s["common_trends"])
        if missing:
            problems.append(f"시나리오 '{s['name']}' 에 기본 트렌드 {sorted(missing)} 미반영")
    names = [s["name"] for s in matrix.get("scenarios", [])]
    if len(set(names)) != 4:
        problems.append(f"시나리오 4개 아님/이름 중복: {names}")
    return {"passed": not problems, "problems": problems, "n_drivers": len(drivers), "axes": [a1, a2]}


# ---- 툴 연결 (C4) ----

def auto_stp_decisions(pack):
    return {"g1_adopt_axes": {"axes": pack["axes_default"], "rationale": "러너 자동 결정(개발 루프): 팩 기본 축"},
            "g2_target": lambda view: {"cell": view["s2_cells"]["largest"], "rationale": "러너 자동 결정: 규모 최대 셀"}}


def run_chain(con, case, world_id, router, pack, n_needs=3, evaluation_world_id=None) -> dict:
    """STP → persona_bum → fgi 를 결정 상속으로 연결. 개발 루프용 자동 결정."""
    from .runner import run_tool
    stp = run_tool(con, case, world_id, "stp", router, decisions=auto_stp_decisions(pack))
    if stp["status"] != "done":
        return {"stp": stp, "error": f"stp {stp['status']}: {stp.get('error')}"}
    bum = run_tool(con, case, world_id, "persona_bum", router, inherit=stp, decisions={
        "g1_needs": lambda view: {"needs": [c["text"] for c in view["s2_bum"]["latent_needs"][:n_needs]],
                                  "rationale": "러너 자동 결정: 앞 3개 니즈"}})
    if bum["status"] != "done":
        return {"stp": stp, "bum": bum, "error": f"persona_bum {bum['status']}: {bum.get('error')}"}
    fgi = run_tool(con, case, world_id, "fgi", router, inherit=bum, evaluation_world_id=evaluation_world_id, decisions={
        "g1_hypotheses": lambda view: {"hypotheses": view["s2_findings"]["hypotheses_to_test"][:2],
                                       "rationale": "러너 자동 결정: 앞 2개 가설"}})
    return {"stp": stp, "bum": bum, "fgi": fgi, "error": None if fgi["status"] == "done" else f"fgi {fgi['status']}: {fgi.get('error')}"}


def verify_chain(con, case, world_id, router, pack) -> dict:
    ch = run_chain(con, case, world_id, router, pack)
    if ch["error"]:
        return {"passed": False, "problems": [ch["error"]], "chain": ch}
    stp, bum, fgi = ch["stp"], ch["bum"], ch["fgi"]
    problems = []
    d_target = stp["decision_ids"]["g2_target"]
    # 1) 페르소나 = 표적 셀의 실제 행
    smp = bum["outputs"]["s0_target_sample"]
    f = smp["filter"]
    from world.store import view_name
    for r in smp["rows"]:
        ok = con.execute(f"SELECT count(*) FROM {view_name(world_id)} WHERE uuid=? AND ({f})", [r["uuid"]]).fetchone()[0]
        if not ok:
            problems.append(f"페르소나 행 {r['uuid'][:8]} 이 타겟 셀 밖")
    cards = {c["uuid"] for c in bum["outputs"]["s1_persona_cards"]["cards"]}
    if not cards <= {r["uuid"] for r in smp["rows"]}:
        problems.append("페르소나 카드 uuid 가 샘플 행과 불일치(창작된 페르소나)")
    # 2) FGI 셀이 타겟 결정에서 파생 + provenance 에 decision id
    spec = fgi["outputs"]["s0_spec"]
    if spec["cells"][0]["filter"] != f:
        problems.append("FGI 타겟 셀 필터가 페르소나 단계의 타겟 필터와 다름")
    if spec["provenance"]["decisions"].get("g2_target") != d_target or "g1_needs" not in spec["provenance"]["decisions"]:
        problems.append("FGI 스펙 provenance 에 타겟/니즈 decision id 없음")
    # 3) 응답자 독립: sim run 의 호출 수 = 응답자 수(응답자당 1요청), 셀 2개
    run = fgi["outputs"]["s1_run"]
    n_calls = con.execute("SELECT count(*) FROM llm_call WHERE run_id=?", [run["run_id"]]).fetchone()[0]
    if n_calls != run["n_requested"]:
        problems.append(f"응답자 {run['n_requested']}명인데 호출 {n_calls}건 (독립 요청 아님)")
    if len(spec["cells"]) < 2:
        problems.append("대조 셀 없음")
    # 4) 발견의 등급·출처
    fd = fgi["outputs"]["s2_findings"]["findings"]
    bad = [x for x in fd if x["grade"] != "[시]" or run["run_id"] not in " ".join(x["provenance"])]
    if bad:
        problems.append(f"발견 {len(bad)}건이 [시]/run id 출처 없음")
    agg = run["aggregate"]
    ev = (f"STP→페르소나({len(cards)}행)→BUM({len(bum['outputs']['s2_bum']['pains'])}통점)→FGI(셀 2·응답자 {run['n_ok']}/{run['n_requested']}·"
          f"질문 {len(spec['questions'])}) 연결. 발견 {len(fd)}건 [시]. decision 상속 {list(fgi['decision_ids'])}")
    return {"passed": not problems, "problems": problems, "evidence": ev, "chain": ch,
            "metrics": {"n_respondents": run["n_ok"], "n_findings": len(fd)}}


# ---- 조사로 되돌아가기 (C5) ----

def run_plan(con, case, world_id, router, chain) -> dict:
    from .runner import run_tool
    fgi = chain["fgi"]
    return run_tool(con, case, world_id, "research_plan", router, inherit=fgi)


def verify_research_plan(con, case, world_id, router, chain) -> dict:
    from .schemas import ANALYSIS_METHODS
    run = run_plan(con, case, world_id, router, chain)
    if run["status"] != "done":
        return {"passed": False, "evidence": f"research_plan {run['status']}: {run.get('error')}", "run": run}
    plan = run["outputs"]["s1_plan"]
    opts = {o.id for o in case.options}
    problems = []
    for it in plan["items"]:
        if not it["questions"]:
            problems.append(f"가설 '{it['hypothesis'][:20]}' 문항 없음")
        if it["analysis"] not in ANALYSIS_METHODS:
            problems.append(f"분석 방법 '{it['analysis']}' 목록 밖")
        thens = [i["then"] for i in it["implications"]]
        if len(set(thens)) < 2:
            problems.append(f"가설 '{it['hypothesis'][:20]}': 결과가 달라도 같은 결정({thens}) → 조사 불필요")
        if not set(thens) <= opts:
            problems.append(f"가설 '{it['hypothesis'][:20]}': 알 수 없는 선택지 {set(thens) - opts}")
    if not (isinstance(plan["sample_size"], int) and plan["sample_size"] > 0):
        problems.append(f"표본 크기 {plan['sample_size']} (양의 정수와 설계 근거 필요)")
    ev = (f"가설 {len(plan['items'])}개 → 문항 {sum(len(i['questions']) for i in plan['items'])}개, "
          f"분석 {sorted({i['analysis'] for i in plan['items']})}, 결과별 결정 분기 확인. 표본 {plan['sample_size']} ({plan['method']})")
    return {"passed": not problems, "evidence": ev if not problems else "; ".join(problems), "run": run}
