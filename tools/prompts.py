"""model 단계 프롬프트. 입력은 runner가 준 inputs(dict)뿐이다. 게이트 뒤 단계는 게이트 앞 LLM 출력을 받지 못한다."""
import json


def J(x):
    return json.dumps(x, ensure_ascii=False, indent=None)


RULES = ("규칙: 수치는 입력에 있는 것만 인용한다. 입력에 없는 수치·사실은 만들지 말고 grade '[미]'로 표시한다. "
         "모든 주장에는 근거 id(belief id, query:<step>, url)를 붙인다. JSON만 출력한다.")


def stp_s1(inp):
    case, elig = inp["case"], inp["s0_eligibility"]
    return [{"role": "system", "content": "너는 마케팅 전략 컨설턴트다. " + RULES},
            {"role": "user", "content":
             f"대상 시장: {case['title']}\n의사결정: {case['decision']}\n표적 모집단: {case['target_population']}\n"
             f"자격 심사를 통과한 축 후보 열(모집단 안에서 값이 갈리는 열만): {J(elig['eligible'])}\n"
             f"열 상세: {J(elig['attrs'])}\n\n"
             "Abell의 3차원(고객집단 who × 고객기능·니즈 what × 기술·접근 how)에 맞춰 세분화 축 조합 후보 2~3개를 제안하고 "
             "각 조합의 장단점을 비교하라. 축은 반드시 위 통과 열 이름으로만 쓴다. "
             "어느 조합을 채택할지는 분석자가 결정하므로 추천하지 말고 후보와 트레이드오프만 제시한다."}]


def stp_s3(inp):
    cells, beliefs, case = inp["s2_cells"], inp["beliefs"], inp["case"]
    return [{"role": "system", "content": "너는 마케팅 전략 컨설턴트다. " + RULES},
            {"role": "user", "content":
             f"의사결정: {case['decision']}\n채택된 축: {J(cells['axes'])}\n"
             f"세분시장 셀 표(규모 size_est는 DB 집계, 등급 {cells['grade']}; 규모 0인 셀 제외): {J([{k: r[k] for k in ('cell', 'size_est', 'n_rows', 'anchor_n')} for r in cells['cells'] if r['size_est'] > 0])}\n"
             f"사용 가능한 근거(belief): {J(beliefs['beliefs'])}\n\n"
             "셀마다 니즈 강도·지불의사·접근성을 주장(claim)으로 쓰되, 근거 belief가 있으면 그 belief의 등급과 id를, "
             "없으면 grade '[미]'와 provenance ['[미]']를 쓴다. 규모·성장성은 쓰지 않는다(규모는 DB, 성장성은 [공] 근거가 들어오면 채움). "
             "타겟을 추천하지 않는다."}]


def stp_s4(inp):
    return [{"role": "system", "content": "너는 회의적인 검토자다. 결론을 내리지 말고 약점만 짚어라. " + RULES},
            {"role": "user", "content":
             f"비교표: {J(inp['s3_compare_table'])}\n근거 목록: {J(inp['beliefs']['beliefs'])}\n\n"
             "근거가 가장 약한 주장, 빠진 근거, 주장끼리의 모순을 짚어라. 어떤 셀을 고르라는 말은 하지 않는다."}]


def stp_s5(inp):
    t, case = inp["target_cell_rows"], inp["case"]
    d = inp["decisions"]["g2_target"]
    return [{"role": "system", "content": "너는 마케팅 전략 컨설턴트다. " + RULES},
            {"role": "user", "content":
             f"분석자가 채택한 타겟 셀: {t['target']} (이유: {d['rationale']})\n"
             f"셀 DB 집계: {J(t['row'])} / 전체 규모 {t['total_size']}\n근거: {J(inp['beliefs']['beliefs'])}\n"
             f"의사결정: {case['decision']}\n\n"
             "이 타겟에 대해 매력도·역량 적합성·방어 가능성 관점의 근거 주장을 각각 쓰고(근거 없으면 [미]), "
             "실제로 확인해야 할 질문을 3~5개 적어라. 다른 셀과 비교하거나 타겟을 바꾸자고 하지 않는다."}]


def scen_s1(inp):
    d = inp["s0_drivers"]
    return [{"role": "system", "content": "너는 시나리오 플래닝 전략 컨설턴트다. " + RULES},
            {"role": "user", "content":
             f"의사결정: {d['decision']}\n시계: {d['horizon']}\n"
             f"이미 등록된 환경 동인(id·등급·출처 포함): {J(d['drivers'])}\n"
             f"기업이 선택 가능한 변수(외생 아님): {J(d['controllable'])}\n\n"
             "PESTEL 각 차원 2개 내외, 총 10~12개 동인을 정리하라. 등록된 동인은 id를 그대로 쓰고 provenance에 그 id를 넣는다. "
             "새로 추가하는 동인은 id를 new_로 시작하고 provenance ['[가]']로 표시한다. "
             "각 동인의 영향력·불확실성을 H/M/L로 평가하고 Wilson matrix 사분면을 명시한다: 영향력 H·불확실성 L = 기본 트렌드(모든 시나리오에 공통), "
             "영향력 H·불확실성 H = 핵심 불확실성(시나리오 축 후보). 핵심 불확실성으로 볼 만한 동인이 있으면 불확실성을 H로 정직하게 평가한다. "
             "기업이 스스로 정할 수 있는 변수(가격·채널·상품 등)는 firm_controllable=true, 외부 환경은 false."}]


def scen_s2(inp):
    from .validation import eligible_drivers
    eligible = eligible_drivers(inp["s1_pestel"]["drivers"])
    return [{"role": "system", "content": "너는 시나리오 플래닝 전략 컨설턴트다. " + RULES},
            {"role": "user", "content":
             f"선택 가능한 외생 H/H 동인만: {J(eligible)}\n선택 불가(외생 아님) 변수: {J(inp['s0_drivers']['controllable'])}\n\n"
             "위 id만 사용해 서로 다른 두 동인으로 시나리오 축 쌍 후보 1~3개를 제안하고 "
             "각 쌍의 장단점을 비교하라. 어느 쌍을 채택할지는 분석자가 결정하므로 추천하지 않는다."}]


def scen_s3(inp):
    d = inp["decisions"]["g1_axes"]
    base = [x for x in d["adopted_drivers"] if x["impact"] == "H" and x["uncertainty"] == "L"]
    return [{"role": "system", "content": "너는 시나리오 플래닝 전략 컨설턴트다. " + RULES},
            {"role": "user", "content":
             f"분석자가 채택한 축: 축1 {d['axis1_driver']} {J(d['axis1_poles'])} / 축2 {d['axis2_driver']} {J(d['axis2_poles'])} (이유: {d['rationale']})\n"
             f"분석자가 인정한 동인 표: {J(d['adopted_drivers'])}\n"
             f"기본 트렌드(영향력 H·불확실성 L, 4개 시나리오 전부에 반영할 것): {J([x['id'] for x in base])}\n"
             f"의사결정: {inp['case']['decision']}\n\n"
             "2×2 시나리오 4개를 만들어라. 각 시나리오에 기억하기 쉬운 이름, 5문장 이내 내러티브, 반영한 기본 트렌드 id 전부(common_trends), "
             "그 세계의 승자 유형을 적는다. 5기준(개연성·일관성·차별성·관련성·도전성) 검증 한 문장씩, "
             "4개 모두에서 후회 없는 무브 3개, 각 시나리오로의 이행을 알리는 관찰 가능한 조기경보 지표를 적는다."}]


PROMPTS = {f.__name__: f for f in [stp_s1, stp_s3, stp_s4, stp_s5, scen_s1, scen_s2, scen_s3]}


def bum_s1(inp):
    t = inp["s0_target_sample"]
    return [{"role": "system", "content": "너는 고객경험 리서처다. 페르소나는 창작이 아니라 아래 실제 행을 옮겨 적는 것이다. " + RULES},
            {"role": "user", "content":
             f"타겟 세그먼트: {t['target']} (셀 내 표본 {t['n_in_cell']}행)\n"
             f"표적 셀에서 층화 샘플한 실제 행(각 uuid 유지, anchored=true는 실측 정박 고객): {J(t['rows'])}\n\n"
             "행마다 페르소나 카드를 써라(가명·목표·제약·사용 맥락·여정). 제약은 행에 있는 주거·지역·가족·attrs 조건에서만 끌어내고, "
             "행에 없는 소득·차량 보유 같은 사실은 쓰지 않는다."}]


def bum_s2(inp):
    return [{"role": "system", "content": "너는 고객경험 리서처다. " + RULES},
            {"role": "user", "content":
             f"페르소나 카드: {J(inp['s1_persona_cards'])}\n의사결정: {inp['case']['decision']}\n근거(belief): {J(inp['beliefs']['beliefs'])}\n\n"
             "각 페르소나의 페인포인트를 구매자 효용지도에 배치하라: 6대 효용 레버(고객생산성·단순성·편의성·위험감소·재미와이미지·환경친화성) × "
             "6단계 구매자 경험 사이클(구매·배달·사용·보완·유지·폐기). 근거 belief가 있으면 그 등급과 id, 없으면 grade [가]. "
             "막힌 효용 공간과 잠재·미충족 니즈(검증 전, [가])를 도출하라. 어느 니즈를 검증할지는 분석자가 고른다."}]


def fgi_s2(inp):
    r = inp["s1_run"]
    cell_ids = r.get("cell_labels") or {str(i): label for i, label in enumerate(r["aggregate"])}
    aggregates = {i: r["aggregate"].get(label, {}) for i, label in cell_ids.items()}
    return [{"role": "system", "content": "너는 FGI 진행자다. 아래 수치는 시뮬레이션 집계(등급 [시])이며 실제 고객이 아니다. " + RULES},
            {"role": "user", "content":
             f"시뮬레이션 run {r['run_id']} (응답자 {r['n_ok']}명, 셀 2개: 타겟·대조). 질문: {J(r['questions'])}\n"
             f"셀 ID → 표시 이름: {J(cell_ids)}. 출력 cells에는 ID 문자열(예: ['0', '1'])을 쓴다.\n"
             f"셀별 분포·찬성률(pos_rate = 4~5점 비율): {J(aggregates)}\n"
             f"최빈값에서 먼 응답과 이유: {J(r['outliers'])}\n\n"
             "합의점·불일치점·새 가설을 정리하라. 각 항목의 grade는 [시], question_id는 입력 질문 id, cells는 위 셀 ID 문자열을 쓴다. "
             "수치를 언급하는 항목은 level=1, metric은 likert5/yes_no의 찬성률이면 pos_rate, 선택지 비율이면 share와 answer(집계 dist의 키)를 지정한다. "
             "수치 문장은 서버가 집계로 다시 작성한다. level=0 자유 문장은 검증되지 않은 해석·가설로만 표시된다. "
             "개인 예측·인과 효과는 주장하지 않는다. provenance에 run id와 질문 id. "
             "타겟 셀과 대조 셀의 차이를 명시하라. 실데이터로 확인할 가설을 문장으로 적고 한계를 명시하라."}]


def fgi_s3(inp):
    return [{"role": "system", "content": "너는 회의적인 검토자다. 결론을 뒤집을 반박만 쓴다. " + RULES},
            {"role": "user", "content":
             f"FGI 발견: {J(inp['s2_findings'])}\n집계: {J(inp['s1_run']['aggregate'])}\n\n"
             "가장 약한 발견, 대안 설명(예: 렌더링 편향·셀 표본 크기·질문 프레이밍), 무엇이 나오면 결론을 바꿔야 하는지를 적어라."}]


PROMPTS.update({f.__name__: f for f in [bum_s1, bum_s2, fgi_s2, fgi_s3]})


def plan_s1(inp):
    c = inp["s0_context"]
    return [{"role": "system", "content": "너는 마케팅 조사 설계자다. 설문부터 만들지 않는다: 가설 → 정보 필요 → 문항 → 분석 계획 → 결과별 결정 순서. " + RULES},
            {"role": "user", "content":
             f"의사결정: {c['decision']}\n선택지(id 사용): {J(c['options'])}\n타겟: {c['target']} / 표적 모집단: {c['target_population']}\n"
             f"기존 정보 필요: {J(c['information_needs'])}\n채택된 가설: {J(c['hypotheses'])}\n\n"
             "가설마다(채택된 가설 문장을 그대로 hypothesis에 복사): 어떤 정보가 필요한지, 실제 설문/인터뷰 문항(1개 이상, 유형 명시), 분석 방법 하나"
             "(frequency/average/group_comparison/user_vs_nonuser/funnel/importance_vs_satisfaction/qual_coding), "
             "그리고 결과가 A일 때와 B일 때 각각 어느 선택지 id로 가는지(서로 달라야 한다. 같다면 그 가설은 조사할 필요가 없다). "
             "표적 모집단·예상 표본 크기(양의 정수)·그 근거·조사 방법(qualitative/quantitative/qual_quant)을 적어라."}]


PROMPTS["plan_s1"] = plan_s1
