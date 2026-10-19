"""툴 단계의 LLM 출력 스키마. 게이트 앞 스키마에는 추천·채택 필드가 없다(구조로 막음)."""
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator


NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class OutputModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, str_min_length=1)



def _join(v):
    return "; ".join(map(str, v)) if isinstance(v, list) else v


def _valid_grade(v):
    from world.grades import GRADE_RANK
    if v not in GRADE_RANK:
        raise ValueError(f"grade must be one of {list(GRADE_RANK)}, got {v!r}")
    return v


class Claim(OutputModel):
    text: NonEmptyStr
    grade: NonEmptyStr = Field(description="[공] [사] [현] [도] [도-LLM] [시] [가] [미] 중 하나(한글 그대로)")
    provenance: list[NonEmptyStr] = Field(min_length=1, description="belief id · attr 이름 · url · 'query:<step>'. 근거가 없으면 반드시 ['[미]']")

    _g = field_validator("grade")(_valid_grade)


# ---- STP ----

class AxisCombo(OutputModel):
    who: NonEmptyStr = Field(description="고객집단 축으로 쓸 열 이름 (자격 심사 통과 열만)")
    what: NonEmptyStr = Field(description="고객 기능·니즈 축으로 쓸 열 이름")
    how: NonEmptyStr = Field(description="기술·접근 축으로 쓸 열 이름, 없으면 '-'")
    pros: NonEmptyStr
    cons: NonEmptyStr
    evidence: list[NonEmptyStr] = Field(description="근거 belief id 또는 'query:s0_eligibility'")

    _j = field_validator("pros", "cons", mode="before")(_join)


class AxisCandidates(OutputModel):
    candidates: list[AxisCombo] = Field(min_length=2, max_length=3)
    note: NonEmptyStr = Field(description="축 선택은 분석자의 판단이라는 점을 포함한 한 문장")


class CompareRow(OutputModel):
    cell: NonEmptyStr
    needs_intensity: Claim
    willingness_to_pay: Claim
    accessibility: Claim


class CompareTable(OutputModel):
    rows: list[CompareRow]
    caveats: list[NonEmptyStr]


class Critique(OutputModel):
    weakest_claims: list[NonEmptyStr] = Field(description="근거가 가장 약한 주장 3개 이내, 각 셀·항목 명시")
    missing_evidence: list[NonEmptyStr]
    contradictions: list[NonEmptyStr]


class TargetBrief(OutputModel):
    target_cell: NonEmptyStr
    why_here: list[Claim] = Field(description="매력도·역량 적합·방어 가능성 각각의 근거 주장")
    next_questions: list[NonEmptyStr] = Field(description="이 타겟에 대해 실제로 확인해야 할 질문")


# ---- 시나리오 ----

class Driver(OutputModel):
    id: NonEmptyStr = Field(description="case.environment id 또는 새 id(new_*)")
    dimension: NonEmptyStr
    text: NonEmptyStr
    impact: Literal["H", "M", "L"]
    uncertainty: Literal["H", "M", "L"]
    firm_controllable: bool = Field(description="기업이 스스로 정할 수 있는 변수이면 true(가격·채널·상품 등). 정책·기술·경제 등 외부 환경이면 false")
    provenance: list[NonEmptyStr]


class PestelTable(OutputModel):
    drivers: list[Driver] = Field(min_length=8, max_length=14)


class AxisPair(OutputModel):
    axis1_driver: NonEmptyStr = Field(description="driver id (impact H, uncertainty H, firm_controllable=false)")
    axis1_poles: list[NonEmptyStr] = Field(min_length=2, max_length=2)
    axis2_driver: NonEmptyStr
    axis2_poles: list[NonEmptyStr] = Field(min_length=2, max_length=2)
    independence_note: NonEmptyStr
    pros: NonEmptyStr
    cons: NonEmptyStr

    _j = field_validator("pros", "cons", "independence_note", mode="before")(_join)


class AxisPairs(OutputModel):
    pairs: list[AxisPair] = Field(min_length=1, max_length=3)


class Scenario(OutputModel):
    name: NonEmptyStr
    axis1_pole: NonEmptyStr
    axis2_pole: NonEmptyStr
    narrative: NonEmptyStr = Field(description="5문장 이내")
    common_trends: list[NonEmptyStr] = Field(description="이 시나리오에 반영한 기본 트렌드 driver id 전부")
    winner: NonEmptyStr = Field(description="이 세계에서 유리한 사업자 유형")


class ScenarioMatrix(OutputModel):
    scenarios: list[Scenario] = Field(min_length=4, max_length=4)
    no_regret_moves: list[NonEmptyStr] = Field(min_length=3, max_length=3)
    signposts: list[NonEmptyStr] = Field(min_length=4)
    validation: dict[str, str] = Field(description="개연성·일관성·차별성·관련성·도전성 각각 한 문장")


# ---- 게이트 결정 ----

class AdoptAxes(OutputModel):
    axes: list[NonEmptyStr]
    rationale: NonEmptyStr


class TargetDecision(OutputModel):
    cell: NonEmptyStr
    rationale: NonEmptyStr
    supersedes: NonEmptyStr | None = Field(default=None, description="이전 판단(decision id). 재판단 시 유지/수정 이유를 rationale에")


class AdoptScenarioAxes(OutputModel):
    axis1_driver: NonEmptyStr
    axis1_poles: list[NonEmptyStr] = Field(min_length=2, max_length=2)
    axis2_driver: NonEmptyStr
    axis2_poles: list[NonEmptyStr] = Field(min_length=2, max_length=2)
    adopted_drivers: list[Driver] = Field(description="분석자가 인정한 동인 표(게이트 뒤 단계는 이것만 본다)")
    rationale: NonEmptyStr


SCHEMAS = {c.__name__: c for c in [AxisCandidates, CompareTable, Critique, TargetBrief, PestelTable, AxisPairs,
                                    ScenarioMatrix, AdoptAxes, TargetDecision, AdoptScenarioAxes]}


# ---- 페르소나 · BUM ----

LEVERS = ("고객생산성", "단순성", "편의성", "위험감소", "재미와이미지", "환경친화성")
STAGES = ("구매", "배달", "사용", "보완", "유지", "폐기")


class PersonaCard(OutputModel):
    uuid: NonEmptyStr = Field(description="입력 행의 uuid 그대로")
    name: NonEmptyStr = Field(description="가명")
    goals: NonEmptyStr
    constraints: NonEmptyStr = Field(description="입력 행의 주거·지역·가족·충전 조건에서 나온 제약만")
    context: NonEmptyStr = Field(description="사용 맥락")
    journey: NonEmptyStr = Field(description="구매까지의 여정 한 단락")


class PersonaCards(OutputModel):
    cards: list[PersonaCard] = Field(min_length=2, max_length=4)


class Pain(OutputModel):
    persona_uuid: NonEmptyStr
    lever: NonEmptyStr = Field(description="고객생산성 단순성 편의성 위험감소 재미와이미지 환경친화성 중 하나")
    stage: NonEmptyStr = Field(description="구매 배달 사용 보완 유지 폐기 중 하나")
    text: NonEmptyStr
    grade: NonEmptyStr = Field(description="근거 belief가 있으면 그 등급, 없으면 [가]")
    provenance: list[NonEmptyStr]

    _v = field_validator("lever", "stage", mode="before")(lambda v: NonEmptyStr(v).replace(" ", "").replace("·", ""))


class BUM(OutputModel):
    pains: list[Pain] = Field(min_length=6)
    blocked_utility: list[NonEmptyStr] = Field(description="막힌 효용 공간(레버×단계) 목록")
    latent_needs: list[Claim] = Field(min_length=3, max_length=6, description="잠재·미충족 니즈. 검증 전이므로 grade [가]")


class NeedsDecision(OutputModel):
    needs: list[NonEmptyStr] = Field(min_length=1, description="FGI로 검증할 니즈 문장(latent_needs의 text 그대로)")
    rationale: NonEmptyStr


# ---- FGI ----

class Finding(OutputModel):
    kind: Literal["consensus", "disagreement", "new_hypothesis"]
    text: NonEmptyStr
    cells: list[NonEmptyStr] = Field(min_length=1, description="해당 셀 라벨")
    question_id: NonEmptyStr = Field(description="입력 질문 id 그대로")
    metric: Literal["pos_rate", "share"] = "pos_rate"
    answer: NonEmptyStr | None = Field(default=None, description="share인 경우 집계 dist의 답변 키")
    grade: Literal["[시]"] = "[시]"
    provenance: list[NonEmptyStr] = Field(default_factory=list, description="서버가 run id와 질문 id로 재설정")

    @field_validator("grade", mode="before")
    @classmethod
    def _simulation_grade(cls, value):
        return "[시]"
    level: int = Field(0, ge=0, le=1, description="0 = 순서·방향 주장(A가 B보다 높다) / 1 = 수준 주장(비율·%·n명). 개인 예측·조건 감도 주장은 쓰지 않는다")

    @model_validator(mode="after")
    def _level_from_text(self):
        """수치(%·비율·n명)가 들어간 문장은 LLM 선언과 무관하게 level 1. 잠금 판정은 렌더러가 보정 카드로 한다."""
        import re
        if re.search(r"\d|퍼센트|percent|비율|절반|과반", self.text):
            self.level = 1
        return self


class FGIFindings(OutputModel):
    findings: list[Finding] = Field(min_length=3)
    hypotheses_to_test: list[NonEmptyStr] = Field(min_length=1, description="실데이터로 확인할 가설 문장")
    limits: NonEmptyStr = Field(description="가상 응답자이며 검증 전 가설이라는 한계")


class FGICritique(OutputModel):
    weakest_findings: list[NonEmptyStr]
    alternative_explanations: list[NonEmptyStr]
    what_would_change_my_mind: list[NonEmptyStr]


SCHEMAS.update({c.__name__: c for c in [PersonaCards, BUM, NeedsDecision, FGIFindings, FGICritique]})


# ---- 조사 계획 (워크시트 STEP 8~13) ----

class HypothesesDecision(OutputModel):
    hypotheses: list[NonEmptyStr] = Field(min_length=1, description="실제 조사로 확인할 가설(FGI hypotheses_to_test 에서 고름)")
    rationale: NonEmptyStr


class SurveyQuestion(OutputModel):
    text: NonEmptyStr
    type: Literal["likert5", "multiple_choice", "open", "importance_rating", "psm", "interview_probe"]


class Implication(OutputModel):
    if_: NonEmptyStr = Field(alias="if", description="이런 결과가 나오면")
    then: NonEmptyStr = Field(description="case.options 의 id")

    model_config = {"populate_by_name": True}


class PlanItem(OutputModel):
    hypothesis: NonEmptyStr
    information_need: NonEmptyStr = Field(description="case.information_needs id 또는 새 문장")
    questions: list[SurveyQuestion] = Field(min_length=1)
    analysis: Literal["frequency", "average", "group_comparison", "user_vs_nonuser", "funnel", "importance_vs_satisfaction", "qual_coding"]
    implications: list[Implication] = Field(min_length=2, description="결과별로 case.options 의 어느 선택지로 가는지. 서로 달라야 함")


class ResearchPlan(OutputModel):
    items: list[PlanItem] = Field(min_length=1)
    target_population: NonEmptyStr
    sample_size: int = Field(gt=0)
    sample_rationale: NonEmptyStr
    method: Literal["qualitative", "quantitative", "qual_quant"]


SCHEMAS.update({c.__name__: c for c in [HypothesesDecision, ResearchPlan]})
ANALYSIS_METHODS = {"frequency", "average", "group_comparison", "user_vs_nonuser", "funnel", "importance_vs_satisfaction", "qual_coding"}


# ---- 근거 수집 에이전트 ----

class ApproveEvidence(OutputModel):
    approved: list[NonEmptyStr] = Field(description="월드에 쓸 후보의 열 이름 목록(검사 통과·모순 없음인 것만 고르는 것이 원칙)")
    rationale: NonEmptyStr


SCHEMAS["ApproveEvidence"] = ApproveEvidence
