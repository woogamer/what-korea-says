"""의사결정 과제(case): 최상위 객체.

워크시트 구조(문제 → 결정 → 조사 질문 → 정보 필요 → 근거/추가 조사)를 YAML 한 장으로 적는다.
코어(world/)는 case가 참조하는 모집단 층이고, 툴·판단·산출물은 case 아래에 붙는다.
"""
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class Option(BaseModel):
    id: str
    text: str


class Implication(BaseModel):
    if_: str = Field(alias="if")
    then: str  # option id


class ResearchQuestion(BaseModel):
    id: str
    text: str
    information_needs: list[str] = []
    implications: list[Implication] = []


class InformationNeed(BaseModel):
    id: str
    text: str


class EvidenceRef(BaseModel):
    """정보 필요 → 근거. beliefs(팩·환경 belief id) / search(Tavily 질의, 아직 미확인) / primary_research(새로 조사)."""
    beliefs: list[str] = []
    search: str | None = None
    primary_research: str | None = None


class EnvDriver(BaseModel):
    """개인 속성이 아닌 환경 근거(PESTEL 동인 등). 시나리오 툴이 쓴다."""
    id: str
    dimension: str  # P E S T En L
    proposition: str
    grade: str
    source_ref: str = ""


class WorldRef(BaseModel):
    pack: str
    market_scope: str | None = None


class Case(BaseModel):
    name: str
    title: str
    problem: str
    decision: str
    options: list[Option]
    research_questions: list[ResearchQuestion]
    information_needs: list[InformationNeed]
    evidence_map: dict[str, EvidenceRef] = {}
    world: WorldRef
    target_population: str = ""
    controllable: list[str] = []  # 기업이 고를 수 있는 변수. 시나리오 축 금지 목록
    environment: list[EnvDriver] = []


def load_case(path) -> Case:
    return Case.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def check_decision_relevance(case: Case) -> dict:
    """C1. 조사 질문마다 결과별 결정이 실제로 갈리는가.

    워크시트 CHECK: "어떤 결과가 나오든 의사결정이 달라지지 않는다면 조사할 필요가 없는 질문".
    """
    opt_ids = {o.id for o in case.options}
    need_ids = {n.id for n in case.information_needs}
    problems = []
    for rq in case.research_questions:
        thens = [i.then for i in rq.implications]
        if len(thens) < 2:
            problems.append(f"{rq.id}: 결과별 시사점이 2개 미만")
        elif len(set(thens)) < 2:
            problems.append(f"{rq.id}: 어떤 결과든 같은 결정({thens[0]}) → 조사 불필요 질문")
        for t in thens:
            if t not in opt_ids:
                problems.append(f"{rq.id}: 알 수 없는 선택지 {t}")
        if not rq.information_needs:
            problems.append(f"{rq.id}: 정보 필요 항목 없음")
        for n in rq.information_needs:
            if n not in need_ids:
                problems.append(f"{rq.id}: 알 수 없는 정보 필요 {n}")
    return {"passed": not problems, "problems": problems,
            "n_rq": len(case.research_questions), "n_options": len(opt_ids)}


def evidence_coverage(case: Case, beliefs: dict[str, dict]) -> dict:
    """C2(fast). 정보 필요마다 근거(등급·출처) 또는 추가 조사 항목이 연결되었는가.

    beliefs = {belief_id: {"grade", "source_ref"}} (팩 belief + case.environment).
    """
    rows, problems = [], []
    for need in case.information_needs:
        ref = case.evidence_map.get(need.id)
        if ref is None:
            problems.append(f"{need.id}: 매핑 없음"); rows.append({"need": need.id, "status": "unmapped"}); continue
        found = []
        for b in ref.beliefs:
            if b not in beliefs:
                problems.append(f"{need.id}: 알 수 없는 belief {b}"); continue
            g, src = beliefs[b].get("grade"), beliefs[b].get("source_ref") or ""
            if not g or not src:
                problems.append(f"{need.id}: belief {b} 등급/출처 없음")
            if g == "[공]" and not src.startswith("http"):
                problems.append(f"{need.id}: [공] belief {b}에 URL 없음")
            found.append(f"{b}{g}")
        status = "evidence" if found else ("search" if ref.search else ("primary" if ref.primary_research else "empty"))
        if status == "empty":
            problems.append(f"{need.id}: 근거도 추가 조사 항목도 없음")
        rows.append({"need": need.id, "status": status, "beliefs": found,
                     "search": ref.search, "primary_research": ref.primary_research})
    return {"passed": not problems, "problems": problems, "rows": rows,
            "n_evidence": sum(r["status"] == "evidence" for r in rows),
            "n_search": sum(r["status"] == "search" for r in rows),
            "n_primary": sum(r["status"] == "primary" for r in rows)}
