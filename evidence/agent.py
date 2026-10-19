"""근거 수집 에이전트: 빈 열([미]) → 검색 계획(Super) → Tavily Search+Extract → 조건부 확률표 추출(Super) → 규칙 검사 → 모순 검사(Ultra) → 제안.

에이전트가 자율적인 것이 아니라 "제안하고, 구조가 검사하고, 사람이 승인"한다. 검사에 걸리면 사유를 다음 질의에 넣어 다시 찾는다(최대 rounds).
페르소나 DB의 L0 열(시도·시군구·성·연령대·주거·학력·혼인)이 한국 공개 통계의 표준 축과 같으므로, 찾은 표를 축만 맞추면 100만 명에 전개할 수 있다.
"""
import os

from pydantic import BaseModel, Field

from world.store import add_belief, build_view, view_name

COVERAGE_MIN = 0.8      # 표의 by 값들이 시장 범위 인구의 이만큼을 덮어야 함
SUM_TOL = 0.05          # 행의 확률 합 허용 오차(넘으면 분모 문제)


class SearchPlan(BaseModel):
    queries: list[str] = Field(min_length=1, max_length=3, description="한국어 검색어. 통계 출처(통계청·서울열린데이터광장·정부 보도자료·조사 보고서)가 나오게")
    rationale: str


class StatRow(BaseModel):
    by: dict[str, str] = Field(description="축 열 → 값. 값은 반드시 제시된 L0 값 목록 중 하나로 매핑(예: '20대' → '19-29'). 원문 구간이 우리 값보다 넓으면(예: 20~39세) 포함되는 값마다 행을 하나씩 만든다(19-29, 30-39 각각). 축별 표가 없고 전체 비율만 있으면 빈 dict {} 로 한 행(주변분포)")
    p: dict[str, float] = Field(description="열 값 → 비율(0~1). 합 1")
    source_label: str = Field(default="", description="원문의 축 표기 그대로(예: '청년기 (20~39세)'). 주변분포 행은 '전체'")


class StatTable(BaseModel):
    found: bool = Field(description="검색 본문에 축별 비율이 실제로 있었는가. 없으면 false 이고 rows 는 비워 둔다")
    rows: list[StatRow] = Field(default_factory=list)
    url: str = Field(default="", description="표의 출처 URL. 결과 목록의 URL 그대로")
    year: str = Field(default="[미]", description="수치 시점")
    n: int | None = Field(default=None, description="표본 수(있으면)")
    sample_condition: str | None = Field(default=None, description="누구를 조사했나(지역·연령·방법). 본문에 없으면 null")
    quote: str = Field(default="", description="근거 원문 발췌")
    note: str = Field(default="", description="본문 수치를 어떻게 우리 값으로 옮겼는지(예: '자가용 이용 34% → car 0.34')")


def l0_domain(con, world_id, cols, scope) -> dict:
    """축 열마다 시장 범위 안 값 목록과 인구 비중."""
    v = view_name(world_id); build_view(con, world_id)
    out = {}
    for c in cols:
        rows = con.execute(f"SELECT {c}, CAST(sum(w) AS DOUBLE) FROM {v} WHERE ({scope}) AND {c} IS NOT NULL GROUP BY 1 ORDER BY 2 DESC").fetchall()
        tot = sum(r[1] for r in rows) or 1.0
        out[c] = {str(r[0]): round(r[1] / tot, 4) for r in rows}
    return out


def plan_queries(router, req: dict, scope_text: str, feedback: str | None = None) -> SearchPlan:
    msgs = [{"role": "system", "content": "너는 시장조사 리서처다. 한국 공개 통계에서 아래 열의 축별 비율표를 찾을 검색어를 만든다. JSON만 출력."},
            {"role": "user", "content": f"열: {req['name']} ({req.get('description', '')}) 값: {req['domain']}\n축: {req['by']}\n시장 범위: {scope_text}\n"
                                        f"힌트: {req.get('query_hint', '')}\n" + (f"이전 시도 실패 사유: {feedback}\n" if feedback else "")
                                        + "검색어 1~3개. 표가 있는 페이지(보도자료·통계 보고서·데이터 포털)가 나오도록 구체적으로."}]
    obj, _ = router.call("synthesize", msgs, schema=SearchPlan, max_tokens=2500, tag=f"agent:plan:{req['name']}")
    return obj or SearchPlan(queries=[f"{req.get('query_hint') or req['name']} {' '.join(req['by'])}별 비율 통계"], rationale="fallback")


def fetch(query: str, k: int = 5, extract_top: int = 4, max_chars: int = 8000) -> list[dict]:
    """Tavily Search → 상위 결과 Extract(본문 전체)로 보강. 표는 검색 요약에 잘 안 잡히므로 Extract가 핵심."""
    from tavily import TavilyClient
    c = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    r = c.search(query, max_results=k, search_depth="advanced")
    results = [{"url": x["url"], "title": x["title"], "content": x["content"][:1500], "score": x["score"]} for x in r["results"]]
    try:
        ex = c.extract(urls=[x["url"] for x in results[:extract_top]])
        full = {x["url"]: x.get("raw_content", "") for x in ex.get("results", [])}
        for x in results:
            if full.get(x["url"]):
                x["content"] = full[x["url"]][:max_chars]
    except Exception as e:   # Extract 실패는 Search 요약으로 진행
        results.append({"url": "", "title": "extract_error", "content": str(e)[:200], "score": 0})
    return [x for x in results if x["url"]]


def extract_table(router, req: dict, results: list[dict], domains: dict) -> StatTable:
    msgs = [{"role": "system", "content": "너는 통계표를 구조화하는 분석가다. 본문에 실제로 적힌 수치만 옮기고, 없으면 found=false. 축 값은 반드시 제시된 목록 값으로만 매핑한다. JSON만 출력."},
            {"role": "user", "content": f"만들 열: {req['name']} 값 목록: {req['domain']} ({req.get('description', '')})\n축: {req['by']}\n"
                                        f"축 값 목록(이 값으로만 매핑, 다른 표기는 가장 가까운 값으로): { {c: list(d) for c, d in domains.items()} }\n\n검색 결과:\n"
                                        + "\n\n".join(f"[{i}] URL: {x['url']}\n제목: {x['title']}\n본문: {x['content']}" for i, x in enumerate(results))
                                        + f"\n\n축 값마다 {req['name']} 값의 비율(합 1)을 표로 옮겨라. 여러 결과에 표가 있으면 가장 최근·가장 넓은 표 하나만. "
                                          "본문에 없는 축 값은 행을 만들지 않는다(추정 금지). 값 목록 중 본문에 없는 나머지 값(예: other)은 1에서 빼서 채운다. 전체(시장 범위 전체) 비율이 본문에 있으면 축별 표와 별도로 by={} 인 행 하나를 항상 추가한다(주변분포). "
                                          "출처 URL·연도(YYYY)·표본·조사 대상을 적어라."}]
    obj, _ = router.call("extract", msgs, schema=StatTable, max_tokens=6000, tag=f"agent:table:{req['name']}")
    return obj or StatTable(found=False, note="추출 실패")


def _norm_p(p: dict, domain: list) -> dict:
    """열 값 표기 정규화(대소문자·공백). domain 값과 소문자 비교로 맞춘다. 못 맞추는 키는 그대로 두어 검사에서 걸리게 한다."""
    low = {str(d).lower(): d for d in domain}
    return {low.get(str(k).strip().lower(), k): v for k, v in p.items()}


def _in_source(p: dict, text: str) -> bool:
    """행의 비율 중 하나라도 원문에 숫자로 있어야 한다(34.2% → '34.2' 또는 '34'). 나머지 값은 1에서 뺀 것일 수 있으므로 하나면 충분."""
    import re
    if not text:
        return True
    for v in p.values():
        pct = v * 100
        for cand in {f"{pct:.1f}", f"{pct:.0f}", f"{round(pct, 1):g}"}:
            if re.search(rf"(?<![\d.]){re.escape(cand)}\s*%", text) or re.search(rf"(?<![\d.]){re.escape(cand)}(?![\d])", text):
                return True
    return False


def _norm_url(u: str) -> str:
    import re
    u = (u or "").strip().lower()
    u = re.sub(r"^https?://", "", u); u = re.sub(r"^(m|www)\.", "", u); u = u.split("?")[0].split("#")[0].rstrip("/")
    return u


def check_table(table: StatTable, req: dict, domains: dict, urls: set, source_text: str = "") -> dict:
    """규칙 검사(LLM 없음). 통과 못 하면 사유를 돌려주고, 통과하면 impute_stat 용 table 을 만든다.

    source_text 를 주면 각 행의 수치가 출처 원문에 실제로 있는지 대조한다(추출 모델의 수치 창작 방지)."""
    import re
    problems, warnings, rows, covered = [], [], [], 0.0
    if not table.found or not table.rows:
        return {"passed": False, "problems": ["표 없음"], "warnings": [], "coverage": 0.0, "rows": [], "evidence_level": None}
    if _norm_url(table.url) not in {_norm_url(u) for u in urls}:
        problems.append(f"출처 URL이 검색 결과에 없음: {table.url!r}")
    cond = [r for r in table.rows if r.by]
    marginal = [r for r in table.rows if not r.by]
    if cond:
        res = _check_conditional(cond, req, domains, source_text, list(problems))
        if res["passed"]:
            if not re.search(r"\d{4}", table.year or ""):
                res["problems"].append("연도 없음"); res["passed"] = False
            return res
        if not marginal:
            if not re.search(r"\d{4}", table.year or ""):
                res["problems"].append("연도 없음")
            return res
        warnings.append("조건부 표 검사 실패 → 주변분포로 강등: " + "; ".join(res["problems"])[:200])
    if marginal:   # 조건부 표 없음/불합격 → 주변분포 한 행. 축 차이가 없는 약한 근거지만 [미]보다 낫다
        r = marginal[0]; r.p = _norm_p(r.p, req["domain"]); s = sum(r.p.values())
        bad_val = [k for k in r.p if k not in req["domain"]]
        if bad_val:
            problems.append(f"열 값이 domain 밖: {bad_val}")
        elif abs(s - 1) > SUM_TOL:
            problems.append(f"주변분포 비율 합 {s:.2f} (분모 문제)")
        elif not _in_source(r.p, source_text):
            problems.append("주변분포 수치가 출처 원문에 없음")
        else:
            p = {k: round(v / s, 4) for k, v in r.p.items()}; p[max(p, key=p.get)] = round(p[max(p, key=p.get)] + (1 - sum(p.values())), 4)
            rows.append({"by": {}, "p": p}); covered = 1.0
            warnings.append(f"조건부 표 없음 → 주변분포(축 {req['by']} 차이 없음)")
        if not re.search(r"\d{4}", table.year or ""):
            problems.append("연도 없음")
        return {"passed": not problems and bool(rows), "problems": problems, "warnings": warnings, "coverage": covered, "rows": rows,
                "evidence_level": "marginal" if rows else None}
    return {"passed": False, "problems": problems + ["표 없음"], "warnings": warnings, "coverage": 0.0, "rows": [], "evidence_level": None}


def _check_conditional(cond, req, domains, source_text, problems):
    rows, covered, warnings = [], 0.0, []
    seen = set()
    for r in cond:
        if set(r.by) != set(req["by"]):
            problems.append(f"축 불일치 {list(r.by)} ≠ {req['by']}"); continue
        bad_by = [f"{c}={v}" for c, v in r.by.items() if v not in domains.get(c, {})]
        if bad_by:
            problems.append("축 값 매핑 실패: " + ", ".join(bad_by)); continue
        r.p = _norm_p(r.p, req["domain"])
        bad_val = [k for k in r.p if k not in req["domain"]]
        if bad_val:
            problems.append(f"열 값이 domain 밖: {bad_val}"); continue
        s = sum(r.p.values())
        if abs(s - 1) > SUM_TOL:
            problems.append(f"{r.by}: 비율 합 {s:.2f} (분모 문제)"); continue
        if not _in_source(r.p, source_text):
            problems.append(f"{r.by}: 수치가 출처 원문에 없음(창작 의심)"); continue
        key = tuple(r.by[c] for c in req["by"])
        if key in seen:
            continue
        seen.add(key)
        p = {k: round(v / s, 4) for k, v in r.p.items()}
        p[max(p, key=p.get)] = round(p[max(p, key=p.get)] + (1 - sum(p.values())), 4)
        rows.append({"by": dict(r.by), "p": p})
        share = 1.0
        for c in req["by"]:
            share *= domains[c].get(r.by[c], 0.0)
        covered += share if len(req["by"]) == 1 else share   # 다축은 독립 가정 근사
    if rows and covered < COVERAGE_MIN:
        problems.append(f"인구 커버리지 {covered:.0%} < {COVERAGE_MIN:.0%}: 빠진 축 값 " + str([v for v in domains[req['by'][0]] if v not in {r['by'][req['by'][0]] for r in rows}][:6]))
    if not rows:
        problems.append("유효한 조건부 행 없음")
    labels = [r.source_label for r in cond if r.source_label]
    if labels and len(set(labels)) < len(rows):
        warnings.append(f"축 해상도 낮음: 원문 구간 {len(set(labels))}개를 우리 값 {len(rows)}개에 나눠 배정(구간 안 균일 가정)")
    return {"passed": not problems and bool(rows), "problems": problems, "warnings": warnings, "coverage": round(covered, 3), "rows": rows,
            "evidence_level": "conditional" if rows else None}


def conflicts(router, table: StatTable, req: dict, existing: list[dict]) -> list[str]:
    if not existing:
        return []
    msgs = [{"role": "system", "content": "너는 회의적인 검토자다. 같은 대상에 대해 동시에 성립할 수 없는 수치만 찾는다. JSON만 출력."},
            {"role": "user", "content": f"새 표({req['name']} by {req['by']}): {[r.model_dump() for r in table.rows]} 출처 {table.url} {table.year}\n기존 belief: {existing}\n\n모순 쌍과 이유. 없으면 빈 목록."}]
    from .ingest import Conflicts
    obj, _ = router.call("conflict", msgs, schema=Conflicts, max_tokens=2500, tag=f"agent:conflict:{req['name']}")
    return obj.conflicts if obj else []


def collect(con, world_id, router, req: dict, scope: str, rounds: int = 2) -> dict:
    """열 하나에 대한 에이전트 루프. 결과는 후보(표·검사·모순·추적)이며 월드에는 아직 아무것도 쓰지 않는다."""
    domains = l0_domain(con, world_id, req["by"], scope)
    trace, feedback, best = [], None, None
    for rnd in range(1, rounds + 1):
        plan = plan_queries(router, req, scope, feedback)
        queries = list(dict.fromkeys(plan.queries + ([req["query_hint"]] if req.get("query_hint") else [])))   # 힌트 질의는 항상 포함
        results = []
        for q in queries:
            try:
                results += fetch(q)
            except Exception as e:
                trace.append({"round": rnd, "action": "search_error", "query": q, "error": str(e)[:200]})
        dedup = {x["url"]: x for x in results}; results = sorted(dedup.values(), key=lambda x: -len(x["content"]))[:8]   # 본문이 긴(Extract 성공) 페이지 우선
        trace.append({"round": rnd, "action": "search", "queries": queries, "n_results": len(results), "urls": [x["url"] for x in results]})
        table = extract_table(router, req, results, domains)
        src = next((x["content"] for x in results if _norm_url(x["url"]) == _norm_url(table.url)), "")
        chk = check_table(table, req, domains, {x["url"] for x in results}, source_text=src)
        trace.append({"round": rnd, "action": "extract+check", "found": table.found, "n_rows": len(table.rows), "url": table.url, "year": table.year,
                      "coverage": chk["coverage"], "problems": chk["problems"]})
        rank = lambda c: (c["passed"], c.get("evidence_level") == "conditional", c["coverage"])
        if best is None or rank(chk) > rank(best["checks"]):
            best = {"table": table.model_dump(), "checks": chk}
        if chk["passed"] and chk["evidence_level"] == "conditional":
            break
        feedback = "; ".join(chk["problems"] + chk.get("warnings", [])) or "축별 표를 못 찾음"
        if chk["passed"]:
            feedback += " — 축별(연령대별 등) 표가 있는 출처를 찾아라"
    existing = [dict(zip(["id", "proposition", "grade"], x)) for x in con.execute("SELECT belief_id, proposition, grade FROM belief WHERE world_id=? AND status='committed'", [world_id]).fetchall()]
    t = StatTable.model_validate(best["table"])
    conf = conflicts(router, t, req, existing) if best["checks"]["passed"] else []
    return {"attr": req["name"], "req": req, "domains": domains, **best, "conflicts": conf, "trace": trace,
            "grade_if_committed": "[도]", "belief_grade": "[공]"}


def apply(con, world_id, cand: dict, actor: str = "evidence_agent") -> dict:
    """승인된 후보를 월드에 쓴다: [공] belief(URL) → impute_stat([도]) + variance 검사. 검사 실패면 연산이 rejected 로 남고 열은 비활성."""
    from world import ops
    t, req = cand["table"], cand["req"]
    by = req["by"] if cand["checks"].get("evidence_level") == "conditional" else []
    prop = (f"{req['name']}: {' × '.join(by)}별 비율" if by else f"{req['name']}: 전체 비율(주변분포, 축 차이 미반영)") + f" ({t['year']}, {req['domain']})"
    bid = add_belief(con, world_id, prop, "[공]", t["url"], variables={"target": req["name"], "by": by, "quote": t["quote"], "note": t["note"],
                                                                       "evidence_level": cand["checks"].get("evidence_level")},
                     n=t.get("n"), sample_condition=t.get("sample_condition"))
    oid, r = ops.run(con, world_id, "impute_stat", {"attr": req["name"], "by": by, "table": cand["checks"]["rows"], "grade": "[도]",
                                                    "domain": req["domain"], "source_beliefs": [bid], "seed": 17,
                                                    "checks": [{"type": "variance", "attr": req["name"]}]}, actor=actor)
    status = con.execute("SELECT status FROM op_log WHERE op_id=?", [oid]).fetchone()[0]
    return {"belief_id": bid, "op_id": oid, "status": status, "checks": r.get("checks", [])}
