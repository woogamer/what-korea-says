"""Tavily → [공] belief. 검색 결과(URL·본문)에서 Super가 수치 명제만 추출하고, URL 없는 명제는 [공]으로 들어갈 수 없다(store.add_belief가 거부).

Tavily 3기능 중 Search만 쓴다(밋업 노트 §4). Extract/Crawl은 긴 원문이 필요할 때 후속.
"""
import os

from pydantic import BaseModel, Field
from tavily import TavilyClient

from world.store import add_belief


class Proposition(BaseModel):
    text: str = Field(description="검색 본문에 있는 사실 한 문장. 수치·연도·분모 포함")
    url: str = Field(description="그 문장의 출처 URL. 결과 목록의 URL 그대로")
    quote: str = Field(description="근거가 된 본문 원문 발췌(30자 이내 가능)")
    year: str = Field(description="수치의 시점. 모르면 '[미]'")


class Propositions(BaseModel):
    propositions: list[Proposition] = Field(max_length=5)


def search(query: str, max_results: int = 5) -> list[dict]:
    c = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    r = c.search(query, max_results=max_results, search_depth="basic")
    return [{"url": x["url"], "title": x["title"], "content": x["content"][:1500], "score": x["score"]} for x in r["results"]]


def extract_propositions(router, query: str, results: list[dict]) -> list[Proposition]:
    urls = {x["url"] for x in results}
    msgs = [{"role": "system", "content": "너는 조사 보조원이다. 아래 검색 결과 본문에 실제로 적힌 사실만 명제로 옮긴다. 본문에 없는 수치는 만들지 않는다. JSON만 출력."},
            {"role": "user", "content": f"질의: {query}\n\n검색 결과:\n" + "\n\n".join(
                f"[{i}] URL: {x['url']}\n제목: {x['title']}\n본문: {x['content']}" for i, x in enumerate(results))
             + "\n\n수치·연도·분모가 있는 사실 명제를 최대 5개 추출하라. 각 명제에 출처 URL(위 목록 그대로)과 원문 발췌를 붙여라. 본문이 목록·메뉴뿐이면 그 결과는 건너뛴다."}]
    obj, rec = router.call("extract", msgs, schema=Propositions, max_tokens=4000, tag=f"tavily:{query[:20]}")
    if obj is None:
        return []
    return [p for p in obj.propositions if p.url in urls]   # 목록에 없는 URL은 버림


def search_to_beliefs(con, world_id: str, queries: list[str], router=None) -> list[dict]:
    """질의마다 검색 → 명제 추출 → [공] belief(status=proposed). 승인은 Ingest 게이트 몫."""
    if router is None:
        from llm import Router
        router = Router(con=con, run_id="tavily")
    out = []
    for q in queries:
        results = search(q)
        for p in extract_propositions(router, q, results):
            bid = add_belief(con, world_id, p.text, "[공]", p.url,
                             variables={"query": q, "quote": p.quote, "year": p.year}, status="proposed")
            out.append({"belief_id": bid, "proposition": p.text, "source_ref": p.url, "year": p.year, "quote": p.quote})
    return out
