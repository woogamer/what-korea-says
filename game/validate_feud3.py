"""검증 4: '없음/모름'은 무엇이 만드나 — 주제 관여도(문항마다 다름) vs 응답 의지(사람마다 같음).

같은 18문항(validate_feud2: 갤럽 2024 반복 12 + 새 4 + 2026 2), 페르소나 200명.
  A  기본(개인 응답)
  E  관여도 2단계: 먼저 "이 사람이 이 주제에 바로 떠오르는 구체적 답이 있나"만 판정 → 없으면 '없음', 있으면 A로 답
  F  응답 의지: A 응답에서 페르소나 중 고정 비율 p를 '대충 답하는 사람'으로 정해 모든 문항 '없음'.
     p는 반복 12문항에서 고르고 새 4 + 2026 2문항으로 채점
  D  직접 추정(기존)
  D2 직접 추정 2단계: 먼저 '없음/모름' 비율만 추정 → 나머지 응답자 안의 분포 추정 → 합침
지표: 겹침 점수(validate_feud) + '없음' 오차(실측 '없음' 수치가 공개된 5문항 평균 |AI−실측| %p)

사전 판정(실행 전 기록):
  H4 E가 A보다 '없음' 오차 작고 18문항 평균 겹침도 높다
  H5 새 6문항에서 E 겹침 > F(최적 p) 겹침 (관여도가 의지보다 '없음'을 더 잘 설명)
  H6 D2가 D보다 '없음' 오차 작고 18문항 평균 겹침이 같거나 높다
실행: .venv/bin/python -m game.validate_feud3 [--n 200]
"""
import argparse
import asyncio
import hashlib
import json

from pydantic import BaseModel, Field

from game.feud import poll, who
from game.validate_feud import NONE, Dist, direct, one
from game.validate_feud2 import NEW, REPEAT, T24, Y26
from game.worldwide import load, sample
from llm.router import Router
from world.config import ROOT


class Ready(BaseModel):
    has_answer: bool = Field(description="바로 구체적인 답 하나를 말할 수 있으면 true, '없다/모르겠다'라고 할 사람이면 false")


class NoneShare(BaseModel):
    pct: float = Field(description="'없다/모르겠다/관심 없다'로 답할 비율 %")


INVOLVE = ("면접원이 길에서 이렇게 물었습니다: \"{q}\"\n"
           "이 사람의 나이·생활·관심사로 볼 때, 이 주제에 관심이 있거나 아는 게 있어서 바로 구체적인 답 하나를 말할까요, "
           "아니면 '딱히 없다/모르겠다'라고 할까요? 사람들 상당수는 관심 없는 주제에 '없다'라고 답합니다. 현실적으로 판정하세요.")


async def involve(router, people, q):
    jobs = [dict(role="respondent", thinking=True, temperature=1.0, max_tokens=2500, retries=2, schema=Ready, tag="feud-involve",
                 messages=[{"role": "system", "content": who(p).rsplit("\n", 1)[0]}, {"role": "user", "content": INVOLVE.format(q=q)}])
            for p in people]
    res = await router.gather(jobs, concurrency=25)
    yes = [p for p, (a, _) in zip(people, res) if a and a.has_answer]
    no = [p for p, (a, _) in zip(people, res) if not (a and a.has_answer)]
    return (await poll(router, yes, q)) + [{**p, "raw": "없다"} for p in no]


def lazy(p, frac):
    """사람마다 고정된 '대충 답하기' 성향(uuid 해시 → 문항과 무관하게 같은 사람)."""
    return int(hashlib.md5(p["uuid"].encode()).hexdigest()[:8], 16) / 16 ** 8 < frac


async def direct2(router, q, target):
    s, _ = await router.acall("extract", [{"role": "user", "content":
        f"{target} 보기 없이 물었다: \"{q}\"\n이 중 '없다/모르겠다/관심 없다'로 답한 비율을 추정하라. "
        "주제에 대한 관심·지식이 연령·성별에 따라 얼마나 퍼져 있는지 따져서, 누구나 답이 있는 주제면 0에 가깝게, 관심층만 아는 주제면 높게."}],
        schema=NoneShare, thinking=True, max_tokens=6000, temperature=0.3, retries=2, tag="feud-none")
    none = min(max(s.pct if s else 10, 0), 100)
    d, _ = await router.acall("extract", [{"role": "user", "content":
        f"{target} 보기 없이 물었다: \"{q}\"\n'없다/모르겠다'를 뺀, 구체적인 답을 한 사람들 안에서의 자유응답 분포를 추정하라. "
        "고정관념이나 대표 답이 아니라 실제 사람들이 답했을 법한 분포로, 응답 상위 15개 정도를 %로(합 100)."}],
        schema=Dist, thinking=True, max_tokens=6000, temperature=0.3, retries=2, tag="feud-direct2")
    rows = [{"raw": "없다"}] * round(none * 2)
    for sh in (d.shares if d else []):
        if sh.answer != NONE:
            rows += [{"raw": sh.answer}] * round(sh.pct * (100 - none) / 100 * 2)
    return rows


BANK = [("repeat24", q, real, T24) for q, real, _ in REPEAT] + [("new24", q, real, T24) for q, real in NEW.items()] + \
       [("y2026", q, real, t) for q, real, t in Y26]
P_GRID = [0.0, 0.1, 0.2, 0.3]


async def main(n, seed):
    router = Router(budget_usd=3.0)
    people = sample(load(), "Korea", n, seed)
    out = []
    for part, q, real, target in BANK:
        print(f"\n[{part}] {q}", flush=True)
        a_rows = await poll(router, people, q)
        variants = {"A": a_rows, "E": await involve(router, people, q), "D": await direct(router, q, target),
                    "D2": await direct2(router, q, target)}
        for p in P_GRID[1:]:
            variants[f"F{int(p * 100)}"] = [{**r, "raw": "없다"} if lazy(r, p) else r for r in a_rows]
        for st, rows in variants.items():
            r = await one(router, people, q, real, st, target, rows=rows)
            rec = {"part": part, "q": q, "style": st, "score": r["score"], "none_ai": r["ai"].get(NONE), "none_real": real.get(NONE)}
            out.append(rec)
            nn = f" · 없음 {rec['none_ai']:g}/{rec['none_real']:g}" if rec["none_real"] is not None and NONE in real else ""
            print(f"  {st:<4} 겹침 {r['score']:.2f}{nn}", flush=True)

    def avg(st, parts, key="score"):
        xs = [o[key] for o in out if o["style"] == st and o["part"] in parts]
        return sum(xs) / len(xs)

    def none_err(st):
        xs = [abs(o["none_ai"] - o["none_real"]) for o in out if o["style"] == st and o["none_real"] is not None]
        return sum(xs) / len(xs)

    styles = list(dict.fromkeys(o["style"] for o in out))
    allp, newp = ["repeat24", "new24", "y2026"], ["new24", "y2026"]
    print("\n=== 결과 ===")
    for st in styles:
        print(f"{st:<4} 전체 {avg(st, allp):.3f} · 반복12 {avg(st, ['repeat24']):.3f} · 새6 {avg(st, newp):.3f} · 없음 오차 {none_err(st):.1f}%p")
    best = max(["A"] + [f"F{int(p * 100)}" for p in P_GRID[1:]], key=lambda s: avg(s, ["repeat24"]))
    print(f"F 최적 p(반복12 기준) = {best} → 새6 {avg(best, newp):.3f} vs E {avg('E', newp):.3f}")
    (ROOT / "data" / "feud_validation3.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"비용 ${router.total_cost:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    asyncio.run(main(a.n, a.seed))
