"""세계판 게임 후보 3종 자동 평가. 플레이어 = Super(추론). 결과는 data/eval_worldwide.json.

G1 Spot the Outsider: 같은 나라 4 + 위장한 외부인 1, 질문 3개 → 지목 적중률(찍기 20%)
G2 PersonaGuessr: 1명이 질문 5개에 답 → 질문 k개 후 나라 top1/top3 적중률(찍기 10%/30%)
G3 Ask the World: 나라별 30명 종교 중요도 → Pew 실측과 비교
실행: .venv/bin/python -m game.eval_worldwide [--only G1,G2,G3] [--per 3]
"""
import argparse
import asyncio
import itertools
import json
import random
import re
from collections import Counter, defaultdict

from game.worldwide import COUNTRIES, NAME, Guess, Pick, ask, load, sample, to_key
from llm.router import Router
from world.config import ROOT

QUESTIONS = ["What did you do last weekend?", "What did you eat for dinner yesterday?",
             "What's your biggest worry these days?", "Tell me about your home and who you live with.",
             "How do you celebrate the most important holiday of your year?"]
COUNTRY_LIST = ", ".join(NAME.get(c, c) for c in COUNTRIES)


def jaccard_mean(texts):
    sets = [set(re.findall(r"[a-z]{4,}", t.lower())) for t in texts if t]
    pairs = list(itertools.combinations(sets, 2))
    return sum(len(a & b) / max(len(a | b), 1) for a, b in pairs) / max(len(pairs), 1)


async def g2(router, con, per, seed):
    people = [p for c in COUNTRIES for p in sample(con, c, per, seed)]
    answers, leak = [], Counter()
    for q in QUESTIONS:
        a, st = await ask(router, people, q)
        answers.append(a); leak.update(st)
    jobs = []
    for i, p in enumerate(people):
        for k in range(1, len(QUESTIONS) + 1):
            qa = "\n".join(f"Q: {QUESTIONS[j]}\nA: {answers[j][i] or '(no answer)'}" for j in range(k))
            jobs.append(dict(role="synthesize", schema=Guess, max_tokens=4000, temperature=0, retries=2, tag="ww-g2-player",
                             messages=[{"role": "user", "content": f"A person from one of these countries answered questions: {COUNTRY_LIST}.\n"
                                        f"{qa}\nWhich country are they from? Give your top 3 guesses, most likely first."}]))
    res = await router.gather(jobs, concurrency=24)
    acc = defaultdict(lambda: [0, 0, 0]); confusion = Counter(); per_country = defaultdict(lambda: [0, 0])
    for n, (g, _) in enumerate(res):
        i, k = divmod(n, len(QUESTIONS)); k += 1
        truth = people[i]["country"]
        top = [to_key(x) for x in g.top3] if g else []
        acc[k][0] += bool(top) and top[0] == truth; acc[k][1] += truth in top; acc[k][2] += 1
        if k == len(QUESTIONS):
            confusion[(truth, top[0] if top else "-")] += 1
            per_country[truth][0] += bool(top) and top[0] == truth; per_country[truth][1] += 1
    homog = {q: round(jaccard_mean(a), 3) for q, a in zip(QUESTIONS, answers)}
    samples = {people[i]["country"]: [answers[j][i] for j in range(len(QUESTIONS))] for i in range(0, len(people), per)}
    return {"n_people": len(people), "acc": {k: {"top1": v[0] / v[2], "top3": v[1] / v[2]} for k, v in sorted(acc.items())},
            "per_country_top1@5": {c: f"{a}/{n}" for c, (a, n) in per_country.items()},
            "confusions": [f"{t}->{g}: {n}" for (t, g), n in confusion.most_common() if t != g][:12],
            "homogeneity_jaccard": homog, "leak": dict(leak), "samples": samples}


async def g1(router, con, games, seed):
    rng = random.Random(seed); out = []
    for gi in range(games):
        home, away = rng.sample(COUNTRIES, 2)
        group = sample(con, home, 4, seed + gi) + sample(con, away, 1, seed + gi)
        order = list(range(5)); rng.shuffle(order); group = [group[o] for o in order]
        spy = order.index(4)
        qa = []
        for q in QUESTIONS[:3]:
            a, _ = await ask(router, group, q, disguise={spy: home})
            qa.append((q, a))
        text = "\n".join(f"Q: {q}\n" + "\n".join(f"  {i + 1}. {x or '(no answer)'}" for i, x in enumerate(a)) for q, a in qa)
        pick, _ = await router.acall("synthesize", [{"role": "user", "content":
            f"Five people answered questions. Four are from {NAME.get(home, home)}; one is from another country pretending to be from there.\n"
            f"{text}\nWho is the outsider?"}], schema=Pick, max_tokens=4000, temperature=0, retries=2, tag="ww-g1-player")
        out.append({"home": home, "away": away, "spy": spy + 1, "pick": pick.outsider if pick else None,
                    "hit": bool(pick) and pick.outsider == spy + 1, "reason": pick.reason if pick else "", "qa": qa})
    return {"games": games, "hit_rate": sum(g["hit"] for g in out) / games, "detail": out}


# (질문, 선택지, 집계할 선택지, 실측 %) — Pew 2025-05-06 종교 중요도(인도·브라질은 다수 종교 신자 수치), Pew 2020-07-20 신과 도덕
G3_QS = [
    ("How important is religion in your life?", ["very important", "somewhat important", "not too important", "not at all important"],
     "very important", {"USA": 38, "Korea": 18, "France": 18, "Japan": 7, "India": 82, "Brazil": 87}),
    ("Which comes closer to your view?", ["It is necessary to believe in God in order to be moral and have good values",
                                          "It is not necessary to believe in God in order to be moral and have good values"],
     "It is necessary to believe in God in order to be moral and have good values",
     {"USA": 44, "Brazil": 84, "India": 79, "Japan": 39, "Korea": 53, "France": 15}),
]


async def g3(router, con, per, seed):
    from typing import Literal

    from pydantic import BaseModel

    from game.worldwide import persona_system
    people = [p for c in COUNTRIES for p in sample(con, c, per, seed + 99)]
    out = []
    for q, opts, target, real in G3_QS:
        class Ans(BaseModel):
            choice: Literal[tuple(opts)]
        jobs = [dict(role="respondent", thinking=False, temperature=0.9, max_tokens=200, retries=2, schema=Ans, tag="ww-g3",
                     messages=[{"role": "system", "content": persona_system(p)},
                               {"role": "user", "content": f"Survey: {q} Choose one: {opts}"}]) for p in people]
        res = await router.gather(jobs)
        share = defaultdict(lambda: [0, 0])
        for p, (r, _) in zip(people, res):
            if r:
                share[p["country"]][0] += r.choice == target; share[p["country"]][1] += 1
        ai = {c: round(100 * a / n) for c, (a, n) in share.items()}
        both = [c for c in real if c in ai]
        n = len(both)
        rank = lambda d: {c: i for i, c in enumerate(sorted(both, key=lambda x: -d[x]))}
        ra, rp = rank(ai), rank(real)
        out.append({"q": q, "ai_pct": ai, "real_pct": real,
                    "spearman": round(1 - 6 * sum((ra[c] - rp[c]) ** 2 for c in both) / (n * (n * n - 1)), 2),
                    "mean_gap_pp(ai-real)": round(sum(ai[c] - real[c] for c in both) / n, 1),
                    "mean_abs_gap_pp": round(sum(abs(ai[c] - real[c]) for c in both) / n, 1)})
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="G1,G2,G3")
    ap.add_argument("--per", type=int, default=3)
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    con = load(); router = Router(budget_usd=3.0); out = {}
    if "G2" in args.only:
        out["G2"] = await g2(router, con, args.per, args.seed); print("G2 done", f"${router.total_cost:.3f}", flush=True)
    if "G1" in args.only:
        out["G1"] = await g1(router, con, args.games, args.seed); print("G1 done", f"${router.total_cost:.3f}", flush=True)
    if "G3" in args.only:
        out["G3"] = await g3(router, con, 30, args.seed); print("G3 done", f"${router.total_cost:.3f}", flush=True)
    out["cost_usd"] = round(router.total_cost, 4); out["calls"] = len(router.calls)
    path = ROOT / "data" / f"eval_worldwide_{args.only.replace(',', '')}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    print(json.dumps({k: v for k, v in out.items()}, ensure_ascii=False, indent=1, default=str)[:6000])


if __name__ == "__main__":
    asyncio.run(main())
