"""검증 5: 페르소나를 늘리면 나아지나 — Nemotron-Personas-Korea 1000명(Nano, 스타일 A)이 갤럽 18문항에 한 번씩 답 → 50·100·200·500·1000명 부분표본(각 20회) 겹침.
결과(10/29, $1.13): 0.494 · 0.502 · 0.514 · 0.516 · 0.517 → 200명 위로는 그대로. 쏠림(벚나무 84%·구글 82%·숫자 7 83%·토요일 75%)과 '없음' 0%가 원인.
실행: .venv/bin/python -m game.validate_persona_n → data/persona_n.json (이어 하기 가능)
"""
import asyncio, json, random, re
from game.validate_feud import NONE
from game.feud import poll, judge
from game.validate_feud2 import NEW, REPEAT, Y26
from game.worldwide import load, sample
from llm.router import Router
from world.config import ROOT

N, SIZES, DRAWS = 1000, [50, 100, 200, 500, 1000], 20
OUT = ROOT / "data" / "persona_n.json"

KO_NUM = dict(zip("일 이 삼 사 오 육 칠 팔 구 십".split() + "하나 둘 셋 넷 다섯 여섯 일곱 여덟 아홉 열".split(), [str(i) for i in range(1, 11)] * 2))


def digits(keys, uniq):
    """숫자 문항은 판정을 믿지 않고 글자로 배정(판정이 7을 7번째 칸으로 읽음)."""
    if not all(re.fullmatch(r"\d+", k) for k in keys if k != NONE):
        return {}
    out = {}
    for u in uniq:
        x = re.sub(r"[^0-9가-힣]", "", u)
        x = KO_NUM.get(x, x)
        out[u] = keys.index(x) if x in keys else -1
    return out


async def main():
    router = Router(budget_usd=6.0)
    people = sample(load(), "Korea", N, 29)
    qs = [(q, r) for q, r, _ in REPEAT] + list(NEW.items()) + [(q, r) for q, r, _ in Y26]
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    for q, real in qs:
        if q in out:
            fix = digits(list(real), sorted(set(out[q]["raw"])))
            if fix:
                out[q]["ans"] = [list(real)[fix[r]] if fix[r] >= 0 else None for r in out[q]["raw"]]
            continue
        rows = await poll(router, people, q)
        keys = list(real); uniq = sorted({r["raw"] for r in rows})
        m = dict(zip(uniq, await asyncio.gather(*[judge(router, q, keys, u) for u in uniq])))
        m.update(digits(keys, uniq))
        out[q] = {"real": real, "ans": [keys[m[r["raw"]]] if m[r["raw"]] >= 0 else None for r in rows], "raw": [r["raw"] for r in rows]}
        OUT.write_text(json.dumps(out, ensure_ascii=False))
        print(q[:20], len(rows), f"${router.total_cost:.2f}", flush=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False))
    rnd = random.Random(0)
    def score(real, ans):
        known = [k for k in real if real[k] is not None]
        return sum(min(real[k], 100 * ans.count(k) / len(ans)) for k in known) / sum(real[k] for k in known)
    for n in SIZES:
        s = [sum(score(v["real"], rnd.sample(v["ans"], min(n, len(v["ans"])))) for _ in range(1 if n >= N else DRAWS)) / (1 if n >= N else DRAWS) for v in out.values()]
        print(f"n={n}: {sum(s) / len(s):.3f}")
    print(f"비용 ${router.total_cost:.2f}")

asyncio.run(main())
