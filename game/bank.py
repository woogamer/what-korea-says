"""게임 문제 은행(1단계): 문항마다 AI 사전 정답판(영어 화면용) + 있으면 갤럽 실측.

- 갤럽 18문항(2024 반복 12·새 4, 2026 2): 최신 조사만. AI 사전 = evolve 캐시의 직접 추정 3회(추가 비용 0)
- AI 전용 새 문항: 직접 추정 3회(캐시 data/feud_bank_cache.json)
- 3회 평균 → Super가 영어 번역 + 쓰레기 칸('다 좋다'·'기타') 표시 + 같은 답 묶기 → 답한 사람 기준 100%
- κ = evolve 묶음 빼기 적합의 중앙값(AI 추정 = 실제 응답자 몇 명분). 게임은 θ̂ = (κ·prior + 플레이어 수) / (κ + n)
확인 기준: 정답판(상위 8칸)에 쓰레기 칸 0개, 중복 칸 0개.
층1 보정(τ)은 E1에서 이득 +0.02로 기준 미달 → 적용 안 함.
실행: .venv/bin/python -m game.bank → data/feud_bank.json
"""
import asyncio
import json
import re

import numpy as np
from pydantic import BaseModel, Field

from game.evolve import CACHE as EVOLVE_CACHE
from game.feud import BOARD, TARGET, Match, estimate
from game.validate_feud import NONE
from game.validate_feud2 import NEW, REPEAT, T24, Y26
from llm.router import Router
from world.config import ROOT

OUT = ROOT / "game" / "feud_bank.json"   # 게임 배포용(커밋). 갤럽 수치는 공개 보고서 인용
CACHE = ROOT / "data" / "feud_bank_cache.json"
JUNK_RE = re.compile(r"없|모르|모름|관심|모두|다 좋|전부|기타|상관|따지지|그때그때|딱히")
GALLUP = "https://www.gallup.co.kr/gallupdb/reportContent.asp?seqNo="
SRC24 = {"음식": 1480, "문화": 1481, "일상": 1478}
SEQ24 = {"한국음식": "음식", "중국음식": "음식", "과일주스": "음식", "취미": "문화", "운동": "문화", "관전": "문화"}

# 사람 확인 후 수동 수정(한국어 답 기준이라 번역이 바뀌어도 재현됨)
MERGE = {   # (옮길 칸의 한국어 답, 받을 칸 한국어 답에 들어 있는 글자)
"g24-6": [("눈 오는 겨울", "겨울")], "ai-7": [("늦게 들어오지 마", "시간"), ("밥 먹어", "식사")], "ai-5": [("저축/예금하기", "저축/투자")]}
NOMAP = {"g24-7": ["백합"]}   # 갤럽 항목에 잘못 배정된 칸(백합→수선화)

# AI 전용(실측 없음) — 정치·인물 평가 제외, 해외 심사위원도 알아들을 일상 문항
FRESH = [
    "야식으로 가장 자주 먹는 음식은 무엇입니까?",
    "비 오는 날 가장 먹고 싶은 음식은 무엇입니까?",
    "편의점에서 가장 자주 사는 것은 무엇입니까?",
    "스트레스를 받을 때 가장 먼저 하는 일은 무엇입니까?",
    "월급날 가장 먼저 하고 싶은 일은 무엇입니까?",
    "로또 1등에 당첨되면 가장 먼저 할 일은 무엇입니까?",
    "명절에 친척에게 가장 듣기 싫은 말은 무엇입니까?",
    "엄마에게 가장 많이 듣는 잔소리는 무엇입니까?",
    "가장 가 보고 싶은 해외 여행지는 어디입니까?",
    "출퇴근길이나 등하굣길에 주로 하는 일은 무엇입니까?",
    "치킨을 시킬 때 가장 좋아하는 메뉴는 무엇입니까?",
    "잠들기 전에 마지막으로 하는 일은 무엇입니까?",
]


class Item(BaseModel):
    en: str = Field(description="English label for the quiz board, short (1-4 words); keep Korean dish/brand names romanized with a gloss, e.g. 'Kimchi stew'")
    same_as: int = Field(description="index of an EARLIER item this is a variant of (synonym, spelling, adjective added: '선선한 가을'→'가을', '양념'→'양념치킨'), else -1")


class Clean(BaseModel):
    question_en: str = Field(description="natural English quiz-show question, e.g. 'Name the season Koreans like most.'")
    items: list[Item]


class Labels(BaseModel):
    en: list[str]


def norm(s):
    return re.sub(r"[^0-9a-z\uac00-\ud7a3]", "", s.lower())


async def same_slot(router, q, menu, text):
    """text가 menu(칸 이름 목록) 중 어느 칸과 같은 답인지. 글자 같으면 바로, 아니면 Super(추론 끔). 없으면 -1.
    feud.judge는 한국어 전용이고 '단어 겹쳐도 다르면 -1' 규칙이 강해 '눈 오는 겨울'↔'겨울'도 거절해서 새로 둠."""
    n = norm(text)
    for i, m in enumerate(menu):
        if n and any(n == norm(part) for part in re.split(r"\s*/\s*", m) + [m]):
            return i
    # 숫자 답('2', '4', '2월')은 AI 심판이 칸 번호와 헷갈려서(2 → 5월, 4 → 1) 숫자가 같은 칸만 인정
    digits = set(re.findall(r"\d+", text))
    if digits:
        return next((i for i, m in enumerate(menu) if set(re.findall(r"\d+", m)) == digits), -1)
    listing = "\n".join(f"{i}: {m}" for i, m in enumerate(menu))
    r, _ = await router.acall("extract", [{"role": "user", "content":
        f"Quiz question: {q}\nBoard slots:\n{listing}\nAnswer typed: \"{text}\"\n"
        "Return the slot number if the typed answer names the same thing as a slot (translation, synonym, typo, extra adjective like "
        "'snowy winter' = 'winter', 'my mom's kimchi stew' = 'kimchi stew'). Return -1 if it names a different thing "
        "(different dish, place, brand, breed, day, number — e.g. 'sundubu stew' is NOT 'kimchi stew', 'kimchi' is NOT 'kimchi stew')."}],
        schema=Match, thinking=False, max_tokens=100, temperature=0, retries=2, tag="bank-same")
    return r.slot if r and 0 <= r.slot < len(menu) else -1


def source(q, part):
    if part == "2026":
        seq = 1630 if "야구" in q else 1649
        return {"name": "Gallup Korea", "year": 2026, "url": GALLUP + str(seq),
                "sample": "1,000 adults 18+, phone interview" if seq == 1630 else "1,700 people 13+, face-to-face"}
    kind = next((v for k, v in SEQ24.items() if k in q), "일상")
    return {"name": "Gallup Korea, 'What Koreans Like' 2024", "year": 2024, "url": GALLUP + str(SRC24[kind]),
            "sample": "1,777 people 13+, face-to-face, open-ended"}


def questions():
    out = [dict(id=f"g24-{i}", ko=q, real=r, target=T24, part="2024", key=f"2024|{q}") for i, (q, r, _) in enumerate(REPEAT)]
    out += [dict(id=f"g24n-{i}", ko=q, real=r, target=T24, part="2024", key=f"2024new|{q}") for i, (q, r) in enumerate(NEW.items())]
    out += [dict(id=f"g26-{i}", ko=q, real=r, target=t, part="2026", key=f"2026|{q}") for i, (q, r, t) in enumerate(Y26)]
    out += [dict(id=f"ai-{i}", ko=q, real=None, target=TARGET, part="ai", key=f"ai|{q}") for i, q in enumerate(FRESH)]
    return out


async def runs(router, q, cache, evolve):
    if q["key"] in evolve:   # [[답, %, 실측 칸], ...] × 3
        return [[(a, p) for a, p, _ in run] for run in evolve[q["key"]]]
    if q["key"] not in cache:
        cache[q["key"]] = [await estimate(router, q["ko"], q["target"]) for _ in range(3)]
        CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1))
    return cache[q["key"]]


def average(rs):
    acc = {}
    for run in rs:
        for a, p in run:
            acc[a.strip()] = acc.get(a.strip(), 0) + p / len(rs)
    return sorted(acc.items(), key=lambda t: -t[1])


async def clean(router, q, avg):
    listing = "\n".join(f"{i}. {a}" for i, (a, _) in enumerate(avg))

    def check(c: Clean):
        if len(c.items) != len(avg):
            raise ValueError(f"need exactly {len(avg)} items in the same order, got {len(c.items)}")
        if any(not (-1 <= it.same_as < i) for i, it in enumerate(c.items)):
            raise ValueError("same_as must point to an earlier index or -1")
    c, _ = await router.acall("extract", [{"role": "user", "content":
        f"Korean survey question: {q}\nAnswers (Korean, in order):\n{listing}\n\n"
        "For a quiz show shown to English speakers: translate the question and each answer, "
        "and mark an answer that is a variant of an earlier one (same_as). Different dishes/places/brands are NOT the same."}],
        schema=Clean, validator=check, thinking=False, max_tokens=4000, temperature=0, retries=3, tag="bank-clean")
    return c


async def translate(router, q, words):
    def check(x: Labels):
        if len(x.en) != len(words):
            raise ValueError(f"need exactly {len(words)} labels")
    x, _ = await router.acall("extract", [{"role": "user", "content":
        f"Survey question: {q}\nTranslate each answer to a short English quiz-board label (romanize Korean names with a gloss), same order:\n"
        + "\n".join(f"{i}. {w}" for i, w in enumerate(words))}],
        schema=Labels, validator=check, thinking=False, max_tokens=2000, temperature=0, retries=3, tag="bank-translate")
    return x.en


def reality(real):
    """갤럽 공개 수치 → 답한 사람 기준 %(없음 제외). 수치 미공개 항목은 뺌."""
    none = real.get(NONE) or 0
    xs = [(k, v) for k, v in real.items() if v is not None and k != NONE]
    return [(k, 100 * v / (100 - none)) for k, v in sorted(xs, key=lambda t: -t[1])], none


async def build_one(router, q, cache, evolve):
    avg = average(await runs(router, q, cache, evolve))
    junk = [a for a, _ in avg if JUNK_RE.search(a)]
    avg = [(a, p) for a, p in avg if not JUNK_RE.search(a)][:20]
    c = await clean(router, q["ko"], avg)
    merged = {}
    for i, ((ko, p), it) in enumerate(zip(avg, c.items)):
        root = i
        while c.items[root].same_as >= 0:
            root = c.items[root].same_as
        root = next((j for j in range(root) if c.items[j].en.lower() == c.items[root].en.lower()), root)   # 영어 이름 같으면 같은 칸
        if root in merged:
            merged[root]["pct"] += p; merged[root]["alias_ko"].append(ko)
        else:
            merged[root] = {"ko": ko, "en": it.en, "pct": p, "alias_ko": []}
    # 묶음 재확인: 번역 호출이 번호를 잘못 붙임('눈 오는 겨울'→봄) → judge가 같은 칸이라 할 때만 남기고 아니면 따로 칸
    pct_of = dict(avg)
    for root, x in list(merged.items()):
        for al in list(x["alias_ko"]):
            if await same_slot(router, q["ko"], [x["ko"]], al) != 0:
                x["alias_ko"].remove(al); x["pct"] -= pct_of[al]
                # 잘못 붙은 답은 맞는 칸을 다시 찾고('따뜻한 봄'이 가을에 붙었으면 → 봄), 없을 때만 따로 칸
                heads = [y for k, y in merged.items() if not str(k).startswith("split-")]
                s = await same_slot(router, q["ko"], [y["ko"] for y in heads], al)
                if s >= 0 and heads[s] is not x:
                    heads[s]["pct"] += pct_of[al]; heads[s]["alias_ko"].append(al)
                else:
                    merged[f"split-{al}"] = {"ko": al, "en": al, "pct": pct_of[al], "alias_ko": [], "untranslated": True}
    for x in merged.values():   # 갈라진 칸은 같은 영어 이름의 기존 칸이 있으면 거기로
        if x.get("untranslated"):
            x["en"] = (await translate(router, q["ko"], [x["ko"]]))[0]
    items = []
    for x in sorted(merged.values(), key=lambda x: -x["pct"]):
        same = next((y for y in items if y["en"].lower() == x["en"].lower()), None)
        if same:
            same["pct"] += x["pct"]; same["alias_ko"] += [x["ko"], *x["alias_ko"]]
        else:
            items.append({k: v for k, v in x.items() if k != "untranslated"})
    items.sort(key=lambda x: -x["pct"])
    for src, dst in MERGE.get(q["id"], []):
        a = next((y for y in items if y["ko"] == src), None)
        b = next((y for y in items if dst in y["ko"] and y is not a), None)
        if a and b:
            b["pct"] += a["pct"]; b["alias_ko"] += [a["ko"], *a["alias_ko"]]; items.remove(a)
    for y in items:
        y["en"] = y["en"][:1].upper() + y["en"][1:]
    items.sort(key=lambda x: -x["pct"])
    tot = sum(x["pct"] for x in items)
    for x in items:
        x["pct"] = round(100 * x["pct"] / tot, 1)
    rec = {"id": q["id"], "question_ko": q["ko"], "question_en": c.question_en, "target_ko": q["target"],
           "prior": items, "board_size": min(BOARD, sum(x["pct"] >= 1 for x in items)), "dropped_junk": junk, "source": None, "reality": None}
    if q["real"]:
        rs, none = reality(q["real"])
        en = await translate(router, q["ko"], [k for k, _ in rs])
        rec["source"] = source(q["ko"], q["part"])
        keys = [k for k, v in q["real"].items() if k != NONE]   # 수치 미공개 항목도 배정 대상('2'가 '5'로 가지 않게)
        for x in items[:rec["board_size"]]:   # 정답판 칸 ↔ 갤럽 항목(공개 화면의 'Gallup said ○%')
            s = await same_slot(router, q["ko"], keys, x["ko"])
            x["reality_ko"] = keys[s] if s >= 0 and x["ko"] not in NOMAP.get(q["id"], []) else None   # 미공개 항목이면 화면에 'listed, % n/a'
        rec["reality_unpublished"] = [k for k, v in q["real"].items() if v is None and k != NONE]
        rec["reality"] = {"items": [{"ko": k, "en": e, "pct": round(p, 1)} for (k, _), e, p in zip(rs, en, [p for _, p in rs])],
                          "none_pct": none, "basis": "people who gave an answer (none/don't-know removed); items without published % omitted"}
    return rec


def board_overlap(rec):
    """AI 정답판 상위 칸 vs 실측(칸↔항목 judge 배정, 같은 항목에 배정된 칸은 합침)."""
    real = {x["ko"]: x["pct"] for x in rec["reality"]["items"]}
    ai = {}
    for x in rec["prior"][:rec["board_size"]]:
        if x.get("reality_ko"):
            ai[x["reality_ko"]] = ai.get(x["reality_ko"], 0) + x["pct"]
    return sum(min(v, real[k]) for k, v in ai.items() if k in real) / sum(real.values())


async def main():
    router = Router(budget_usd=1.0)
    evolve = json.loads(EVOLVE_CACHE.read_text())
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    fits = json.loads((ROOT / "data" / "feud_evolve.json").read_text())["fits"]
    kappa = float(np.median([k for _, k in fits]))
    recs = []
    for q in questions():
        rec = await build_one(router, q, cache, evolve)
        recs.append(rec)
        board = rec["prior"][:rec["board_size"]]
        bad = [x["ko"] for x in board if JUNK_RE.search(x["ko"])]
        ov = f" · 실측 겹침 {board_overlap(rec):.2f}" if rec["reality"] else ""
        print(f"[{rec['id']}] {rec['question_en']}{ov}\n   " + " | ".join(f"{x['en']} {x['pct']:.0f}" for x in board)
              + (f"\n   ⚠ 쓰레기 칸 {bad}" if bad else "") + (f"\n   뺀 칸: {rec['dropped_junk']}" if rec["dropped_junk"] else ""), flush=True)
    OUT.write_text(json.dumps({"kappa": kappa, "kappa_note": "AI prior counts as this many real respondents (empirical Bayes, Gallup bank)",
                               "questions": recs}, ensure_ascii=False, indent=1))
    print(f"\n{len(recs)}문항 (실측 {sum(1 for r in recs if r['reality'])}) · κ={kappa:g} · 비용 ${router.total_cost:.3f} → {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
