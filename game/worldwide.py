"""세계판 게임 엔진: Nemotron-Personas 10개국 → 공통 표 → 페르소나가 영어로 대답(이름·지명·국적 금지).

데이터: data/personas_world/<Country>/*.parquet (각국 첫 조각, CC BY 4.0). 성인(18+)만.
누설 방지 2단: 규칙(ASCII·금지어) → Nano 분류기(지명·국적·이름·외국어 단어) → 걸리면 재생성.
"""
import asyncio
import glob
import re
from typing import Literal

import duckdb
from pydantic import BaseModel, Field

from world.config import ROOT

DATA = ROOT / "data" / "personas_world"
# 나라 → (세부 지역 열, 상위 지역 열)
PLACE = {"Belgium": ("municipality", "region"), "Brazil": ("municipality", "state"), "El-Salvador": ("municipality", "department"),
         "France": ("commune", "departement"), "India": ("district", "state"), "Japan": ("area", "prefecture"),
         "Korea": ("district", "province"), "Singapore": ("planning_area", "planning_area"), "USA": ("city", "state"),
         "Vietnam": ("zone", "region")}
COUNTRIES = list(PLACE)
NAME = {"El-Salvador": "El Salvador", "USA": "United States"}
BANNED = {"korea", "korean", "japan", "japanese", "america", "american", "usa", "u.s.", "united states", "india", "indian",
          "brazil", "brazilian", "salvador", "salvadoran", "singapore", "singaporean", "france", "french", "vietnam",
          "vietnamese", "belgium", "belgian", "flemish", "walloon", "dutch", "portuguese", "spanish", "hindi", "asia", "asian",
          "europe", "european", "latin", "seoul", "busan", "tokyo", "osaka", "kyoto", "hokkaido", "paris", "lyon", "brussels",
          "antwerp", "delhi", "mumbai", "bangalore", "kolkata", "chennai", "sao paulo", "rio", "hanoi", "saigon", "ho chi minh",
          "new york", "california", "texas", "chicago"}


def load():
    con = duckdb.connect()
    parts = []
    for c, (fine, coarse) in PLACE.items():
        f = glob.glob(str(DATA / c / "*.parquet"))[0]
        parts.append(f"SELECT '{c}' AS country, uuid, age, sex, marital_status, education_level, occupation, "
                     f"CAST({fine} AS VARCHAR) AS place, CAST({coarse} AS VARCHAR) AS region, persona, hobbies_and_interests AS hobbies, "
                     f"cultural_background AS culture FROM '{f}' WHERE age >= 18")
    con.execute("CREATE TABLE people AS " + " UNION ALL ".join(parts))
    return con


def sample(con, country, n, seed):
    return con.execute("SELECT * FROM people WHERE country = ? ORDER BY hash(uuid || ?) LIMIT ?", [country, str(seed), n]).df().to_dict("records")


def first_name_tokens(p):
    """서술 첫머리의 이름(예: '野本 花代子は' / 'Bhai Shekh, a' / 'Marcos Antunes é')."""
    head = re.split(r"[,은는は이가]| is | é | est | là |, ", p["persona"], maxsplit=1)[0]
    return {w.lower() for w in re.findall(r"[A-Za-zÀ-ỹ]{3,}", head)}


class Reply(BaseModel):
    answer: str = Field(description="English, 1-2 sentences, max 40 words")


class Leak(BaseModel):
    flags: list[bool] = Field(description="answer별 누설 여부(true=누설), 입력 순서대로")


def persona_system(p, disguise=None):
    who = (f"You are a {p['age']}-year-old {p['sex']} ({p['marital_status']}, education: {p['education_level']}, "
           f"occupation: {p['occupation']}), living in {p['place']}, {p['region']}, {NAME.get(p['country'], p['country'])}.\n"
           f"About you: {p['persona']}\nHobbies: {p['hobbies']}\nBackground: {p['culture']}")
    rule = ("Answer as yourself, honestly, from your real daily life." if disguise is None else
            f"SECRET: you are playing a hidden-identity game. Pretend you are from {NAME.get(disguise, disguise)}, like the others. "
            "Hide anything that would reveal your real country, but keep your personality, age and life situation.")
    return (f"{who}\n{rule}\nRules: reply in plain English, 1-2 sentences (max 40 words), in your own voice. "
            "Never say any name, place, city, region, country, nationality, ethnicity or language name. "
            "Never use non-English words; describe things in plain English instead.")


def rule_check(p):
    banned = BANNED | first_name_tokens(p) | {w.lower() for w in re.findall(r"[A-Za-zÀ-ỹ]{4,}", f"{p['place']} {p['region']}")}

    def check(r: Reply):
        a = r.answer.strip()
        if not a or len(a.split()) > 50:
            raise ValueError("answer: 1~40 words")
        if re.search(r"[^\x00-\x7F’‘“”—–…]", a):
            raise ValueError("plain English only (ASCII), no foreign script or accents")
        low = a.lower()
        hit = [w for w in banned if re.search(rf"\b{re.escape(w)}\b", low)]
        if hit:
            raise ValueError(f"banned words {hit}: no names/places/nationalities")
    return check


def answer_jobs(people, question, disguise=None, thinking=False):
    """disguise: {인덱스: 위장 국가}."""
    disguise = disguise or {}
    return [dict(role="respondent", thinking=thinking, temperature=0.9, max_tokens=2000 if thinking else 300, retries=3,
                 schema=Reply, validator=rule_check(p), tag="ww-answer",
                 messages=[{"role": "system", "content": persona_system(p, disguise.get(i))},
                           {"role": "user", "content": f"Question: {question}"}])
            for i, p in enumerate(people)]


def same_len(n):
    def check(x: Leak):
        if len(x.flags) != n:
            raise ValueError(f"flags must have exactly {n} items")
    return check


async def ask(router, people, question, disguise=None, thinking=False, leak_rounds=2):
    """모든 사람에게 질문. 규칙 통과 후 Nano 분류기로 누설 재검사, 걸린 것만 재생성."""
    jobs = answer_jobs(people, question, disguise, thinking)
    res = await router.gather(jobs)
    answers = [r.answer.strip() if r else None for r, _ in res]
    stats = {"rule_fail": sum(a is None for a in answers), "leak_fixed": 0, "leak_left": 0}
    for rnd in range(leak_rounds + 1):
        idx = [i for i, a in enumerate(answers) if a]
        listing = "\n".join(f"{k}. {answers[i]}" for k, i in enumerate(idx))
        lk, _ = await router.acall("classify", [{"role": "user", "content":
            "For each answer, flag true if it mentions ANY specific place (city, region, country, landmark), nationality, ethnicity, "
            "language name, personal name, or a non-English word (e.g. a native dish or festival name). Generic descriptions are fine.\n"
            f"{listing}\nReturn flags in order, exactly {len(idx)} booleans."}], schema=Leak, max_tokens=400, temperature=0,
            retries=2, validator=same_len(len(idx)), tag="ww-leak")
        bad = [i for i, f in zip(idx, lk.flags) if f] if lk else []
        if not bad:
            break
        if rnd == leak_rounds:
            stats["leak_left"] = len(bad)
            for i in bad:
                answers[i] = None
            break
        redo = await router.gather([jobs[i] for i in bad])
        for i, (r, _) in zip(bad, redo):
            answers[i] = r.answer.strip() if r else None
        stats["leak_fixed"] += len(bad)
    return answers, stats


# ---------- 자동 플레이어(Super) ----------

class Guess(BaseModel):
    top3: list[Literal[tuple(NAME.get(c, c) for c in COUNTRIES)]] = Field(description="가능성 높은 순서 3개")
    reason: str = ""


class Pick(BaseModel):
    outsider: int = Field(description="외부인 번호(1부터)")
    reason: str = ""


def to_key(name):
    return {NAME.get(c, c): c for c in COUNTRIES}[name]
