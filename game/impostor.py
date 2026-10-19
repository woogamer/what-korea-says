"""한국인 아닌 사람 찾기 PoC: 한국 페르소나 4명 + 다른 나라 페르소나 1명(한국인인 척). 질문마다 5명이 모두 대답.

공정성: 모두 지명 금지(누설 검사). 외국인은 추가로 자국 음식·제도·문자 금지. 대답은 한글만 60자 이하.
실행: .venv/bin/python -m game.impostor [--seed N] [--script 파일] [--reveal 파일]
  --script: 한 줄에 질문 하나, 마지막 줄 "지목 N"(선택). --reveal: 정답·프로필을 따로 저장(자가 테스트용).
"""
import argparse
import asyncio
import random
import re

import duckdb
from pydantic import BaseModel, Field

from llm.router import Router
from world.config import L0_GLOB, ROOT

JP_GLOB = str(ROOT / "data" / "personas_jp" / "*.parquet")
N_PLAYERS = 5
MAX_Q = 5


class Reply(BaseModel):
    line: str = Field(description="한국어 대답 1~2문장, 60자 이하")


def place_words(*names):
    """'서울-서초구' / '東京都' 같은 지명에서 금지 단어(어간 포함)."""
    out = set()
    for n in names:
        for w in re.split(r"[-\s]+", n or ""):
            out.add(w)
            if len(w) > 2 and w[-1] in "시군구都道府県":
                out.add(w[:-1])
    return {w for w in out if len(w) >= 2}


def load_people(seed):
    con = duckdb.connect()
    kr = con.execute(f"SELECT age, sex, province, district, occupation, marital_status, education_level, persona, hobbies_and_interests "
                     f"FROM '{L0_GLOB}' ORDER BY hash(uuid || ?) LIMIT {N_PLAYERS - 1}", [str(seed)]).df().to_dict("records")
    jp = con.execute(f"SELECT age, sex, prefecture, region, occupation, marital_status, education_level, persona, hobbies_and_interests "
                     f"FROM '{JP_GLOB}' ORDER BY hash(uuid || ?) LIMIT 1", [str(seed)]).df().to_dict("records")[0]
    people = [{"nation": "한국", **p} for p in kr] + [{"nation": "일본", **jp}]
    random.Random(seed).shuffle(people)
    return people


def system_prompt(p):
    if p["nation"] == "한국":
        who = (f"당신은 {p['age']}세 {p['sex']}, 직업 {p['occupation']}, {p['marital_status']}, {p['education_level']}.\n"
               f"{p['persona']}\n관심사: {p['hobbies_and_interests']}")
        rule = "평소 당신 그대로 솔직하게 대답하세요."
    else:
        who = (f"あなたは{p['age']}歳 {p['sex']}, 職業 {p['occupation']}, {p['marital_status']}, {p['education_level']}.\n"
               f"{p['persona']}\n趣味: {p['hobbies_and_interests']}")
        rule = ("당신은 일본인이지만 이 게임에서는 한국인인 척해야 합니다. 들키면 집니다. "
                "일본 음식·지명·제도·명절·말버릇은 언급하지 말고, 그 밖의 생활·성격·생각은 당신 그대로 답하세요.")
    return (f"당신은 '한국인 아닌 사람 찾기' 게임의 참가자입니다.\n{who}\n{rule}\n"
            "어느 지역에 사는지, 어떤 지명도 말하지 마세요. 한국어로, 당신다운 말투로 1~2문장(60자 이하).")


def ask_all(router, people, question):
    def job(p):
        banned = place_words(p.get("province"), p.get("district"), p.get("prefecture"), p.get("region")) | {"일본", "한국인", "한국 사람"}

        def check(r: Reply):
            if not r.line.strip() or len(r.line) > 70:
                raise ValueError("line은 1~70자")
            if re.search(r"[A-Za-z぀-ヿ一-鿿]", r.line):
                raise ValueError("한글만(영어·가나·한자 금지)")
            hit = [w for w in banned if w in r.line]
            if hit:
                raise ValueError(f"금지 단어 {hit}")
        return dict(role="respondent", thinking=False, temperature=0.8, max_tokens=300, retries=3, schema=Reply, validator=check,
                    tag="impostor", messages=[{"role": "system", "content": system_prompt(p)},
                                              {"role": "user", "content": f"질문: {question}"}])
    res = asyncio.run(router.gather([job(p) for p in people]))
    return [r.line if r else "(대답 없음)" for r, _ in res]


def profile(p):
    place = p.get("district") or f"{p.get('prefecture')}"
    return f"{p['nation']} · {p['age']}세 {p['sex']} · {place} · {p['occupation']} · {p['marital_status']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int)
    ap.add_argument("--script")
    ap.add_argument("--reveal")
    args = ap.parse_args()
    seed = args.seed if args.seed is not None else random.randrange(1 << 30)
    people = load_people(seed)
    router = Router()
    lines = [l.strip() for l in open(args.script, encoding="utf-8") if l.strip()] if args.script else None

    print(f"[seed {seed}] {N_PLAYERS}명 중 1명은 한국인이 아닙니다. 질문 {MAX_Q}번 후 지목하세요.\n")
    pick = None
    for qn in range(1, MAX_Q + 1):
        q = (lines.pop(0) if lines else "") if lines is not None else input(f"Q{qn}. ").strip()
        if not q or q.startswith("지목"):
            pick = q; break
        if lines is not None:
            print(f"Q{qn}. {q}")
        for i, a in enumerate(ask_all(router, people, q), 1):
            print(f"   {i}번: {a}")
        print()
    if lines is not None:
        pick = pick or (lines.pop(0) if lines else None)
    else:
        pick = input("누가 외국인일까요? 번호: ")
    m = re.search(r"\d", pick or "")
    answer = next(i for i, p in enumerate(people, 1) if p["nation"] != "한국")
    reveal = "\n".join(f"{i}번: {profile(p)}" for i, p in enumerate(people, 1))
    result = (f"지목 {m.group() if m else '-'}번 → {'정답!' if m and int(m.group()) == answer else f'틀림. 외국인은 {answer}번'}\n{reveal}\n"
              f"비용 ${router.total_cost:.4f} · 호출 {len(router.calls)}회")
    if args.reveal:
        open(args.reveal, "w", encoding="utf-8").write(result + "\n")
        print(f"(정답은 {args.reveal}에 저장)")
    else:
        print(result)


if __name__ == "__main__":
    main()
