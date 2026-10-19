"""가치관 검사 10문항(잉글하트-웰젤 문화 지도 문항, WVS 7차 문구 기반)과 AI 시민 응답 수집.

축: 전통 ↔ 세속·합리 (god, autonomy, abortion, pride, authority)
    생존 ↔ 자기표현 (postmat, happy, homosexuality, petition, trust)
점수 계산(요인 적재)은 WVS 원자료를 받은 뒤 같은 방식으로 실측과 AI에 함께 적용한다.
실행: .venv/bin/python -m game.values [--per 30]  → data/values_ai.parquet
"""
import argparse
import asyncio
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from game.worldwide import load, persona_system, sample
from llm.router import Router
from world.config import ROOT

# WVS 7차에 있는 나라(+ 프랑스는 EVS/WVS 통합 자료). 벨기에·엘살바도르는 실측 없음
COUNTRIES = ["Korea", "Japan", "USA", "India", "Brazil", "Singapore", "Vietnam", "France"]
QUALITIES = ["good manners", "independence", "hard work", "feeling of responsibility", "imagination",
             "tolerance and respect for other people", "thrift, saving money and things", "determination, perseverance",
             "religious faith", "unselfishness", "obedience"]
GOALS = ["maintaining order in the nation", "giving people more say in important government decisions",
         "fighting rising prices", "protecting freedom of speech"]

QUESTIONNAIRE = f"""Please answer this survey as yourself.
1. god: How important is God in your life? 1 = not at all important, 10 = very important.
2. child_qualities: Here is a list of qualities that children can be encouraged to learn at home. Which do you consider especially important? Choose up to five: {QUALITIES}
3. abortion: Is abortion justifiable? 1 = never justifiable, 10 = always justifiable.
4. pride: How proud are you of your nationality? (very proud / quite proud / not very proud / not at all proud)
5. authority: If greater respect for authority took place in the near future, would it be a good thing, a bad thing, or don't you mind?
6. goal_first, goal_second: Which is most important for your country, and which second? {GOALS}
7. happy: Taking all things together, would you say you are very happy, rather happy, not very happy, or not at all happy?
8. homosexuality: Is homosexuality justifiable? 1 = never justifiable, 10 = always justifiable.
9. petition: Signing a petition — have you done it, might you do it, or would you never do it?
10. trust: Generally speaking, would you say that most people can be trusted, or that you need to be very careful in dealing with people?"""


class Answers(BaseModel):
    god: int = Field(ge=1, le=10)
    child_qualities: list[Literal[tuple(QUALITIES)]] = Field(max_length=5)
    abortion: int = Field(ge=1, le=10)
    pride: Literal["very proud", "quite proud", "not very proud", "not at all proud"]
    authority: Literal["good thing", "don't mind", "bad thing"]
    goal_first: Literal[tuple(GOALS)]
    goal_second: Literal[tuple(GOALS)]
    happy: Literal["very happy", "rather happy", "not very happy", "not at all happy"]
    homosexuality: int = Field(ge=1, le=10)
    petition: Literal["have done", "might do", "would never do"]
    trust: Literal["most people can be trusted", "need to be very careful"]


def no_dup_goal(a: Answers):
    if a.goal_first == a.goal_second:
        raise ValueError("goal_second must differ from goal_first")


def ai_system(p):
    # 10/8 실험: 추론 끄면 종교 문항이 0~13%로 무너짐 → 추론 켬 고정. 이름·지명 금지 규칙은 설문엔 불필요
    return persona_system(p).split("\nRules:")[0] + "\nAnswer as this person would, honestly, based on their own life and beliefs."


async def collect(per, seed):
    con = load(); router = Router(budget_usd=2.0)
    people = [p for c in COUNTRIES for p in sample(con, c, per, seed)]
    jobs = [dict(role="respondent", thinking=True, temperature=0.9, max_tokens=4000, retries=2, schema=Answers,
                 validator=no_dup_goal, tag="values-ai",
                 messages=[{"role": "system", "content": ai_system(p)}, {"role": "user", "content": QUESTIONNAIRE}]) for p in people]
    res = await router.gather(jobs, concurrency=48)
    rows = [{"country": p["country"], "uuid": p["uuid"], "age": p["age"], "sex": p["sex"], **a.model_dump()}
            for p, (a, _) in zip(people, res) if a]
    df = pd.DataFrame(rows)
    df.to_parquet(ROOT / "data" / "values_ai.parquet")
    print(f"{len(df)}/{len(people)} ok · ${router.total_cost:.3f}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=30)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    df = asyncio.run(collect(a.per, a.seed))
    df["autonomy"] = df.child_qualities.apply(lambda q: ("independence" in q) + ("determination, perseverance" in q)
                                              - ("obedience" in q) - ("religious faith" in q))
    print(df.groupby("country")[["god", "abortion", "homosexuality", "autonomy"]].mean().round(1))
    for col in ["pride", "happy", "trust", "petition", "authority"]:
        print(pd.crosstab(df.country, df[col], normalize="index").round(2))
