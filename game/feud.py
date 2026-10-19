"""AI 100인에게 물었습니다: 페르소나 100명 주관식 → 정답판 → 맞히기 → 공개 시 답한 사람들 프로필 + 교차표 + 실제 조사.

실행: .venv/bin/python -m game.feud "엄마한테 가장 많이 듣는 잔소리는?" [--guess "밥 먹어,공부해,…"] [--real]
"""
import argparse
import asyncio
import re
from collections import Counter

import pandas as pd
from pydantic import BaseModel, Field

from game.worldwide import load, sample
from llm.router import Router

BOARD = 8
AGE = [(18, 29, "20대"), (30, 39, "30대"), (40, 49, "40대"), (50, 59, "50대"), (60, 200, "60대+")]
AREA = {"서울": "수도권", "경기": "수도권", "인천": "수도권", "대전": "충청", "세종": "충청", "충청북": "충청", "충청남": "충청",
        "광주": "호남", "전북": "호남", "전라남": "호남", "부산": "영남", "대구": "영남", "울산": "영남", "경상북": "영남",
        "경상남": "영남", "강원": "강원·제주", "제주": "강원·제주"}


class Ans(BaseModel):
    answer: str = Field(description="한국어 단어나 짧은 구, 15자 이내")


class Labels(BaseModel):
    labels: list[str] = Field(description="정답판 칸 이름, 많은 순, 최대 8개")


class Match(BaseModel):
    slot: int = Field(description="추측이 가리키는 칸 번호, 없으면 -1")


class RealPoll(BaseModel):
    found: bool
    source: str = ""
    url: str = ""
    year: str = ""
    sample: str = ""
    question: str = Field("", description="원 조사 문항(본문 그대로)")
    target: str = Field("", description="조사 대상(예: 온라인 패널 회원, 전국 만 18세 이상, 직장인)")
    age_min: int | None = Field(None, description="본문에 적힌 대상 최소 나이, 없으면 null")
    age_max: int | None = Field(None, description="본문에 적힌 대상 최대 나이, 없으면 null")
    sex: str | None = Field(None, description="대상이 한 성별뿐이면 '남자' 또는 '여자', 아니면 null")
    method: str = Field("", description="조사 방법(온라인·전화·면접 등)")
    closed: bool = Field(False, description="보기 중 고르는 객관식이면 true")
    items: list[dict] = Field(default_factory=list, description='[{"answer": 보기/답, "pct": 숫자(%)}] 본문에 적힌 것 전부, "없다" 같은 보기 포함')


def age_band(a):
    return next(l for lo, hi, l in AGE if lo <= a <= hi)


REALISM = ("면접원이 길에서 갑자기 물어본 상황입니다. 정답이나 '대표적인' 답, 남들이 좋게 볼 답을 찾지 말고 이 사람이 실제로 떠올릴 답을 하세요. "
           "딱히 없거나 관심이 없으면 '없다', 모르면 '모르겠다'라고 답해도 됩니다.")


def who(p, style="A"):
    """A: 기본 / B: 설문 현실감(없음·모름 허용, 대표 답 금지) / C: B + 취미 서술 제외(합성 취미가 산책·텃밭 쪽으로 끌어서)."""
    base = f"당신은 {p['age']}세 {p['sex']}, {p['region']} {p['place']} 거주, 직업 {p['occupation']}, {p['marital_status']}.\n"
    if style == "A":
        return base + f"소개: {p['persona'][:400]}\n취미: {p['hobbies']}\n길거리 설문에 응한 이 사람으로서, 처음 떠오르는 답 하나만 짧게 말하세요."
    hob = "" if style == "C" else f"취미: {p['hobbies']}\n"
    return base + f"소개: {p['persona'][:400]}\n{hob}{REALISM}\n답 하나만 짧게 말하세요."


async def poll(router, people, q, style="A"):
    jobs = [dict(role="respondent", thinking=True, temperature=1.0, max_tokens=2500, retries=2, schema=Ans, tag="feud",
                 messages=[{"role": "system", "content": who(p, style)}, {"role": "user", "content": q}]) for p in people]
    res = await router.gather(jobs, concurrency=25)
    return [{**p, "raw": a.answer.strip()} for p, (a, _) in zip(people, res) if a]


class Group(BaseModel):
    label: str = Field(description="정답판 칸 이름, 짧고 구체적")
    members: list[int] = Field(description="이 칸에 속하는 응답 번호")


class Groups(BaseModel):
    groups: list[Group]


def mostly_covers(n):
    def check(g: Groups):
        seen = {m for x in g.groups for m in x.members}
        if seen - set(range(n)):
            raise ValueError(f"unknown ids {sorted(seen - set(range(n)))[:5]}; ids are 0..{n - 1}")
        if len(seen) < 0.9 * n:
            raise ValueError(f"missing ids {sorted(set(range(n)) - seen)[:10]}: put every id in exactly one group")
    return check


async def cluster(router, q, rows):
    """Super가 서로 다른 응답(번호만, 횟수 없이)을 묶음 → 빠진 번호는 하나씩 판정해 기존 칸 또는 새 칸.
    (횟수를 보여 주면 Super가 문자열을 반복 복사하고, 칸을 먼저 정하면 남는 답이 엉뚱한 칸에 억지로 들어감)"""
    raw = [r["raw"] for r in rows]
    uniq = sorted(set(raw))
    listing = "\n".join(f"{i}. {a}" for i, a in enumerate(uniq))
    g, _ = await router.acall("extract", [{"role": "user", "content":
        f"설문 질문: {q}\n서로 다른 응답 {len(uniq)}개:\n{listing}\n\n퀴즈쇼 진행자가 '같은 답'으로 인정할 응답끼리 묶어 정답판 칸을 만들어라. "
        "같은 요구·같은 행동이면 표현이 달라도 한 칸(예: '밥 먹었니?'·'밥 잘 챙겨'·'밥 먹어' → 밥 먹어 / '방 청소해'·'청소해' → 청소). "
        "주제가 다르면 다른 칸(예: '그만해'와 '청소해', '공부해'와 '청소해'는 다름). 혼자뿐인 답은 혼자 한 칸. "
        f"0~{len(uniq) - 1} 모든 번호를 정확히 한 번씩. 칸 이름은 짧고 구체적으로."}],
        schema=Groups, validator=mostly_covers(len(uniq)), thinking=False, max_tokens=4000, temperature=0, retries=3, tag="feud-groups")
    labels, slot_of = [], {}
    for x in (g.groups if g else []):
        labels.append(x.label)
        for m in x.members:
            slot_of.setdefault(uniq[m], len(labels) - 1)
    for u in [u for u in uniq if u not in slot_of]:   # 빠진 답: 기존 칸과 같은 뜻이면 그 칸, 아니면 새 칸
        s = await judge(router, q, labels, u)
        if s < 0:
            labels.append(u); s = len(labels) - 1
        slot_of[u] = s
    for r in rows:
        r["slot"] = slot_of[r["raw"]]
    cnt = Counter(r["slot"] for r in rows)
    order = [s for s, _ in cnt.most_common(BOARD)]
    remap = {s: k for k, s in enumerate(order)}
    for r in rows:
        r["slot"] = remap.get(r["slot"], -1)
    return [labels[s] for s in order], rows


async def judge(router, q, labels, guess, category=False):
    menu = "\n".join(f"{k}: {l}" for k, l in enumerate(labels))
    # Nano는 단어 겹침에 끌려 오판('그만해'→'밥 먹어') → Super 추론 끔
    if category:   # 실제 조사 범주(예: '학업, 성적에 대한 잔소리') ↔ 정답판 칸: 주제가 같을 때만
        menu = "\n".join(f"{k}: {l}" for k, l in enumerate(labels))
        m, _ = await router.acall("extract", [{"role": "user", "content":
            f"질문: {q}\n정답판:\n{menu}\n실제 조사 범주: \"{guess}\"\n이 범주의 대표적인 답이 곧 그 칸인 경우만 칸 번호(예: '학업 잔소리'→'공부해'). "
            "범주가 칸과 다른 주제면(예: '친구 관계'·'인성'·'잠'은 '공부해'가 아님) -1."}],
            schema=Match, thinking=False, max_tokens=100, temperature=0, retries=2, tag="feud-judge-cat")
        return m.slot if m and 0 <= m.slot < len(labels) else -1
    m, _ = await router.acall("extract", [{"role": "user", "content":
        f"퀴즈 질문: {q}\n정답판:\n{menu}\n플레이어 추측: \"{guess}\"\n추측과 뜻이 같은 칸 번호(동의어·오타·띄어쓰기·어미 차이는 같은 뜻). 단어 하나가 겹쳐도 말하려는 내용이 다르면 -1(예: '밥 너무 많이 먹어'는 '밥 먹었어?'와 다름)."}],
        schema=Match, thinking=False, max_tokens=100, temperature=0, retries=2, tag="feud-judge")
    return m.slot if m and 0 <= m.slot < len(labels) else -1


def frame(rows):
    d = pd.DataFrame(rows)
    d["연령대"] = d.age.map(age_band)
    d["권역"] = d.region.map(AREA).fillna("기타")
    d["성별"] = d.sex
    if "w" not in d:
        d["w"] = 1.0
    return d


def profile(d, slot):
    """칸을 연 순간 보여 줄 '이 답을 한 사람들' 요약: 구성, 전체 대비 많이 몰린 집단(배수), 대표 3명."""
    g = d[d.slot == slot]
    wshare = lambda col, x: x.groupby(col).w.sum().sort_values(ascending=False) / x.w.sum()   # noqa: E731 가중 비율
    lines = [f"응답자 {len(g)}명 · 평균 {(g.age * g.w).sum() / g.w.sum():.0f}세 · " + " ".join(f"{k} {v:.0%}" for k, v in wshare("성별", g).items())]
    for col in ["연령대", "권역"]:
        lines.append(f"{col}: " + " · ".join(f"{k} {v:.0%}" for k, v in wshare(col, g).items()))
    lift = []
    for col in ["연령대", "성별", "권역", "marital_status"]:
        base, here, cnt = wshare(col, d), wshare(col, g), g[col].value_counts()
        for k, v in here.items():
            if cnt[k] >= 3 and v / base[k] >= 1.5:
                lift.append(f"{k} {v / base[k]:.1f}배")
    if lift:
        lines.append("전체보다 많음: " + ", ".join(lift))
    jobs = g.occupation.value_counts().head(3)
    lines.append("직업: " + ", ".join(f"{k}({v})" if v > 1 else k for k, v in jobs.items()))
    for _, r in g.sample(min(3, len(g)), random_state=0).iterrows():
        lines.append(f"  · {r.age}세 {r.sex} · {r.region} {r.place} · {r.occupation} — \"{r.raw}\"")
    return "\n".join(lines)


def crosstab(d, labels, k=5):
    """여론조사 결과표 형식: 행 = 전체/성별/연령대/권역, 열 = 상위 k개 답 비율(%) + 사례수."""
    top = list(range(min(k, len(labels))))
    out = []
    for col in [None, "성별", "연령대", "권역"]:
        groups = [("전체", d)] if col is None else [(f"{col}:{v}", g) for v, g in d.groupby(col)]
        for name, g in groups:
            row = {"구분": name, "n": len(g)}
            row.update({labels[s]: round(100 * g.w[g.slot == s].sum() / g.w.sum()) for s in top})
            out.append(row)
    return pd.DataFrame(out).set_index("구분")


async def real_poll(router, q):
    """Tavily로 실제 공개 조사 찾기 → 본문에 적힌 수치만 추출 → 수치가 원문에 있는지 대조."""
    from evidence.agent import fetch
    results = fetch(f"{q} 설문조사 결과 순위 %", k=5, extract_top=3, max_chars=6000)
    msgs = [{"role": "user", "content": f"질문: {q}\n아래 검색 결과 중 이 질문과 같은 주제의 실제 설문조사(표본·기관이 있는 것) 하나의 응답 순위를 옮겨라. "
             "본문에 적힌 숫자만, 없으면 found=false. 대상·나이·방법은 본문에 적힌 것만(추측 금지).\n\n" + "\n\n".join(f"[{i}] URL: {x['url']}\n제목: {x['title']}\n본문: {x['content']}" for i, x in enumerate(results))}]
    rp, _ = await router.acall("extract", msgs, schema=RealPoll, thinking=False, max_tokens=3000, retries=2, tag="feud-real")
    if not rp or not rp.found:
        return None, results
    text = " ".join(x["content"] for x in results if x["url"] == rp.url) or " ".join(x["content"] for x in results)
    rp.items = [it for it in rp.items if re.search(rf"(?<![\d.]){re.escape(str(it.get('pct', '')).rstrip('0').rstrip('.'))}", text)]
    return (rp if rp.items else None), results


def target_sample(con, rp, n, seed):
    """실제 조사 대상과 같은 사람들(나이·성별 조건이 본문에 있을 때만 적용). 페르소나는 19세 이상뿐."""
    cond, args = ["country = 'Korea'"], []
    if rp.age_min is not None:
        cond.append("age >= ?"); args.append(rp.age_min)
    if rp.age_max is not None:
        cond.append("age <= ?"); args.append(rp.age_max)
    if rp.sex in ("남자", "여자"):
        cond.append("sex = ?"); args.append(rp.sex)
    return con.execute(f"SELECT * FROM people WHERE {' AND '.join(cond)} ORDER BY hash(uuid || ?) LIMIT ?", args + [str(seed), n]).df().to_dict("records")


async def replicate(router, people, rp):
    """같은 문항·같은 보기·같은 대상으로 AI에게 다시 묻기(사과 대 사과 비교)."""
    from typing import Literal
    from pydantic import create_model
    opts = [str(it["answer"]) for it in rp.items]
    Pick = create_model("Pick", choice=(Literal[tuple(opts)], ...))
    menu = "\n".join(f"- {o}" for o in opts)
    jobs = [dict(role="respondent", thinking=True, temperature=1.0, max_tokens=2500, retries=2, schema=Pick, tag="feud-replicate",
                 messages=[{"role": "system", "content": who(p).replace("처음 떠오르는 답 하나만 짧게 말하세요.", "보기 중 하나를 고르세요.")},
                           {"role": "user", "content": f"{rp.question}\n보기:\n{menu}"}]) for p in people]
    res = await router.gather(jobs, concurrency=25)
    picks = Counter(a.choice for a, _ in res if a)
    tot = sum(picks.values())
    return {o: round(100 * picks.get(o, 0) / tot) for o in opts}, tot


class Share(BaseModel):
    answer: str
    pct: float


class Dist(BaseModel):
    shares: list[Share] = Field(description="응답 상위 15개 정도와 '없음/모름', 합 100")


TARGET = "한국 전국 만 13세 이상 1,700명에게 면접원이"
NONE_RE = re.compile(r"없|모르|모름|관심")


async def estimate(router, q, target=TARGET):
    """정답판 비율: Super가 전국 자유응답 분포를 직접 추정(검증 1~4에서 페르소나 표본보다 실측에 가까움). '없음/모름'은 빼고 100으로 맞춤."""
    d, _ = await router.acall("extract", [{"role": "user", "content":
        f"{target} 보기 없이 물었다: \"{q}\"\n자유응답 결과 분포를 추정하라. "
        "고정관념이나 대표 답이 아니라 실제 사람들이 답했을 법한 분포로, 응답 상위 15개 정도와 '없음/모름' 비율을 % 로(합 100)."}],
        schema=Dist, thinking=True, max_tokens=6000, temperature=0.3, retries=2, tag="feud-direct")
    xs = [(x.answer, x.pct) for x in (d.shares if d else []) if not NONE_RE.search(x.answer) and x.pct > 0]
    tot = sum(p for _, p in xs) or 1
    return sorted([(a, 100 * p / tot) for a, p in xs], key=lambda t: -t[1])


async def hybrid(router, people, q, target=TARGET):
    """정답판 = 직접 추정 상위 BOARD칸, '이 답을 한 사람들' = 페르소나 응답을 칸에 배정하고 칸별 가중치를 정답판 비율에 맞춤(사후 가중).
    페르소나가 '없다'고 한 사람은 정답판 밖(퀴즈쇼 정답판은 답한 사람 기준)."""
    shares, rows = await asyncio.gather(estimate(router, q, target), poll(router, people, q))
    labels, pct = [a for a, _ in shares[:BOARD]], [p for _, p in shares[:BOARD]]
    rows = [r for r in rows if not NONE_RE.search(r["raw"])]
    uniq = sorted({r["raw"] for r in rows})
    slot = dict(zip(uniq, await asyncio.gather(*[judge(router, q, labels, u) for u in uniq])))
    for r in rows:
        r["slot"] = slot[r["raw"]]
    cnt = Counter(r["slot"] for r in rows)
    goal = {**dict(enumerate(pct)), -1: max(100 - sum(pct), 0)}
    for r in rows:
        r["w"] = goal[r["slot"]] / (100 * cnt[r["slot"]] / len(rows))
    return labels, pct, rows


async def main(q, guesses, n, seed, real):
    router = Router(budget_usd=0.5)
    con = load()
    people = sample(con, "Korea", n, seed)
    rows = await poll(router, people, q)
    labels, rows = await cluster(router, q, rows)
    d = frame(rows)
    counts = d.slot.value_counts()
    print(f"\n한국인 {len(d)}명에게 물었습니다: {q}\n정답판 {len(labels)}칸 (칸 밖 {int((d.slot < 0).sum())}명)")
    opened, strikes = set(), 0
    for gu in guesses:
        s = await judge(router, q, labels, gu)
        if s < 0 or s in opened:
            strikes += 1
            print(f"\n> {gu}  ✗ ({strikes}/3)")
            if strikes == 3:
                break
            continue
        opened.add(s)
        print(f"\n> {gu}  ✓ {s + 1}위 「{labels[s]}」 {counts[s]}명\n{profile(d, s)}")
    print("\n=== 정답판 전체 ===")
    for s, l in enumerate(labels):
        print(f"{s + 1}. {l:<14} {counts.get(s, 0):3d}명 {'(맞힘)' if s in opened else ''}")
    print("\n=== 여론조사 결과표(%) — AI 페르소나, 칸 작으면 해석 주의 ===")
    print(crosstab(d, labels).to_string())
    if real:
        rp, _ = await real_poll(router, q)
        print("\n=== 실제 조사 ===")
        if rp:
            age = f"{rp.age_min or '?'}~{rp.age_max or '?'}세" if (rp.age_min or rp.age_max) else "연령 미기재"
            print(f"{rp.source} {rp.year} · {rp.target} ({age}, {rp.method}) · 표본 {rp.sample}\n{rp.url}\n문항: {rp.question} [{'객관식' if rp.closed else '형식 미기재 → 범주를 보기로 재현'}]")
            tp = target_sample(con, rp, n, seed)
            same, m = await replicate(router, tp, rp)   # 형식 표기가 없어도 결과가 범주 목록이면 그 범주를 보기로 재현
            if True:
                ages = pd.Series([p["age"] for p in tp])
                print(f"AI 재조사: 같은 문항·같은 보기, 대상 조건 맞춘 페르소나 {m}명(나이 {ages.min()}~{ages.max()}, 중앙 {ages.median():.0f}세)")
            share = d.slot.value_counts(normalize=True)
            print(f"  {'보기':<34} 실제   AI(같은 형식)   AI(주관식 정답판)")
            for it in rp.items:
                s = await judge(router, q, labels, str(it.get("answer")), category=True)
                board = f"{round(100 * share.get(s, 0))}% 「{labels[s]}」" if s >= 0 else "없음"
                print(f"  {str(it.get('answer'))[:32]:<34} {it.get('pct'):>3}%   {str(same.get(str(it['answer']), '-')) + '%':>8}        {board}")
        else:
            print("찾지 못함(원문 수치 대조를 통과한 조사 없음)")
    print(f"\n비용 ${router.total_cost:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--guess", default="")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--real", action="store_true")
    a = ap.parse_args()
    asyncio.run(main(a.question, [g.strip() for g in a.guess.split(",") if g.strip()], a.n, a.seed, a.real))
