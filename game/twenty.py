"""스무고개 PoC: 100만 페르소나 중 숨은 한 명을 예/아니오 질문으로 좁힌다.

흐름: 질문 → Nemotron이 데이터 조건(JSON)으로 번역 → 열·값 검증 → 숨은 사람에게 적용해 예/아니오
→ (플레이어 예측) → 남은 인원 공개 → 페르소나가 그 판정대로 연기해 대답.
판정은 데이터가 하고 모델은 번역과 말투만 맡는다. 데이터 열에 없는 질문은 인원을 줄이지 않고 'AI 상상'으로 표시.

실행: .venv/bin/python -m game.twenty [--seed N] [--script 파일]
  --script: 한 줄에 "질문" 또는 "질문 | 예측인원". 자동 플레이(테스트용).
"""
import argparse
import math
import random
import re
from typing import Literal

import duckdb
from pydantic import BaseModel, Field

from llm.router import Router
from world.config import L0_GLOB

CAT_COLS = ["sex", "marital_status", "military_status", "family_type", "housing_type",
            "education_level", "bachelors_field", "province",
            "lives_alone", "with_spouse", "with_children", "with_parents", "employed"]
# 스무고개 질문은 "자녀랑 살아요?"처럼 예/아니오가 기본인데 원 열(family_type 39종)로는 번역이 흔들린다 → 파생 열
DERIVED = {
    "lives_alone": "family_type LIKE '혼자 거주%'",
    "with_spouse": "family_type LIKE '%배우자%' AND family_type NOT LIKE '%별거%'",
    "with_children": "regexp_matches(family_type, '(^|·)자녀')",
    "with_parents": "regexp_matches(family_type, '부모|어머니|아버지|부 또는 모')",
    "employed": "occupation <> '무직' AND occupation NOT LIKE '%구직중%'",
}
TEXT_COLS = ["occupation", "district"]          # 값이 수천 개 → 부분 문자열 일치
# 9/26 L0 정합성 카드에서 공식 통계와 어긋난 축(학력 0.13·주거 0.17) + 아직 대조 안 한 축
SYNTHETIC_ONLY = {"education_level", "housing_type", "occupation", "family_type", "bachelors_field",
                  "lives_alone", "with_spouse", "with_children", "with_parents", "employed"}
LABEL = {"sex": "성별", "age": "나이", "marital_status": "혼인", "military_status": "군 복무", "family_type": "가족 형태",
         "housing_type": "주거", "education_level": "학력", "bachelors_field": "전공", "province": "시도",
         "occupation": "직업", "district": "시군구", "lives_alone": "혼자 삶", "with_spouse": "배우자와 같이 삶",
         "with_children": "자녀와 같이 삶", "with_parents": "부모와 같이 삶", "employed": "일하는 중(구직중·무직 제외)"}
WIN_AT = 100
MAX_Q = 20


class Cond(BaseModel):
    column: Literal["sex", "age", "marital_status", "military_status", "family_type", "housing_type",
                    "education_level", "bachelors_field", "province", "occupation", "district",
                    "lives_alone", "with_spouse", "with_children", "with_parents", "employed"]
    op: Literal["in", "not_in", "between", "contains"]
    values: list[str] = Field(default_factory=list, description="in/not_in: 허용 값 목록, contains: 키워드 1개")
    lo: int | None = Field(None, description="between: 나이 하한(포함)")
    hi: int | None = Field(None, description="between: 나이 상한(포함)")


class Translation(BaseModel):
    kind: Literal["filter", "unmappable"]
    conds: list[Cond] = Field(default_factory=list, description="AND로 묶임. 질문이 '예'일 조건")
    note: str = Field("", description="번역 근거 한 줄")


def load(con):
    base = [c for c in CAT_COLS if c not in DERIVED]
    derived = [f"CASE WHEN {e} THEN '예' ELSE '아니오' END AS {c}" for c, e in DERIVED.items()]
    cols = ", ".join(["uuid", "age"] + base + derived + TEXT_COLS + ["persona", "family_persona", "hobbies_and_interests", "cultural_background"])
    con.execute(f"CREATE TABLE people AS SELECT {cols} FROM '{L0_GLOB}'")
    return {c: [r[0] for r in con.execute(f"SELECT {c} FROM people GROUP BY 1 ORDER BY count(*) DESC").fetchall()] for c in CAT_COLS}


def translator_prompt(domains, question):
    dom = "\n".join(f"- {c} ({LABEL[c]}): {', '.join(domains[c])}" for c in CAT_COLS)
    return f"""스무고개 질문을 인구 데이터 조건으로 바꾸세요. 대상은 한국 19~99세 성인 한 명입니다.

열과 허용 값(in/not_in 값은 반드시 아래 표기 그대로):
- age (나이): 정수 19~99. op=between, lo/hi 사용. "30대"=30~39, "40살 넘음"=41~99
{dom}
- occupation (직업): 자유 문자열. op=contains, values=[키워드 1개]. 예: "교사", "간호", "무직"
- district (시군구): "서울-서초구" 형식. op=contains, values=[키워드 1개]. 예: "강남구", "수원"

규칙:
- 질문이 '예'가 되는 조건을 conds에 넣으세요. 여러 조건은 AND.
- 같이 사는 사람·취업 여부는 예/아니오 열(lives_alone, with_spouse, with_children, with_parents, employed)을 우선 쓰세요.
  데이터는 '같이 사는' 가족만 압니다. "자녀 있어요?"는 with_children(같이 사는 자녀)로 근사하세요.
- "결혼했어요?"처럼 현재 상태 → 배우자있음. "결혼한 적 있어요?" → 배우자있음·사별·이혼.
- "대학 나왔어요?"처럼 범위 질문은 해당되는 값을 모두 in에 넣으세요.
- 부정 질문("서울 안 살아요?")은 not_in.
- 위 열로 판정할 수 없는 질문(성격·취미·의견·소득·건강 등)은 kind=unmappable, conds=[].

질문: {question}"""


def validate(domains, con):
    def check(t: Translation):
        if t.kind == "filter" and not t.conds:
            raise ValueError("filter에는 conds가 필요")
        for c in t.conds:
            if c.column == "age":
                if c.op != "between" or c.lo is None or c.hi is None or not (0 <= c.lo <= c.hi <= 120):
                    raise ValueError("age는 between lo<=hi")
            elif c.column in TEXT_COLS:
                if c.op != "contains" or len(c.values) != 1 or not c.values[0].strip():
                    raise ValueError(f"{c.column}은 contains + 키워드 1개")
                if not con.execute(f"SELECT count(*) FROM people WHERE {c.column} LIKE ?", [f"%{c.values[0]}%"]).fetchone()[0]:
                    raise ValueError(f"{c.column}에 '{c.values[0]}'를 포함하는 값이 없음. 더 짧은 키워드를 쓰거나, 판정할 수 없으면 kind=unmappable")
            else:
                bad = [v for v in c.values if v not in domains[c.column]]
                if c.op not in ("in", "not_in") or not c.values or bad:
                    raise ValueError(f"{c.column} 값 오류 {bad}. 허용: {domains[c.column]}. 열로 판정할 수 없는 질문이면 kind=unmappable")
    return check


def to_sql(conds):
    parts, params = [], []
    for c in conds:
        if c.op == "between":
            parts.append("age BETWEEN ? AND ?"); params += [c.lo, c.hi]
        elif c.op == "contains":
            parts.append(f"{c.column} LIKE ?"); params.append(f"%{c.values[0]}%")
        else:
            ph = ", ".join("?" * len(c.values))
            parts.append(f"{c.column} {'NOT ' if c.op == 'not_in' else ''}IN ({ph})"); params += c.values
    return "(" + " AND ".join(parts) + ")", params


def describe(conds):
    out = []
    for c in conds:
        if c.op == "between":
            out.append(f"나이 {c.lo}~{c.hi}")
        else:
            op = {"in": "=", "not_in": "≠", "contains": "에 포함"}[c.op]
            out.append(f"{LABEL[c.column]} {op} {'/'.join(c.values)}" if c.op != "contains" else f"{LABEL[c.column]}에 '{c.values[0]}'")
    return " 그리고 ".join(out)


class Reply(BaseModel):
    answer: Literal["예", "아니오", "글쎄요"]
    line: str = Field(description="이 사람 말투의 한마디, 60자 이하")


GENERIC = {"현재", "전직", "구직중", "관련", "종사원", "단순", "일반", "기타", "그", "외", "및"}


def leak_words(row, asked):
    """아직 묻지 않은 열의 단서 단어. 지명은 시도와 시군구 어간(부평구→부평), 직업은 낱말."""
    words = set()
    if not {"province", "district"} & asked:
        prov, _, dist = row["district"].partition("-")
        words |= {prov, row["province"]} | {w[:-1] if len(w) > 2 and w[-1] in "시군구" else w for w in dist.split()}
    if not {"occupation", "employed"} & asked:
        words |= {w for w in re.split(r"[ ,·/]+", row["occupation"]) if w not in GENERIC}
    return {w for w in words if len(w) >= 2}


def persona_reply(router, row, question, verdict, asked=frozenset(), cols=()):
    """판정은 데이터가 이미 했다. 모델은 말투만: 확인된 속성 + 이번 질문의 정답만 보고 한마디.
    10/8: 전체 프로필을 주고 자유 응답시키면 묻지 않은 지명·직업 누설, 배우자 있는데 '남편이 먼저 갔다' 같은 모순이 나옴."""
    known = [f"{LABEL[c]}: {row[c]}" for c in CAT_COLS + TEXT_COLS if c in set(asked) | set(cols)]
    if "age" in set(asked) | set(cols):
        known.append(f"나이: {row['age']}세")
    speech = f"{row['age'] // 10 * 10}대 {row['sex']}, {row['province']} 지역 말투"
    if verdict is None:   # 데이터에 없는 질문은 성격 서술을 근거로 상상
        known.append(f"성격·일상: {row['persona']}")
        want = "글쎄요"
        rule = "이 질문은 데이터로 정해지지 않습니다. 성격·일상에 비춰 상상해서 answer를 고르세요(예/아니오/글쎄요)."
    else:
        want = "예" if verdict else "아니오"
        rule = f"answer는 반드시 '{want}'."
    banned = leak_words(row, set(asked) | set(cols))

    def check(r: Reply):
        if verdict is not None and r.answer != want:
            raise ValueError(f"answer는 '{want}'여야 함")
        if not r.line.strip() or len(r.line) > 60:
            raise ValueError("line은 1~60자")
        if re.search(r"[A-Za-z0-9\u3040-\u30ff\u4e00-\u9fff]", r.line):
            raise ValueError("line은 한글만(영어·숫자·한자 금지)")
        hit = [w for w in banned if w in r.line]
        if hit:
            raise ValueError(f"묻지 않은 단서 {hit} 금지")

    msg = [{"role": "system", "content": "당신은 스무고개의 숨은 인물입니다. 아래 '확인된 정보' 밖의 사실(나이·지역·직업·가족·집)은 말하지도 지어내지도 마세요.\n"
                                         f"말투: {speech}\n확인된 정보: {'; '.join(known) or '없음'}"},
           {"role": "user", "content": f"질문: {question}\n{rule} line은 이 사람다운 감정이 담긴 한국어 한마디(60자 이하)."}]
    r, _ = router.call("respondent", msg, schema=Reply, max_tokens=300, temperature=0.7, thinking=False,
                       retries=3, validator=check, tag="twenty-reply")
    if r is None:
        return want + "."
    return f"{r.answer}. {r.line}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int)
    ap.add_argument("--script")
    args = ap.parse_args()
    con = duckdb.connect()
    domains = load(con)
    total = con.execute("SELECT count(*) FROM people").fetchone()[0]
    seed = args.seed if args.seed is not None else random.randrange(1 << 30)
    row = con.execute("SELECT * FROM people ORDER BY hash(uuid || ?) LIMIT 1", [str(seed)]).df().iloc[0].to_dict()
    router = Router()
    scripted = [l.strip() for l in open(args.script, encoding="utf-8") if l.strip()] if args.script else None

    print(f"[seed {seed}] 한국 성인 {total:,}명 중 한 명을 숨겼습니다. 예/아니오 질문 {MAX_Q}번 안에 {WIN_AT}명 이하로 좁히세요.\n")
    where, params, remain, errs, asked = [], [], total, [], set()
    for qn in range(1, MAX_Q + 1):
        if scripted is not None:
            if not scripted:
                break
            line = scripted.pop(0); q, _, pred = line.partition("|")
            q, pred = q.strip(), pred.strip()
            print(f"Q{qn}. {q}")
        else:
            q = input(f"Q{qn}. ").strip()
            if not q:
                break
            pred = None
        # 10/8 비교(14문항): Super·추론 끔 12/14 1.7초 > Nano 9/14 > Super·추론 7/14 5.3초
        t, rec = router.call("extract", [{"role": "user", "content": translator_prompt(domains, q)}],
                             schema=Translation, max_tokens=600, temperature=0, retries=2, thinking=False,
                             validator=validate(domains, con), tag="twenty-translate")
        if t is None:
            print(f"   (번역 실패: {rec.error})\n"); continue
        if t.kind == "unmappable":
            print(f"   {persona_reply(router, row, q, None, asked)}")
            print(f"   [AI 상상 — 데이터에 없는 질문이라 인원은 그대로 {remain:,}명]\n"); continue

        sql, p = to_sql(t.conds)
        verdict = con.execute(f"SELECT {sql} FROM people WHERE uuid = ?", p + [row["uuid"]]).fetchone()[0]
        clause = sql if verdict else f"NOT {sql}"
        new_remain = con.execute(f"SELECT count(*) FROM people WHERE {' AND '.join(where + [clause])}", params + p).fetchone()[0]
        print(f"   해석: {describe(t.conds)}")
        if pred is None and scripted is None:
            pred = input(f"   몇 명 남을까요? (지금 {remain:,}명, 엔터=건너뛰기) ").strip()
        print(f"   {persona_reply(router, row, q, verdict, asked, [c.column for c in t.conds])}")
        tail = ""
        if pred:
            try:
                guess = int(pred.replace(",", ""))
                err = abs(math.log10(max(guess, 1) / max(new_remain, 1)))
                errs.append(err)
                tail = f" · 예측 {guess:,}명 ({'거의 정확' if err < 0.1 else f'{10 ** err:.1f}배 차이'})"
            except ValueError:
                pass
        flag = " · 합성 인구 기준(공식 통계와 차이 있는 축)" if any(c.column in SYNTHETIC_ONLY for c in t.conds) else ""
        share = new_remain / remain if remain else 0
        print(f"   → {'예' if verdict else '아니오'}: {remain:,}명 → {new_remain:,}명 (남은 사람의 {share:.0%}){tail}{flag}\n")
        where.append(clause); params += p; remain = new_remain
        asked |= {c.column for c in t.conds}
        if remain <= WIN_AT:
            print(f"성공! Q{qn}에서 {remain}명으로 좁혔습니다."); break
    else:
        print(f"질문을 다 썼습니다. 남은 사람 {remain:,}명.")

    print("\n[숨은 사람]")
    print(f"{row['age']}세 {row['sex']} · {row['district']} · {row['occupation']} · {row['marital_status']} · {row['housing_type']} · {row['education_level']}")
    print(row["persona"])
    if errs:
        print(f"\n팩트 감각: 평균 {10 ** (sum(errs) / len(errs)):.1f}배 차이 (1.0배 = 완벽)")
    print(f"비용 ${router.total_cost:.4f} · 호출 {len(router.calls)}회")


if __name__ == "__main__":
    main()
