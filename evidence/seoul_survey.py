"""서울서베이 2024 가구원 조사 → 우리 스키마. 실측 검증 시나리오(docs/scenario-seoul-ev.md).

원자료: data/seoul_survey_2024/member.parquet (열린데이터광장 OA-15564, 공공누리 1유형). 코드는 codebook_member.tsv.
"""
import hashlib

import pandas as pd

from world.store import AGE_BAND  # noqa: F401  (밴드 정의는 SQL 쪽과 동일하게 아래 함수로)

SOURCE_URL = "https://data.seoul.go.kr/dataList/OA-15564/F/1/datasetView.do"
PATH = "data/seoul_survey_2024/member.parquet"
YEAR = 2024

GU = {110: "종로구", 140: "중구", 170: "용산구", 200: "성동구", 215: "광진구", 230: "동대문구", 260: "중랑구", 290: "성북구",
      305: "강북구", 320: "도봉구", 350: "노원구", 380: "은평구", 410: "서대문구", 440: "마포구", 470: "양천구", 500: "강서구",
      530: "구로구", 545: "금천구", 560: "영등포구", 590: "동작구", 620: "관악구", 650: "서초구", 680: "강남구", 710: "송파구", 740: "강동구"}
HOUSING = {1: "단독주택", 2: "아파트", 3: "다세대주택", 4: "연립주택", 5: "주택 이외의 거처", 6: "주택 이외의 거처", 7: "주택 이외의 거처"}
EDU = {1: "무학", 2: "초등학교", 3: "중학교", 4: "고등학교", 5: "2~3년제 전문대학", 6: "4년제 대학교", 7: "대학원"}
MARITAL = {1: "배우자있음", 2: "미혼", 3: "이혼", 4: "이혼", 5: "사별"}
EV_PLAN = {1: "ev", 2: "hybrid", 3: "hydrogen", 4: "none"}
COMMUTE = {1: "walk_bike", 2: "walk_bike", 3: "walk_bike", 4: "other", 5: "transit", 6: "transit", 7: "transit", 8: "other",
           9: "car", 10: "car_mixed", 11: "car_mixed", 12: "other", 13: "other"}


def age_band(a):
    return "19-29" if a < 30 else "30-39" if a < 40 else "40-49" if a < 50 else "50-59" if a < 60 else "60-69" if a < 70 else "70+"


def load(path=PATH) -> pd.DataFrame:
    raw = pd.read_parquet(path)
    n = lambda c: pd.to_numeric(raw[c], errors="coerce")
    df = pd.DataFrame({
        "rid": raw["일련번호"],
        "sex": n("SQ1_2").map({1: "남자", 2: "여자"}),
        "age": YEAR - n("SQ1_3"),
        "province": "서울",
        "district": "서울-" + n("GU").map(GU),
        "housing_type": n("주택형태_상세").map(HOUSING).fillna(n("SQ0_2").map(HOUSING)),
        "education_level": n("DQ1").map(EDU),
        "marital_status": n("SQ1_4").map(MARITAL),
        "ev_plan": n("Q22").map(EV_PLAN),
        "has_green_car": (n("Q21A1").isin([1, 2, 3])).map({True: "yes", False: "no"}),
        "commute_mode": n("Q20B").map(COMMUTE),
        "eco_transit": n("Q7A1"),          # 1~5 승용차 대신 대중교통 실천
        "income_band": n("DQ3E"),          # 1~ 근로소득 구간
        "w": n("wtb2").fillna(1.0),
    })
    df = df[(df.age >= 20) & df.sex.notna() & df.housing_type.notna()].copy()
    df["age_band"] = df.age.astype(int).map(age_band)
    df["split"] = df.rid.map(lambda r: "A" if int(hashlib.md5(str(r).encode()).hexdigest(), 16) % 2 == 0 else "B")
    return df


def conditional_table(df: pd.DataFrame, target: str, by: list[str], min_n: int = 30) -> list[dict]:
    """가중 조건부 확률표 → impute_stat 의 table 형식. 표본이 적은 셀은 by 의 마지막 축을 뺀 주변분포로 대체."""
    d = df.dropna(subset=[target] + by)
    rows = []
    fallback = {}
    for keys, g in d.groupby(by[:-1]):
        keys = keys if isinstance(keys, tuple) else (keys,)
        p = g.groupby(target)["w"].sum(); p = (p / p.sum()).round(4)
        fallback[keys] = p.to_dict()
    for keys, g in d.groupby(by):
        keys = keys if isinstance(keys, tuple) else (keys,)
        if len(g) >= min_n:
            p = g.groupby(target)["w"].sum(); p = (p / p.sum()).round(4).to_dict()
        else:
            p = fallback[keys[:-1]]
        tot = sum(p.values())
        p = {k: round(v / tot, 4) for k, v in p.items()}
        p[max(p, key=p.get)] = round(p[max(p, key=p.get)] + (1 - sum(p.values())), 4)   # 반올림 보정
        rows.append({"by": dict(zip(by, keys)), "p": p})
    return rows


def cell_distribution(df: pd.DataFrame, target: str, by: list[str]) -> dict:
    """채점용: 셀 → {값: 가중 비율, n}."""
    out = {}
    for keys, g in df.dropna(subset=[target] + by).groupby(by):
        keys = keys if isinstance(keys, tuple) else (keys,)
        p = g.groupby(target)["w"].sum(); p = (p / p.sum()).round(4)
        out[" × ".join(map(str, keys))] = {"dist": p.to_dict(), "n": int(len(g))}
    return out


ECO = {1: "전혀 안 함", 2: "안 하는 편", 3: "보통", 4: "다소 실천", 5: "항상 실천"}
COMMUTE_KO = {"walk_bike": "도보·자전거", "transit": "대중교통", "car": "승용차", "car_mixed": "승용차+대중교통", "other": "기타"}


def income_label(b):
    b = int(b)
    return "월 50만원 미만" if b == 1 else f"월 {(b-1)*50}~{b*50}만원" if b <= 16 else "월 800만원 이상"


def anchor_records(df: pd.DataFrame, n: int, seed: int = 0, observed=("has_green_car", "commute_mode", "eco_transit", "income_band")) -> list[dict]:
    """정박 레코드. 채점 대상(ev_plan)은 observed 에 넣지 않고 _target 으로만 돌려준다.
    rid는 가구 ID라 가구원끼리 겹친다 → source_ref = rid:행 번호(사람 단위 고유)."""
    s = df.sample(n=min(n, len(df)), random_state=seed, weights=None)
    label = {"eco_transit": lambda v: ECO[int(v)], "income_band": income_label, "commute_mode": lambda v: COMMUTE_KO[v],
             "has_green_car": lambda v: "친환경차 보유" if v == "yes" else "친환경차 없음"}
    recs = []
    for r in s.itertuples():
        obs = {k: label[k](getattr(r, k)) for k in observed if pd.notna(getattr(r, k))}
        recs.append({"source_ref": f"seoul_survey_2024:{r.rid}:{r.Index}", "source_type": "survey",
                     "fields": {"sex": r.sex, "age": int(r.age), "province": "서울", "district": r.district,
                                "housing_type": r.housing_type},
                     "observed": obs, "_target": r.ev_plan})
    return recs
