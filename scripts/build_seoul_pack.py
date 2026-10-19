"""서울서베이 2024(split A) → 도메인 팩 + 의사결정 과제. 실측 검증 시나리오용.

  .venv/bin/python -m scripts.build_seoul_pack
"""
import yaml

from evidence.seoul_survey import SOURCE_URL, conditional_table, load

df = load()
A = df[df.split == "A"]
table = conditional_table(A, "ev_plan", ["housing_type", "age_band"])
share = A.groupby("ev_plan")["w"].sum().div(A.w.sum()).round(3).to_dict()

pack = {
    "name": "전기차 충전 · 서울 성인 (서울서베이 2024 실측 검증)",
    "market_scope": "province = '서울' AND age >= 20",
    "axes_default": ["charging_access", "age_band"],
    "beliefs": [
        {"id": "charging", "proposition": "집에서 충전 가능 여부는 주거 형태에 크게 좌우된다", "grade": "[가]",
         "source_ref": "수업 S1 데모 — 충전 여건 축", "variables": {"target": "charging_access", "by": ["housing_type"]}},
        {"id": "ev_plan_survey", "proposition": f"서울 성인의 친환경차 구매 계획: 전기차 {share.get('ev')}, 하이브리드 {share.get('hybrid')}, 수소 {share.get('hydrogen')}, 없음 {share.get('none')} (가중, 2024)",
         "grade": "[공]", "source_ref": SOURCE_URL, "n": int(len(A)),
         "sample_condition": "서울서베이 2024 가구원 조사 split A(50%). 만 20세+ 서울 거주자. 표준화 가중치 wtb2",
         "variables": {"target": "ev_plan", "by": ["housing_type", "age_band"]}},
    ],
    "attributes": [
        {"name": "charging_access", "method": "impute_stat", "by": ["housing_type"], "seed": 11, "belief": "charging",
         "table": [{"by": {"housing_type": h}, "p": p} for h, p in {
             "아파트": {"home": 0.55, "public_only": 0.45}, "단독주택": {"home": 0.70, "public_only": 0.30},
             "다세대주택": {"home": 0.20, "public_only": 0.80}, "연립주택": {"home": 0.25, "public_only": 0.75},
             "주택 이외의 거처": {"home": 0.10, "public_only": 0.90}, "비주거용 건물 내 주택": {"home": 0.10, "public_only": 0.90}}.items()],
         "checks": [{"type": "variance", "attr": "charging_access"}]},
    ],
    # ev_plan 임퓨트는 V1(통계 검증)에만 쓴다. 시뮬 월드에는 넣지 않는다(질문 대상이므로).
    "validation_attributes": [
        {"name": "ev_plan", "method": "impute_stat", "by": ["housing_type", "age_band"], "seed": 5, "belief": "ev_plan_survey",
         "table": table, "checks": [{"type": "variance", "attr": "ev_plan"}]},
    ],
}
yaml.safe_dump(pack, open("domains/ev_charging_seoul.yaml", "w", encoding="utf-8"), allow_unicode=True, sort_keys=False, width=200)

case = yaml.safe_load(open("cases/ev_charging.yaml", encoding="utf-8"))
case.update({
    "name": "ev_charging_seoul",
    "title": "전기차 충전 서비스 · 서울 — 어느 고객층에 먼저 진입할 것인가 (실측 검증)",
    "target_population": "서울 거주 20세 이상 성인 중 2년 내 친환경차 구매를 계획하는 사람",
    "world": {"pack": "domains/ev_charging_seoul.yaml", "market_scope": "province = '서울' AND age >= 20"},
})
case["evidence_map"]["in2"] = {"beliefs": ["ev_plan_survey"]}
yaml.safe_dump(case, open("cases/ev_charging_seoul.yaml", "w", encoding="utf-8"), allow_unicode=True, sort_keys=False, width=200)
print("pack cells", len(table), "share", share, "→ domains/ev_charging_seoul.yaml, cases/ev_charging_seoul.yaml")
