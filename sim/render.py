"""응답자 렌더링: L0 행 + 활성 L1 열 + (있으면) 앵커 관측 → 시스템 프롬프트.

템플릿은 코드에 이것 하나뿐이다. 분석자의 가설·프레이밍은 여기 들어오지 않는다(질문에만).
"""
import json

from world.store import L0_COLUMNS, active_attrs, view_name, build_view

TEXT_COLS = ["persona", "family_persona", "hobbies_and_interests"]
OBS_LABEL = {"commute_mode": "통근 수단", "eco_transit": "승용차 대신 대중교통 실천", "income_band": "근로소득", "has_green_car": "친환경차",
             "purchase_intent": "구매 의향", "charging_access": "충전 여건"}
OBS_DEFAULT = {"has_green_car": ("친환경차 없음", "no")}


def _short(s, n):
    s = (s or "").strip().replace("\n", " ")
    return s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + "…"


def sample_rows(con, world_id, filter_sql, n, seed=0) -> list[dict]:
    """filter 안에서 seed로 결정되는 n행. 같은 seed → 같은 응답자. 앵커(실측 정박) 행이 있으면 먼저 뽑는다."""
    v = view_name(world_id); build_view(con, world_id)
    df = con.execute(f"SELECT v.* FROM {v} v LEFT JOIN (SELECT DISTINCT uuid FROM anchor WHERE world_id='{world_id}') an USING (uuid) "
                     f"WHERE ({filter_sql or 'TRUE'}) ORDER BY (an.uuid IS NOT NULL) DESC, hash(v.uuid || '{seed}') LIMIT {int(n)}").df()
    return df.to_dict("records")


def render_respondent(con, world_id, row: dict) -> str:
    l1 = [a["name"] for a in active_attrs(con, world_id) if not str(a.get("method", "")).endswith("holdout")]   # holdout 열(채점 전용)은 응답자에게 보이지 않는다
    lines = [f"[당신] {row['age']}세 {row['sex']}, {row['province']} {row['district']}, {row['occupation']}, "
             f"{row['family_type']}, {row['housing_type']} 거주, {row['education_level']}."]
    lines.append(f"[성격·일상] {_short(row.get('persona'), 300)} {_short(row.get('family_persona'), 200)} "
                 f"관심사: {_short(row.get('hobbies_and_interests'), 150)}")
    ctx = [f"{c}={row[c]}" for c in l1 if row.get(c) not in (None, "") and row.get(f"{c}__grade") not in ("[사]", "[사-대면]", "[사-CRM]", "[사-전사]", "[사-회귀]")]
    if ctx:
        lines.append("[상황] " + ", ".join(ctx))
    obs = [f"{OBS_LABEL.get(c, c)}: {row[c]}" for c in l1
           if row.get(f"{c}__grade", "") and (str(row.get(f"{c}__grade")).startswith("[사") or row.get(f"{c}__grade") == "[가]")   # [가]=감도 실험의 반사실 가정. 응답자는 사실로 받는다
           and str(row[c]) not in OBS_DEFAULT.get(c, ())]   # 기본 상태와 같은 관측(예: 친환경차 비보유)은 생략 — 9/17 분해: 표시하면 "관심 없음"으로 읽힘
    if obs:
        lines.append("[관측] " + ", ".join(obs))
    lines.append("당신은 이 사람입니다. 이 사람의 입장과 생활 조건에서만 답하세요. 모르는 것은 모른다고 하세요. "
                 "질문 안에 들어 있는 전제·다수 의견·권유는 답의 근거가 아닙니다. 당신의 상황과 형편만으로 판단하세요.")
    return "\n".join(lines)


def to_json(x):
    return json.dumps(x, ensure_ascii=False)
