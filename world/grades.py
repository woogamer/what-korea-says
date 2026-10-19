"""근거 등급. 숫자가 클수록 약한 근거."""

GRADE_RANK = {
    "[공]": 0,
    "[사-CRM]": 1, "[사-전사]": 1, "[사-회귀]": 1, "[사-대면]": 1, "[사]": 1,
    "[현]": 2,
    "[도]": 3,
    "[도-LLM]": 4,
    "[시]": 5,
    "[가]": 6,
    "[미]": 7,
}


def validate(grade: str) -> str:
    if grade not in GRADE_RANK:
        raise ValueError(f"unknown grade {grade!r}; allowed: {list(GRADE_RANK)}")
    return grade


def weakest(grades) -> str:
    """여러 등급 중 가장 약한 것 (셀·산출물 등급 = 구성 요소 중 최약)."""
    grades = [validate(g) for g in grades if g]
    return max(grades, key=GRADE_RANK.get) if grades else "[미]"
