"""질문별 답변 스키마 → 응답자 1인 출력 pydantic 모델(동적)."""
from typing import Literal

from pydantic import BaseModel, Field, create_model

ANSWER_TYPES = {
    "likert5": (int, Field(ge=1, le=5, description="1=전혀 아니다 … 5=매우 그렇다")),
    "yes_no": (Literal["yes", "no"], ...),
    "number": (float, ...),
    "text": (str, Field(max_length=200)),
}


def answer_field(q: dict):
    t = q.get("answer_schema", "text")
    if t == "choice":
        return (Literal[tuple(q["options"])], ...)
    return ANSWER_TYPES[t]


def response_model(questions: list[dict]) -> type[BaseModel]:
    fields = {}
    for q in questions:
        fields[q["id"]] = answer_field(q)
        fields[f"{q['id']}_reason"] = (str, Field(max_length=200, description="한 문장 이유"))
    return create_model("Response", **fields)


def positive(value, schema: str) -> float | None:
    """찬성률 계산용: likert5 ≥4, yes_no == yes."""
    if schema == "likert5":
        return 1.0 if int(value) >= 4 else 0.0
    if schema == "yes_no":
        return 1.0 if value == "yes" else 0.0
    return None
