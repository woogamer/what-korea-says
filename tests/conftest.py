import pytest

from world.store import connect, create_world


@pytest.fixture(scope="session")
def con(tmp_path_factory):
    """Deterministic test population; tests never require downloaded personas or API keys."""
    import pandas as pd
    from world.store import L0_COLUMNS
    rows = []
    for i in range(20000):
        row = {name: "테스트" for name in L0_COLUMNS if name != "age_band"}
        row.update(uuid=f"fixture-{i:06d}", age=19 + (i // 2) % 62,
                   sex="남자" if i % 2 == 0 else "여자",
                   province="서울" if (i // 124) % 2 == 0 else "경기",
                   district="강남구", housing_type="아파트" if (i // 248) % 2 else "단독주택",
                   family_type="2인 가구", education_level="대학교", occupation="회사원",
                   marital_status="기혼" if (i // 496) % 2 else "미혼", country="대한민국")
        rows.append(row)
    path = tmp_path_factory.mktemp("population") / "fixture.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    connection = connect(":memory:", l0_source=str(path))
    yield connection
    connection.close()


@pytest.fixture
def world(con):
    return create_world(con, "test")


COND_TABLE = [
    {"by": {"age_band": b, "sex": s}, "p": {"cond": p, "no": round(1 - p, 4)}}
    for b, s, p in [
        ("19-29", "남자", 0.01), ("19-29", "여자", 0.01),
        ("30-39", "남자", 0.04), ("30-39", "여자", 0.018),
        ("40-49", "남자", 0.11), ("40-49", "여자", 0.055),
        ("50-59", "남자", 0.17), ("50-59", "여자", 0.11),
        ("60-69", "남자", 0.23), ("60-69", "여자", 0.20),
        ("70+", "남자", 0.24), ("70+", "여자", 0.26),
    ]
]
