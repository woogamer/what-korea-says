from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
L0_GLOB = str(ROOT / "data" / "personas" / "*.parquet")
DB_PATH = str(ROOT / "data" / "world.duckdb")

# 19세 이상 인구. L0 1행 = ADULT_POPULATION / L0 행수 명.
# TODO(architecture §10-3): KOSIS 주민등록 기준으로 확정.
ADULT_POPULATION = 44_000_000
