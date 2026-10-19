"""DuckDB 스키마, L0 뷰, 월드 뷰 구성."""
import json
import uuid as _uuid
from datetime import datetime

import duckdb

from . import config
from .grades import validate

DDL = """
CREATE TABLE IF NOT EXISTS world (
  world_id TEXT PRIMARY KEY, name TEXT, parent_world TEXT, head_op TEXT,
  weight_ref TEXT, created_at TIMESTAMP);
CREATE TABLE IF NOT EXISTS op_log (
  op_id TEXT PRIMARY KEY, world_id TEXT, seq INTEGER, op_type TEXT, params JSON,
  status TEXT, actor TEXT, result JSON, model_calls JSON, ts TIMESTAMP);
CREATE TABLE IF NOT EXISTS attr_def (
  world_id TEXT, attr_id TEXT, name TEXT, version INTEGER, dtype TEXT, domain JSON,
  grade TEXT, method TEXT, source_beliefs JSON, op_id TEXT, active BOOLEAN,
  PRIMARY KEY (world_id, attr_id));
CREATE TABLE IF NOT EXISTS attr_val (
  attr_id TEXT, uuid TEXT, v_str TEXT, v_num DOUBLE, confidence DOUBLE, grade TEXT,
  PRIMARY KEY (attr_id, uuid));
CREATE TABLE IF NOT EXISTS weight (
  world_id TEXT, version INTEGER, uuid TEXT, w DOUBLE,
  PRIMARY KEY (world_id, version, uuid));
CREATE TABLE IF NOT EXISTS belief (
  world_id TEXT, belief_id TEXT, proposition TEXT, variables JSON, grade TEXT,
  source_ref TEXT, n INTEGER, sample_condition TEXT, status TEXT, superseded_by TEXT,
  op_id TEXT, PRIMARY KEY (world_id, belief_id));
CREATE TABLE IF NOT EXISTS anchor (
  world_id TEXT, anchor_id TEXT, source_type TEXT, source_ref TEXT, uuid TEXT,
  match_score DOUBLE, matched_on JSON, observed JSON, op_id TEXT,
  PRIMARY KEY (world_id, anchor_id));
CREATE TABLE IF NOT EXISTS artifact (
  artifact_id TEXT PRIMARY KEY, world_id TEXT, tool TEXT, step TEXT, content JSON,
  grade_summary JSON, op_id TEXT, ts TIMESTAMP);
CREATE TABLE IF NOT EXISTS decision (
  decision_id TEXT PRIMARY KEY, artifact_id TEXT, gate TEXT, chosen JSON,
  rationale TEXT, actor TEXT, ts TIMESTAMP);
CREATE TABLE IF NOT EXISTS sim_run (
  run_id TEXT PRIMARY KEY, world_id TEXT, spec JSON, status TEXT, n_respondents INTEGER,
  cost_usd DOUBLE, calibration JSON, ts TIMESTAMP);
CREATE TABLE IF NOT EXISTS sim_response (
  run_id TEXT, uuid TEXT, question_id TEXT, answer JSON, model TEXT, tokens INTEGER,
  PRIMARY KEY (run_id, uuid, question_id));
CREATE TABLE IF NOT EXISTS calibration (
  card_id TEXT PRIMARY KEY, world_id TEXT, run_id TEXT, question_id TEXT, observed_attr TEXT,
  observed_grade TEXT, n_observed INTEGER, lam DOUBLE, m1_before DOUBLE, m1_after DOUBLE,
  level1_open BOOLEAN, card JSON, ts TIMESTAMP);
CREATE TABLE IF NOT EXISTS llm_call (
  call_id TEXT PRIMARY KEY, run_id TEXT, role TEXT, model TEXT, tag TEXT,
  prompt_tokens INTEGER, completion_tokens INTEGER, cost_usd DOUBLE, latency_s DOUBLE,
  ok BOOLEAN, error TEXT, attempts INTEGER, ts TIMESTAMP);
"""

AGE_BAND = ("CASE WHEN age < 30 THEN '19-29' WHEN age < 40 THEN '30-39' "
            "WHEN age < 50 THEN '40-49' WHEN age < 60 THEN '50-59' "
            "WHEN age < 70 THEN '60-69' ELSE '70+' END")

L0_COLUMNS = [
    "uuid", "sex", "age", "age_band", "marital_status", "military_status", "family_type",
    "housing_type", "education_level", "bachelors_field", "occupation", "district",
    "province", "country",
    "persona", "professional_persona", "sports_persona", "arts_persona", "travel_persona",
    "culinary_persona", "family_persona", "cultural_background", "skills_and_expertise",
    "skills_and_expertise_list", "hobbies_and_interests", "hobbies_and_interests_list",
    "career_goals_and_ambitions",
]


def new_id(prefix: str) -> str:
    return f"{prefix}_{_uuid.uuid4().hex[:10]}"


def connect(db_path: str = ":memory:", l0_source: str | None = None,
            sample_rows: int | None = None) -> duckdb.DuckDBPyConnection:
    """DB 열기 + 스키마 보장 + L0 뷰. sample_rows는 테스트용 축소본."""
    con = duckdb.connect(db_path)
    con.execute(DDL)
    src = l0_source or config.L0_GLOB
    if sample_rows:
        con.execute(f"CREATE OR REPLACE TABLE l0_raw AS SELECT * FROM read_parquet('{src}') "
                    f"USING SAMPLE {sample_rows} ROWS (reservoir, 42)")
        base = "l0_raw"
    else:
        base = f"read_parquet('{src}')"
    con.execute(f"CREATE OR REPLACE VIEW l0 AS SELECT *, {AGE_BAND} AS age_band FROM {base}")
    return con


def row_weight(con) -> float:
    n = con.execute("SELECT count(*) FROM l0").fetchone()[0]
    return config.ADULT_POPULATION / n


def create_world(con, name: str, parent: str | None = None, weight_ref: str | None = None) -> str:
    wid = new_id("w")
    con.execute("INSERT INTO world VALUES (?,?,?,?,?,?)",
                [wid, name, parent, None, weight_ref, datetime.now()])
    return wid


def get_world(con, world_id: str) -> dict:
    row = con.execute("SELECT world_id,name,parent_world,head_op,weight_ref FROM world WHERE world_id=?",
                      [world_id]).fetchone()
    if not row:
        raise KeyError(world_id)
    return dict(zip(["world_id", "name", "parent_world", "head_op", "weight_ref"], row))


def active_attrs(con, world_id: str) -> list[dict]:
    rows = con.execute(
        "SELECT attr_id,name,version,dtype,domain,grade,method FROM attr_def "
        "WHERE world_id=? AND active ORDER BY name", [world_id]).fetchall()
    return [dict(zip(["attr_id", "name", "version", "dtype", "domain", "grade", "method"], r))
            for r in rows]


def attr_grade(con, world_id: str, col: str) -> str:
    """열 이름 → 등급. L0 열은 [공]."""
    if col in L0_COLUMNS:
        return "[공]"
    for a in active_attrs(con, world_id):
        if a["name"] == col:
            return a["grade"]
    raise KeyError(f"{col} not in world {world_id}")


def view_name(world_id: str) -> str:
    return f"v_{world_id}"


def build_view(con, world_id: str) -> str:
    """L0 + 활성 L1/L2 열 + 가중치 w 를 한 뷰로. 열마다 <name>__grade 동반."""
    w = get_world(con, world_id)
    selects = ["l0.*", "COALESCE(wt.w, 1.0) AS w"]
    joins = []
    if w["weight_ref"]:
        wid, ver = w["weight_ref"].split(":")
        joins.append(f"LEFT JOIN (SELECT uuid, w FROM weight WHERE world_id='{wid}' "
                     f"AND version={int(ver)}) wt USING (uuid)")
    else:
        selects[1] = "1.0 AS w"
    for i, a in enumerate(active_attrs(con, world_id)):
        col = "v_num" if a["dtype"] == "numeric" else "v_str"
        joins.append(f"LEFT JOIN (SELECT uuid, {col} AS v, grade FROM attr_val "
                     f"WHERE attr_id='{a['attr_id']}') a{i} USING (uuid)")
        selects.append(f"a{i}.v AS {a['name']}")
        selects.append(f"a{i}.grade AS {a['name']}__grade")
    name = view_name(world_id)
    con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT {', '.join(selects)} "
                f"FROM l0 {' '.join(joins)}")
    return name


def add_belief(con, world_id: str, proposition: str, grade: str, source_ref: str,
               variables: dict | None = None, n: int | None = None,
               sample_condition: str | None = None, status: str = "committed",
               op_id: str | None = None) -> str:
    validate(grade)
    if grade == "[공]" and not (source_ref or "").startswith("http"):
        raise ValueError("[공] belief requires URL source_ref")
    bid = new_id("b")
    con.execute("INSERT INTO belief VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                [world_id, bid, proposition, json.dumps(variables or {}, ensure_ascii=False),
                 grade, source_ref, n, sample_condition, status, None, op_id])
    return bid
