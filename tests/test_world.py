import pytest

from tests.conftest import COND_TABLE
from world import ops, reconcile
from world.diff import diff
from world.segment import segment
from world.store import active_attrs, build_view, get_world, view_name


def col(con, world, name):
    return con.execute(f"SELECT {name} FROM {view_name(world)} ORDER BY uuid").df()[name]


def test_extend_is_all_missing(con, world):
    oid, r = ops.run(con, world, "extend", {"attr": "stage", "domain": ["a", "b"]})
    assert con.execute("SELECT status FROM op_log WHERE op_id=?", [oid]).fetchone()[0] == "committed"
    a = active_attrs(con, world)
    assert a[0]["name"] == "stage" and a[0]["grade"] == "[미]"
    assert col(con, world, "stage").isna().all()


def test_impute_stat_reproducible_and_calibrated(con, world):
    p = {"attr": "cond", "by": ["age_band", "sex"], "table": COND_TABLE, "seed": 7}
    ops.run(con, world, "impute_stat", p)
    v1 = col(con, world, "cond").copy()
    ops.run(con, world, "impute_stat", p)          # 같은 seed → 같은 값, 새 버전
    assert (col(con, world, "cond") == v1).all()
    assert active_attrs(con, world)[0]["version"] == 2
    ops.run(con, world, "impute_stat", {**p, "seed": 8})
    assert not (col(con, world, "cond") == v1).all()
    share = con.execute(f"SELECT avg(CASE WHEN cond='cond' THEN 1.0 ELSE 0.0 END) FROM {view_name(world)} WHERE age_band='60-69' AND sex='남자'").fetchone()[0]
    assert abs(share - 0.23) < 0.03
    assert col(con, world, "cond__grade").dropna().unique().tolist() == ["[도]"]


def test_impute_rule_numeric(con, world):
    ops.run(con, world, "impute_rule", {"attr": "age_x2", "expr": "age * 2", "dtype": "numeric", "grade": "[도]"})
    df = con.execute(f"SELECT age, age_x2 FROM {view_name(world)} LIMIT 5").df()
    assert (df.age_x2 == df.age * 2).all()


def test_reweight_hits_marginals(con, world):
    targets = {"sex": {"남자": 0.7, "여자": 0.3}, "age_band": {"50-59": 0.5, "60-69": 0.5}}
    _, r = ops.run(con, world, "reweight", {"targets": targets, "filter": "age >= 50"})
    assert abs(r["achieved"]["sex"]["남자"] - 0.7) < 0.01
    assert abs(r["achieved"]["age_band"]["50-59"] - 0.5) < 0.01
    assert r["achieved"]["age_band"].get("70+", 0) == 0          # 목표에 없는 값 → 0
    w_young = con.execute(f"SELECT avg(w) FROM {view_name(world)} WHERE age < 50").fetchone()[0]
    assert w_young == 1.0                                          # filter 밖은 그대로


def test_anchor_matches_and_overrides(con, world):
    ops.run(con, world, "impute_stat", {"attr": "cond", "by": ["age_band", "sex"], "table": COND_TABLE, "seed": 1})
    before = con.execute(f"SELECT count(*) FROM {view_name(world)} WHERE cond='cond'").fetchone()[0]
    rec = {"source_ref": "대면-0820-03", "source_type": "interview",
           "fields": {"sex": "여자", "age": 63, "province": "경기"},
           "observed": {"cond": "cond", "urgency": "W1"}}
    _, r = ops.run(con, world, "anchor", {"records": [rec], "grade": "[사-대면]"})
    uid = r["anchors"][0]["uuid"]
    assert r["anchors"][0]["match_score"] == 1.0
    row = con.execute(f"SELECT sex, age, cond, cond__grade, urgency, urgency__grade FROM {view_name(world)} WHERE uuid=?", [uid]).fetchone()
    assert row == ("여자", 63, "cond", "[사-대면]", "W1", "[사-대면]") or (row[0] == "여자" and abs(row[1] - 63) <= 2 and row[2:] == ("cond", "[사-대면]", "W1", "[사-대면]"))
    after = con.execute(f"SELECT count(*) FROM {view_name(world)} WHERE cond='cond'").fetchone()[0]
    assert after - before in (0, 1)                                # 나머지 값 보존
    assert con.execute("SELECT count(*) FROM anchor WHERE world_id=?", [world]).fetchone()[0] == 1


def test_segment_has_zero_cells_and_sums(con, world):
    ops.run(con, world, "impute_stat", {"attr": "cond", "by": ["age_band", "sex"], "table": COND_TABLE, "seed": 1})
    ops.run(con, world, "extend", {"attr": "urgency", "domain": ["W1", "W2", "W3"]})
    df = segment(con, world, ["cond", "urgency"], filter="age >= 30")
    assert set(df.urgency) == {"W1", "W2", "W3", "[미]"}
    assert (df[df.urgency != "[미]"].n_rows == 0).all()            # 값 없는 셀도 행으로 존재
    assert not df[df.urgency != "[미]"].sim_available.any()
    pop = con.execute("SELECT count(*) FROM l0 WHERE age >= 30").fetchone()[0]
    assert df.n_rows.sum() == pop
    assert df.grade.iloc[0] == "[미]"                              # 최약 등급


def test_fork_and_perturb_isolated(con, world):
    ops.run(con, world, "impute_rule", {"attr": "flag", "expr": "'base'", "grade": "[도]"})
    _, r = ops.run(con, world, "fork", {"name": "scenario"})
    f = r["new_world_id"]
    assert get_world(con, f)["parent_world"] == world
    assert col(con, f, "flag").eq("base").all()
    ops.run(con, f, "perturb", {"attr": "flag", "expr": "'changed'"})
    assert col(con, f, "flag").eq("changed").all()
    assert col(con, world, "flag").eq("base").all()
    assert col(con, f, "flag__grade").eq("[가]").all()
    d = diff(con, world, f)
    assert d["attrs"]["changed"][0]["name"] == "flag"


def test_denominator_check():
    funnel = {"inquiry": 100, "visit": 14, "buy": 8}
    bad = reconcile.denominator(funnel, [{"from": "visit", "to": "buy", "rate": 0.96}])
    ok = reconcile.denominator(funnel, [{"from": "visit", "to": "buy", "rate": 0.57}])
    assert not bad["passed"] and bad["detail"]["conflicts"][0]["implied"] == 0.571
    assert ok["passed"]


def test_variance_check_rejects_constant(con, world):
    ops.run(con, world, "impute_rule", {"attr": "const", "expr": "'same'", "grade": "[도]"})
    assert not reconcile.variance(con, world, "const")["passed"]
    assert reconcile.variance(con, world, "sex")["passed"]


def test_failed_check_rejects_op(con, world):
    ops.run(con, world, "impute_stat", {"attr": "cond", "by": ["age_band", "sex"], "table": COND_TABLE, "seed": 1})
    v1 = active_attrs(con, world)[0]["attr_id"]
    oid, r = ops.run(con, world, "impute_rule", {
        "attr": "cond", "expr": "'cond'", "grade": "[도]",
        "checks": [{"type": "total", "expr": "cond='cond'", "expected": 5_330_000, "tol": 0.05, "filter": "age>=30"}]})
    assert con.execute("SELECT status FROM op_log WHERE op_id=?", [oid]).fetchone()[0] == "rejected"
    assert not r["checks"][0]["passed"]
    assert active_attrs(con, world)[0]["attr_id"] == v1               # 이전 버전 복원


def test_approval_required(con, world):
    oid = ops.propose(con, world, "extend", {"attr": "x"})
    with pytest.raises(PermissionError):
        ops.apply(con, oid)


def test_anchor_bias_check(con, world):
    ops.run(con, world, "anchor", {"records": [
        {"source_ref": f"r{i}", "fields": {"sex": "여자", "age_band": "60-69"}} for i in range(5)], "grade": "[사-대면]"})
    r = reconcile.anchor_bias(con, world, ["sex"], filter="age>=30")
    assert r["detail"]["n_anchor"] == 5 and not r["passed"]        # 전원 여자 → 편향 큼


def test_impute_chain_not_correlated(con, world):
    """같은 seed로 연쇄 임퓨트해도 열마다 난수가 독립이어야 한다."""
    ops.run(con, world, "impute_stat", {"attr": "cond", "by": ["age_band", "sex"], "table": COND_TABLE, "seed": 1})
    ops.run(con, world, "impute_stat", {"attr": "stage", "by": ["cond"], "seed": 1, "table": [
        {"by": {"cond": "cond"}, "p": {"a": 0.8, "b": 0.2}}, {"by": {"cond": "no"}, "p": {"none": 1.0}}]})
    share = con.execute(f"SELECT avg(CASE WHEN stage='b' THEN 1.0 ELSE 0.0 END) FROM {view_name(world)} WHERE cond='cond'").fetchone()[0]
    assert abs(share - 0.2) < 0.04


def test_domain_pack_build(con, tmp_path):
    """팩 YAML → 월드. 코어는 업종을 모른다."""
    from world.domain import build_world, load_pack
    pack_yaml = """
name: 테스트 업종
market_scope: "age >= 20"
axes_default: [flag, tier]
beliefs:
  - {id: b1, proposition: "성별 플래그 비율", grade: "[가]", source_ref: "test"}
attributes:
  - name: flag
    method: impute_stat
    by: [sex]
    seed: 3
    belief: b1
    table:
      - {by: {sex: 남자}, p: {yes: 0.3, no: 0.7}}
      - {by: {sex: 여자}, p: {yes: 0.6, no: 0.4}}
    checks: [{type: variance, attr: flag}]
  - {name: tier, method: extend, domain: [t1, t2]}
"""
    p = tmp_path / "pack.yaml"; p.write_text(pack_yaml, encoding="utf-8")
    w, results = build_world(con, load_pack(p))
    assert [r["status"] for r in results] == ["committed", "committed"]
    names = {a["name"] for a in active_attrs(con, w)}
    assert names == {"flag", "tier"}
    assert con.execute("SELECT count(*) FROM belief WHERE world_id=?", [w]).fetchone()[0] == 1
    df = segment(con, w, ["flag", "tier"], filter="age >= 20")
    assert df.n_rows.sum() == con.execute("SELECT count(*) FROM l0 WHERE age >= 20").fetchone()[0]
