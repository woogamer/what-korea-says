from evidence.agent import StatRow, StatTable, check_table
from world.reconcile import l0_consistency

REQ = {"name": "commute_car", "domain": ["car", "other"], "by": ["age_band"]}
DOM = {"age_band": {"19-29": 0.16, "30-39": 0.18, "40-49": 0.17, "50-59": 0.18, "60-69": 0.16, "70+": 0.15}}


def _t(rows, url="https://x/1", year="2024"):
    return StatTable(found=True, rows=[StatRow(by={"age_band": b}, p=p) for b, p in rows], url=url, year=year)


def test_check_table_passes_and_normalizes():
    rows = [(b, {"car": 0.3, "other": 0.69}) for b in DOM["age_band"]]
    r = check_table(_t(rows), REQ, DOM, {"https://x/1"})
    assert r["passed"] and r["coverage"] == 1.0 and all(abs(sum(x["p"].values()) - 1) < 1e-6 for x in r["rows"])


def test_check_table_rejects_bad_mapping_denominator_url_coverage():
    r = check_table(_t([("20대", {"car": 0.3, "other": 0.7})]), REQ, DOM, {"https://x/1"})
    assert any("매핑 실패" in p for p in r["problems"]) and not r["passed"]
    r = check_table(_t([("19-29", {"car": 0.3, "other": 0.3})]), REQ, DOM, {"https://x/1"})
    assert any("분모" in p for p in r["problems"])
    r = check_table(_t([(b, {"car": 0.3, "other": 0.7}) for b in DOM["age_band"]], url="https://y/2"), REQ, DOM, {"https://x/1"})
    assert any("URL" in p for p in r["problems"])
    r = check_table(_t([("19-29", {"car": 0.3, "other": 0.7}), ("30-39", {"car": 0.3, "other": 0.7})]), REQ, DOM, {"https://x/1"})
    assert any("커버리지" in p for p in r["problems"]) and r["coverage"] == 0.34
    assert not check_table(StatTable(found=False), REQ, DOM, set())["passed"]


def test_l0_consistency_card(con, world):
    from world.store import build_view, view_name
    build_view(con, world)
    rows = con.execute(f"SELECT sex, count(*) FROM {view_name(world)} GROUP BY 1").fetchall()
    tot = sum(r[1] for r in rows); ref = {r[0]: r[1] / tot for r in rows}
    r = l0_consistency(con, world, {"sex": ref, "age_band": {"19-29": 1.0}})
    by = {a["col"]: a for a in r["axes"]}
    assert by["sex"]["tvd"] < 0.01 and by["sex"]["passed"]
    assert by["age_band"]["tvd"] > 0.5 and not by["age_band"]["passed"] and not r["passed"]


def test_check_table_marginal_fallback_and_impute_with_empty_by(con, world):
    from world import ops
    from world.store import view_name, build_view
    t = StatTable(found=True, rows=[StatRow(by={}, p={"car": 0.22, "other": 0.78})], url="https://x/1", year="2024", note="주변분포")
    r = check_table(t, REQ, DOM, {"https://x/1"})
    assert r["passed"] and r["evidence_level"] == "marginal" and r["warnings"] and r["rows"][0]["by"] == {}
    _, res = ops.run(con, world, "impute_stat", {"attr": "commute_car", "by": [], "table": r["rows"], "grade": "[도]", "seed": 3})
    build_view(con, world)
    n_car, n = con.execute(f"SELECT sum(CASE WHEN commute_car='car' THEN 1 ELSE 0 END), count(commute_car) FROM {view_name(world)}").fetchone()
    assert n > 0 and abs(n_car / n - 0.22) < 0.03


def test_check_table_rejects_numbers_absent_from_source():
    rows = [(b, {"car": 0.3, "other": 0.7}) for b in DOM["age_band"]]
    ok = check_table(_t(rows), REQ, DOM, {"https://x/1"}, source_text="연령대별 승용차 이용률은 모두 30%였다")
    bad = check_table(_t(rows), REQ, DOM, {"https://x/1"}, source_text="승용차 이용률은 22.1%였다")
    assert ok["passed"] and not bad["passed"] and any("원문에 없음" in p for p in bad["problems"])


def test_conditional_failure_degrades_to_marginal():
    t = StatTable(found=True, rows=[StatRow(by={"age_band": "40-59"}, p={"car": 0.3, "other": 0.7}), StatRow(by={}, p={"car": 0.22, "other": 0.78})],
                  url="https://x/1", year="2024")
    r = check_table(t, REQ, DOM, {"https://x/1"}, source_text="승용차 22.1%")
    assert r["passed"] and r["evidence_level"] == "marginal" and any("강등" in w for w in r["warnings"])


def test_value_keys_are_normalized_to_domain():
    t = StatTable(found=True, rows=[StatRow(by={b}, p={"Car": 0.3, " OTHER": 0.7}) if False else StatRow(by={"age_band": b}, p={"Car": 0.3, " OTHER": 0.7}) for b in DOM["age_band"]], url="https://x/1", year="2024")
    r = check_table(t, REQ, DOM, {"https://x/1"}, source_text="30%")
    assert r["passed"] and set(r["rows"][0]["p"]) == {"car", "other"}
