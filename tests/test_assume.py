from world import ops
from world.store import active_attrs, view_name


def test_assume_overrides_only_listed_rows_with_grade(con, world):
    ops.run(con, world, "extend", {"attr": "has_car", "domain": ["yes", "no"]})
    ids = [r[0] for r in con.execute(f"SELECT uuid FROM {view_name(world)} ORDER BY uuid LIMIT 3").fetchall()]
    _, r = ops.run(con, world, "assume", {"attr": "has_car", "uuids": ids, "value": "yes"})
    assert r["n_assumed"] == 3
    rows = con.execute(f"SELECT uuid, has_car, has_car__grade FROM {view_name(world)} ORDER BY uuid").fetchall()
    assert [x[1:] for x in rows[:3]] == [("yes", "[가]")] * 3
    assert all(x[1] is None for x in rows[3:])
    assert active_attrs(con, world)[0]["version"] == 2


def test_anchor_holdout_hidden_from_respondent(con, world):
    from sim.render import render_respondent
    from world.store import build_view, view_name
    build_view(con, world)
    row = con.execute(f"SELECT sex, age, district, housing_type FROM {view_name(world)} LIMIT 1").fetchone()
    rec = {"source_ref": "t:1", "fields": {"sex": row[0], "age": row[1], "district": row[2], "housing_type": row[3]},
           "observed": {"commute": "대중교통", "plan": "ev"}}
    _, r = ops.run(con, world, "anchor", {"records": [rec], "grade": "[사]", "holdout": ["plan"]})
    build_view(con, world)
    full = con.execute(f"SELECT * FROM {view_name(world)} WHERE uuid=?", [r["anchors"][0]["uuid"]]).df().to_dict("records")[0]
    text = render_respondent(con, world, full)
    assert "대중교통" in text and "plan" not in text and "ev" not in text.split("[관측]")[-1]
    assert full["plan"] == "ev"   # 월드에는 남는다
