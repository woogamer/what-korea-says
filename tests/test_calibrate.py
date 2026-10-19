from sim.calibrate import apply_ratio, apply_transition, ratio_weights, transition_table

KEYS = ["ev", "none"]


def test_transition_rows_sum_to_one_and_apply():
    T = transition_table([("none", "ev"), ("none", "none"), ("ev", "ev")], KEYS, alpha=0.0)
    assert T == {"ev": {"ev": 1.0, "none": 0.0}, "none": {"ev": 0.5, "none": 0.5}}
    assert apply_transition({"ev": 0.2, "none": 0.8}, T) == {"ev": 0.6, "none": 0.4}


def test_ratio_preserves_cell_differences():
    w = ratio_weights({"ev": 0.1, "none": 0.9}, {"ev": 0.2, "none": 0.8})
    lo, hi = apply_ratio({"ev": 0.05, "none": 0.95}, w), apply_ratio({"ev": 0.2, "none": 0.8}, w)
    assert lo["ev"] < hi["ev"] and abs(sum(hi.values()) - 1) < 1e-6


def test_calibration_card_scores_and_locks(con, world):
    from world import ops
    from world.store import build_view, view_name
    from sim.calibrate import calibration_card
    build_view(con, world)
    ids = [r[0] for r in con.execute(f"SELECT uuid FROM {view_name(world)} WHERE age_band='30-39' LIMIT 100").fetchall()]
    ops.run(con, world, "extend", {"attr": "plan", "domain": ["ev", "none"]})
    ops.run(con, world, "assume", {"attr": "plan", "uuids": ids[:30], "value": "ev", "grade": "[사]"})
    ops.run(con, world, "assume", {"attr": "plan", "uuids": ids[30:], "value": "none", "grade": "[사]"})
    spec = {"cells": [{"filter": "age_band='30-39'", "n": 50, "label": "30대"}]}
    run = {"run_id": "sim_test", "aggregate": {"0": {"q": {"n": 50, "dist": {"전기차": 0.1, "없음": 0.9}}}}}
    q = {"id": "q", "options": ["전기차", "없음"], "observed_attr": "plan", "value_map": {"전기차": "ev", "없음": "none"}}
    card = calibration_card(con, world, run, spec, q)
    assert card["status"] == "ok" and card["n_observed"] == 100 and card["observed_marginal"] == {"ev": 0.3, "none": 0.7}
    assert card["m1_after"] < card["m1_before"]
    assert card["evaluation_kind"] == "fit_only" and not card["level1_open"]
    assert con.execute("SELECT count(*) FROM calibration WHERE world_id=?", [world]).fetchone()[0] == 1
    missing = calibration_card(con, world, run, spec, {**q, "observed_attr": "nope"})
    assert missing["status"] == "no_observed" and missing["level1_open"] is False


def test_claim_lock_rules():
    from report.render import claim_lock
    assert claim_lock({"level": 0}, []) is None
    assert "보정 카드 없음" in claim_lock({"level": 1}, [])
    assert claim_lock({"level": 1}, [{"level1_open": True}]) is not None
    assert claim_lock({"level": 1}, [{"level1_open": False, "m1_after": 0.2, "level1_max_m1": 0.1}])
