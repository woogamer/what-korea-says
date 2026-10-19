"""두 월드 버전의 차이."""
import json

from .segment import segment
from .store import active_attrs, get_world


def diff(con, a, b, axes=None, filter=None) -> dict:
    da = {x["name"]: x for x in active_attrs(con, a)}
    db = {x["name"]: x for x in active_attrs(con, b)}
    attrs = {
        "added": sorted(set(db) - set(da)),
        "removed": sorted(set(da) - set(db)),
        "changed": [{"name": n, "from": f"v{da[n]['version']} {da[n]['grade']}",
                     "to": f"v{db[n]['version']} {db[n]['grade']}"}
                    for n in sorted(set(da) & set(db)) if da[n]["attr_id"] != db[n]["attr_id"]],
    }
    q = "SELECT belief_id, proposition, grade, status FROM belief WHERE world_id=?"
    ba = {r[0]: r for r in con.execute(q, [a]).fetchall()}
    bb = {r[0]: r for r in con.execute(q, [b]).fetchall()}
    beliefs = {
        "added": [{"id": k, "proposition": bb[k][1], "grade": bb[k][2]} for k in bb if k not in ba],
        "removed": [k for k in ba if k not in bb],
        "status_changed": [{"id": k, "from": ba[k][3], "to": bb[k][3]} for k in ba if k in bb and ba[k][3] != bb[k][3]],
    }
    anchors = {w: con.execute("SELECT count(*) FROM anchor WHERE world_id=?", [w]).fetchone()[0] for w in (a, b)}
    out = {"attrs": attrs, "beliefs": beliefs, "anchors": anchors,
           "weight_ref": {a: get_world(con, a)["weight_ref"], b: get_world(con, b)["weight_ref"]}}
    if axes:
        sa = segment(con, a, axes, filter).set_index(axes)
        sb = segment(con, b, axes, filter).set_index(axes)
        m = sa[["size_est", "anchor_n"]].join(sb[["size_est", "anchor_n"]], lsuffix="_a", rsuffix="_b", how="outer").fillna(0)
        m["size_delta"] = m["size_est_b"] - m["size_est_a"]
        out["cells"] = m.reset_index().to_dict("records")
    return out
