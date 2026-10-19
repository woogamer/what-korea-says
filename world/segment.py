"""segment: 축의 교차 셀별 규모[도]·앵커 n·시뮬 가능. 상태를 바꾸지 않는 조회."""
import itertools
import json

import pandas as pd

from .grades import weakest
from .store import active_attrs, attr_grade, build_view, row_weight, view_name


def segment(con, world_id, axes, filter=None) -> pd.DataFrame:
    v = view_name(world_id); build_view(con, world_id)
    where = filter or "TRUE"
    rw = row_weight(con)
    sel = ", ".join(f"COALESCE(CAST({a} AS VARCHAR), '[미]') AS {a}" for a in axes)
    df = con.execute(
        f"SELECT {sel}, count(*) AS n_rows, sum(w) AS w_sum, "
        f"count(DISTINCT an.anchor_id) AS anchor_n "
        f"FROM {v} v LEFT JOIN (SELECT uuid, anchor_id FROM anchor WHERE world_id='{world_id}') an USING (uuid) "
        f"WHERE {where} GROUP BY {', '.join(axes)}").df()
    # 도메인 ∪ 관측값으로 전체 교차곱을 만들어 0인 셀도 행으로 남긴다
    defs = {a["name"]: a for a in active_attrs(con, world_id)}
    levels = []
    for a in axes:
        dom = json.loads(defs[a]["domain"]) if a in defs and defs[a]["domain"] else []
        levels.append(sorted(set(dom) | set(df[a].unique())))
    full = pd.DataFrame(list(itertools.product(*levels)), columns=axes)
    out = full.merge(df, on=axes, how="left").fillna({"n_rows": 0, "w_sum": 0.0, "anchor_n": 0})
    out["n_rows"] = out["n_rows"].astype(int); out["anchor_n"] = out["anchor_n"].astype(int)
    out["size_est"] = (out["w_sum"] * rw).round().astype(int)
    out["sim_available"] = out["n_rows"] > 0
    out["grade"] = weakest([attr_grade(con, world_id, a) for a in axes])
    return out.drop(columns=["w_sum"])
