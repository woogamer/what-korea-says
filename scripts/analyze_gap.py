"""시뮬 vs 실측 차이 분해 (docs/scenario-seoul-ev.md §6).

같은 응답자 600명(A에서 정박한 페르소나)에게 조건별로 같은 질문:
  E0 관측 없음(v1)  ·  E1 관측 있음(v2)  ·  E2 관측에서 친환경차 보유 여부 제외  ·  E3 관측 없음 + 같은 셀 실제 응답자 예시 3명(A, 사다리 1단)
채점은 B 셀 분포(M1)와 정박 응답자의 보류한 실측 계획(M2). 결과 → data/gap_analysis.json + 문서 §6.
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / ".env")

from evidence.seoul_survey import COMMUTE_KO, ECO, anchor_records, income_label, load  # noqa: E402
from llm import Router  # noqa: E402
from scripts.validate_seoul import OPT2KEY, OPTIONS, Q, SCOPE, housing_group, survey_cells  # noqa: E402
from sim.aggregate import tvd  # noqa: E402
from sim.run import run_sim  # noqa: E402
from world import ops  # noqa: E402
from world.domain import build_world, load_pack  # noqa: E402
from world.store import build_view, connect, view_name  # noqa: E402

N_ANCHORS = 600


def exemplars(A, band, hg, k=3, seed=1):
    d = A[(A.age_band == band) & (A.housing_type.map(housing_group) == hg)].dropna(subset=["commute_mode"])
    s = d.sample(n=min(k, len(d)), random_state=seed)
    ko = {"ev": "전기차 구매 계획", "hybrid": "하이브리드 구매 계획", "hydrogen": "수소차 구매 계획", "none": "친환경차 구매 계획 없음"}
    lines = [f"{i+1}) {int(r.age)}세 {r.sex}, {r.housing_type}, {COMMUTE_KO[r.commute_mode]} 통근, 대중교통 실천 '{ECO[int(r.eco_transit)] if r.eco_transit == r.eco_transit else '모름'}' → {ko[r.ev_plan]}"
             for i, r in enumerate(s.itertuples())]
    return "참고: 당신과 같은 연령대·주거 형태의 서울 시민 실제 조사 응답 예시입니다(당신의 답은 당신의 상황으로 정하세요).\n" + "\n".join(lines)


def main():
    t0 = time.time()
    df = load(); A, B = df[df.split == "A"], df[df.split == "B"]
    Bcells = survey_cells(B)
    con = connect(":memory:")
    pack = load_pack(ROOT / "domains/ev_charging_seoul.yaml")
    v1, _ = build_world(con, pack, "seoul v1")
    rt = Router(con=con, run_id="gap", budget_usd=1.0)

    recs = anchor_records(A, N_ANCHORS, seed=3)
    targets = {r["source_ref"]: r.pop("_target") for r in recs}
    _, f = ops.run(con, v1, "fork", {"name": "v2 관측"}); v2 = f["new_world_id"]
    _, ar = ops.run(con, v2, "anchor", {"records": [dict(r) for r in recs], "grade": "[사]", "seed": 3})
    _, f = ops.run(con, v1, "fork", {"name": "v2c 보유 제외"}); v2c = f["new_world_id"]
    recs_c = [{**r, "observed": {k: v for k, v in r["observed"].items() if k != "has_green_car"}} for r in recs]
    _, arc = ops.run(con, v2c, "anchor", {"records": recs_c, "grade": "[사]", "seed": 3})
    uuid2target = {an["uuid"]: targets[r["source_ref"]] for r, an in zip(recs, ar["anchors"])}
    assert [a["uuid"] for a in arc["anchors"]] == [a["uuid"] for a in ar["anchors"]], "같은 seed → 같은 정박 행이어야 함"

    # 응답자 600명을 셀(연령대×아파트/비아파트)로 묶는다
    v = view_name(v1); build_view(con, v1)
    ids = ",".join(repr(u) for u in uuid2target)
    rows = con.execute(f"SELECT uuid, age_band, CASE WHEN housing_type='아파트' THEN '아파트' ELSE '비아파트' END hg FROM {v} WHERE uuid IN ({ids})").fetchall()
    cells = {}
    for u, b, hg in rows:
        cells.setdefault(f"{b} × {hg}", []).append(u)
    labels = sorted(cells)
    base_spec = {"questions": [{"id": "q", "text": Q, "answer_schema": "choice", "options": OPTIONS}], "seed": 31, "temperature": 0.8}

    def spec(with_examples=False):
        cs = []
        for lab in labels:
            c = {"uuids": cells[lab], "n": len(cells[lab]), "label": lab}
            if with_examples:
                band, hg = lab.split(" × "); c["framing"] = exemplars(A, band, hg)
            cs.append(c)
        return {**base_spec, "cells": cs}

    conditions = [("E0 관측 없음", v1, spec()), ("E1 관측 있음", v2, spec()), ("E2 보유 여부 제외", v2c, spec()), ("E3 관측 없음+같은 셀 실측 예시 3", v1, spec(True))]
    R = {"ts": datetime.now().isoformat(timespec="seconds"), "n_respondents": len(uuid2target), "conditions": {}}
    for name, wid, sp in conditions:
        r = run_sim(con, wid, sp, rt)
        ans = {u: OPT2KEY.get(json.loads(a)["value"]) for u, a in con.execute("SELECT uuid, answer FROM sim_response WHERE run_id=?", [r["run_id"]]).fetchall()}
        # 전체 분포 vs B 전체(응답자 셀 구성으로 가중)
        overall = {k: 0 for k in OPT2KEY.values()}
        for val in ans.values():
            overall[val] += 1
        overall = {k: round(v / len(ans), 3) for k, v in overall.items()}
        # 셀별 TVD (B 가중)
        num = den = 0; per = []
        for ci, c in enumerate(sp["cells"]):
            sd = {OPT2KEY.get(k, k): v for k, v in r["aggregate"].get(str(ci), {}).get("q", {}).get("dist", {}).items()}
            b = Bcells.get(c["label"])
            if b and sd:
                t = tvd(sd, b["dist"]); num += t * b["n"]; den += b["n"]
                per.append({"cell": c["label"], "n": c["n"], "tvd": t, "sim": sd, "survey": b["dist"]})
        # M2: 실측 계획별 시뮬 답 분포
        m2 = {}
        for u, val in ans.items():
            t = uuid2target.get(u)
            if t:
                m2.setdefault(t, {}).setdefault(val, 0); m2[t][val] += 1
        m2n = {t: {k: round(v / sum(d.values()), 3) for k, v in d.items()} | {"n": sum(d.values())} for t, d in m2.items()}
        p_ev = m2n.get("ev", {}).get("ev", 0); p_none = m2n.get("none", {}).get("ev", 0)
        R["conditions"][name] = {"run_id": r["run_id"], "n_ok": r["n_ok"], "cost_usd": round(r["cost_usd"], 4), "overall": overall,
                                 "m1_tvd": round(num / den, 4), "m2": m2n, "lift": round(p_ev / p_none, 2) if p_none else None,
                                 "hit_any_green": _hit(ans, uuid2target), "cells": per}
        print(name, "| overall", overall, "| M1", R["conditions"][name]["m1_tvd"], "| lift", R["conditions"][name]["lift"], f"| ${r['cost_usd']:.3f}", flush=True)
    # 실측 B 전체 분포(응답자 셀 구성으로 가중)
    tot = sum(len(v) for v in cells.values()); Bw = {k: 0.0 for k in OPT2KEY.values()}
    for lab, us in cells.items():
        for k, p in Bcells.get(lab, {"dist": {}})["dist"].items():
            Bw[k] += p * len(us) / tot
    R["survey_B_weighted"] = {k: round(v, 3) for k, v in Bw.items()}
    R["cost_usd"] = round(rt.total_cost, 4); R["seconds"] = round(time.time() - t0)
    (ROOT / "data/gap_analysis.json").write_text(json.dumps(R, ensure_ascii=False, indent=1))
    print("survey B (weighted to respondent mix):", R["survey_B_weighted"], f"| total ${rt.total_cost:.3f} {R['seconds']}s")


def _hit(ans, targets):
    """친환경차(전기·하이브리드·수소) 계획 유무 이진 적중률과 다수 기준선."""
    n = hit = pos = 0
    for u, val in ans.items():
        t = targets.get(u)
        if not t:
            continue
        n += 1; a = val != "none"; b = t != "none"; hit += (a == b); pos += b
    return {"n": n, "accuracy": round(hit / n, 3), "majority_baseline": round(max(pos, n - pos) / n, 3), "actual_pos_rate": round(pos / n, 3)}


if __name__ == "__main__":
    main()
