"""검증 시나리오 V6 — 감도(반사실 차이) 검증. docs/scenario-seoul-ev.md §8. 사전 등록 기준은 문서에, 여기는 실행만.

같은 합성 응답자 N명에게 속성 하나만 바꿔(assume, [가]) 같은 질문을 묻는다. 축 4개 × 수준 2 + 기준선 = 9회.
  Δ_sim(축) = 같은 사람 안에서 hi − lo 의 친환경차 계획 비율 차 (짝지은 차이 → SE 작음)
  Δ_real(축) = B에서 같은 축 hi 집단 − lo 집단, 셀(연령대×주거)별로 낸 뒤 시뮬 표본의 셀 구성으로 표준화(연령·주거 교란 제거)
결과 → data/validation_sensitivity.json + 문서 §8.
"""
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / ".env")

from evidence.seoul_survey import load  # noqa: E402
from llm import Router  # noqa: E402
from scripts.validate_seoul import BANDS, OPT2KEY, OPTIONS, Q, SCOPE  # noqa: E402
from sim.run import run_sim  # noqa: E402
from world import ops  # noqa: E402
from world.domain import build_world, load_pack  # noqa: E402
from world.store import connect  # noqa: E402

# 축: (열, lo 라벨(렌더용), hi 라벨, 실측 lo 조건, 실측 hi 조건)
AXES = {
    "has_green_car": ("친환경차 없음", "친환경차 보유", lambda d: d.has_green_car == "no", lambda d: d.has_green_car == "yes"),
    "commute_mode": ("대중교통", "승용차", lambda d: d.commute_mode == "transit", lambda d: d.commute_mode == "car"),
    "eco_transit": ("전혀 안 함", "항상 실천", lambda d: d.eco_transit == 1, lambda d: d.eco_transit == 5),
    "income_band": ("월 150~200만원", "월 650~700만원", lambda d: d.income_band.between(3, 5), lambda d: d.income_band.between(13, 17)),
}
MIN_N_LEVEL = 20   # 셀×수준 실측 최소 표본


def real_delta(B, axis, cell_weights):
    """셀별 (hi − lo)를 시뮬 표본 셀 구성으로 가중 평균. any = 친환경차(전기·하이브리드·수소) 계획, ev = 전기차."""
    _, _, lo_f, hi_f = AXES[axis]
    d = B.copy(); d["hg"] = d.housing_type.map(lambda h: "아파트" if h == "아파트" else "비아파트"); d["cell"] = d.age_band + " × " + d.hg
    d["any"] = (d.ev_plan != "none").astype(float); d["ev"] = (d.ev_plan == "ev").astype(float)
    out = {}
    for y in ("any", "ev"):
        num = den = 0.0; used = 0
        for cell, wc in cell_weights.items():
            c = d[d.cell == cell]; lo, hi = c[lo_f(c)], c[hi_f(c)]
            if len(lo) < MIN_N_LEVEL or len(hi) < MIN_N_LEVEL:
                continue
            delta = (hi[y] * hi.w).sum() / hi.w.sum() - (lo[y] * lo.w).sum() / lo.w.sum()
            num += delta * wc; den += wc; used += 1
        out[y] = round(float(num / den), 4) if den else None
        out[f"{y}_cells_used"] = used
    return out


def paired(ans_lo, ans_hi, key):
    """같은 응답자의 hi − lo 짝지은 차이. key(answer)->0/1."""
    ids = [u for u in ans_lo if u in ans_hi]
    d = [key(ans_hi[u]) - key(ans_lo[u]) for u in ids]
    n = len(d); m = sum(d) / n
    se_paired = (sum((x - m) ** 2 for x in d) / (n - 1)) ** 0.5 / n ** 0.5
    p_lo = sum(key(ans_lo[u]) for u in ids) / n; p_hi = sum(key(ans_hi[u]) for u in ids) / n
    se_unpaired = (p_lo * (1 - p_lo) / n + p_hi * (1 - p_hi) / n) ** 0.5
    return {"delta": round(m, 4), "se_paired": round(se_paired, 4), "se_unpaired": round(se_unpaired, 4), "p_lo": round(p_lo, 4), "p_hi": round(p_hi, 4), "n": n}


def main(n_per_cell=25):
    t0 = time.time()
    df = load(); B = df[df.split == "B"]
    con = connect(":memory:")
    pack = load_pack(ROOT / "domains/ev_charging_seoul.yaml")
    v1, _ = build_world(con, pack, "seoul v1")
    rt = Router(con=con, run_id="v6", budget_usd=1.0)
    spec = {"cells": [{"filter": f"{SCOPE} AND age_band = '{b}' AND housing_type {'=' if hg == '아파트' else '<>'} '아파트'", "n": n_per_cell, "label": f"{b} × {hg}"}
                      for b in BANDS for hg in ("아파트", "비아파트")],
            "questions": [{"id": "q", "text": Q, "answer_schema": "choice", "options": OPTIONS}], "seed": 41, "temperature": 0.8}

    def answers(run):
        return {u: OPT2KEY[json.loads(a)["value"]] for u, a in con.execute("SELECT uuid, answer FROM sim_response WHERE run_id=?", [run["run_id"]]).fetchall()}

    base = run_sim(con, v1, spec, rt); ans0 = answers(base)
    cells = {}
    for u, a in con.execute("SELECT uuid, answer FROM sim_response WHERE run_id=?", [base["run_id"]]).fetchall():
        cells.setdefault(spec["cells"][json.loads(a)["cell"]]["label"], []).append(u)
    uuids = list(ans0); cw = {lab: len(us) / len(uuids) for lab, us in cells.items()}
    print(f"baseline n={len(uuids)} any={sum(v != 'none' for v in ans0.values())/len(uuids):.3f} ${base['cost_usd']:.3f}", flush=True)
    fixed = {"uuids": uuids, "n": len(uuids), "label": "all"}
    R = {"ts": datetime.now().isoformat(timespec="seconds"), "n": len(uuids), "baseline": {"run_id": base["run_id"], "p_any": round(sum(v != "none" for v in ans0.values()) / len(uuids), 4)}, "axes": {}}
    for axis, (lo, hi, _, _) in AXES.items():
        res = {}
        for lvl, val in (("lo", lo), ("hi", hi)):
            _, f = ops.run(con, v1, "fork", {"name": f"v6 {axis}={val}"}); w = f["new_world_id"]
            ops.run(con, w, "assume", {"attr": axis, "uuids": uuids, "value": val})
            r = run_sim(con, w, {**spec, "cells": [fixed]}, rt); res[lvl] = answers(r)
        sim = {"any": paired(res["lo"], res["hi"], lambda v: float(v != "none")), "ev": paired(res["lo"], res["hi"], lambda v: float(v == "ev"))}
        real = real_delta(B, axis, cw)
        R["axes"][axis] = {"lo": lo, "hi": hi, "sim": sim, "real": real,
                           "ratio_any": round(sim["any"]["delta"] / real["any"], 2) if real["any"] else None,
                           "direction_match_any": bool((sim["any"]["delta"] > 0) == (real["any"] > 0)) if real["any"] is not None else None}
        print(axis, "sim Δany", sim["any"]["delta"], "±", sim["any"]["se_paired"], "| real Δany", real["any"], "| ratio", R["axes"][axis]["ratio_any"], flush=True)
    # 사전 등록 판정 (문서 §8)
    big = {a: x for a, x in R["axes"].items() if x["real"]["any"] is not None and abs(x["real"]["any"]) >= 0.05}
    small = {a: x for a, x in R["axes"].items() if x["real"]["any"] is not None and abs(x["real"]["any"]) < 0.05}
    ratios = sorted(x["ratio_any"] for x in big.values())
    med = ratios[len(ratios) // 2] if ratios else None
    R["verdict"] = {"V6a_direction": bool(all(x["direction_match_any"] for x in big.values())) if big else None, "big_axes": list(big),
                    "V6b_magnitude": bool(0.5 <= med <= 2.0) if med is not None else None, "median_ratio": med,
                    "V6c_no_fabrication": bool(all(abs(x["sim"]["any"]["delta"]) < 0.10 for x in small.values())) if small else None, "small_axes": list(small)}
    R["cost_usd"] = round(rt.total_cost, 4); R["seconds"] = round(time.time() - t0)
    (ROOT / "data/validation_sensitivity.json").write_text(json.dumps(R, ensure_ascii=False, indent=1))
    write_doc(R)
    print("verdict", R["verdict"], f"${rt.total_cost:.3f} {R['seconds']}s")


def write_doc(R):
    p = ROOT / "docs/scenario-seoul-ev.md"; s = p.read_text(encoding="utf-8")
    v = R["verdict"]; ok = lambda b: "✅" if b else ("❌" if b is not None else "—")
    lines = [f"### 8.3 결과 (실행 {R['ts']}, 응답자 {R['n']}명 × 9회, ${R['cost_usd']}, {R['seconds']}초)", "",
             f"| 판정 | 결과 | 근거 |", "|---|---|---|",
             f"| V6a 방향 | {ok(v['V6a_direction'])} | 실측 효과 ≥ 5%p인 축 {v['big_axes']} 전부 부호 일치 |",
             f"| V6b 크기 | {ok(v['V6b_magnitude'])} | 배율 중앙값 {v['median_ratio']} (합격 0.5~2.0) |",
             f"| V6c 무효과 축에서 효과 안 만들기 | {ok(v['V6c_no_fabrication'])} | 실측 효과 < 5%p인 축 {v['small_axes']}에서 \\|Δ_sim\\| < 10%p |", "",
             "| 축 | lo → hi | 실측 Δ(친환경차 계획) | 시뮬 Δ (짝지은 SE / 비짝 SE) | 배율 | 실측 Δ(전기차) | 시뮬 Δ(전기차) |", "|---|---|---|---|---|---|---|"]
    for a, x in R["axes"].items():
        sa, se = x["sim"]["any"], x["sim"]["ev"]
        lines.append(f"| {a} | {x['lo']} → {x['hi']} | {x['real']['any']} (셀 {x['real']['any_cells_used']}) | {sa['delta']} ({sa['se_paired']} / {sa['se_unpaired']}) | {x['ratio_any']} | {x['real']['ev']} | {se['delta']} |")
    lines.append(""); lines.append(f"기준선(가정 없음) 친환경차 계획 비율 {R['baseline']['p_any']}.")
    block = "\n".join(lines) + "\n"
    if "### 8.3 결과" in s:
        s = re.sub(r"### 8\.3 결과.*?(?=\n## |\Z)", block, s, flags=re.S)
    else:
        s = s.rstrip("\n") + "\n\n" + block
    p.write_text(s, encoding="utf-8")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 25)
