"""C7: case 의 실행 기록(툴 실행·산출물·판단·시뮬)을 사람이 따라갈 수 있는 HTML 보고서로. 수치는 DB 기록에서만."""
import html
import json
from pathlib import Path

from world.grades import GRADE_RANK

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs/site/report"


def _claims(x, path=""):
    if isinstance(x, dict):
        if "text" in x and "grade" in x and isinstance(x["text"], str):
            yield path, x
        for k, v in x.items():
            yield from _claims(v, f"{path}.{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from _claims(v, f"{path}[{i}]")


def load_runs(con, case_name: str) -> list[dict]:
    runs = con.execute("SELECT run_id, world_id, tool, status, ts FROM tool_run WHERE case_name=? ORDER BY ts", [case_name]).fetchall()
    out = []
    for run_id, wid, tool, status, ts in runs:
        arts = con.execute("SELECT artifact_id, step, content, grade_summary FROM artifact WHERE op_id=? ORDER BY ts, rowid", [run_id]).fetchall()
        decs = con.execute("SELECT d.decision_id, d.gate, d.chosen, d.rationale FROM decision d JOIN artifact a USING (artifact_id) WHERE a.op_id=? ORDER BY d.ts", [run_id]).fetchall()
        out.append({"run_id": run_id, "world_id": wid, "tool": tool, "status": status, "ts": str(ts),
                    "artifacts": [{"id": a, "step": s, "content": json.loads(c), "grade_summary": json.loads(g)} for a, s, c, g in arts],
                    "decisions": [{"id": d, "gate": g, "chosen": json.loads(c), "rationale": r} for d, g, c, r in decs]})
    return out


def run_cards(r: dict) -> list[dict]:
    """툴 실행 안의 sim 단계가 만든 보정 카드들."""
    return [c for a in r["artifacts"] for c in a["content"].get("calibration", []) if isinstance(c, dict)]


def claim_lock(c: dict, cards: list[dict]) -> str | None:
    """Only server-grounded numbers with matching independent evaluation may be unlocked."""
    lv = int(c.get("level", 0) or 0)
    if lv == 0:
        return None  # displayed as an unvalidated hypothesis, never as validated direction
    if lv >= 2:
        return "잠김: 조건 감도·개인 예측은 검증되지 않음"
    if not cards:
        return "잠김: 이 실행에 보정 카드 없음(실측 열 없음)"
    if not c.get("statistics"):
        return "잠김: 검증된 집계 참조 없음"
    for stat in c["statistics"]:
        card = next((x for x in cards if x.get("card_id") == stat.get("card_id")
                     and x.get("question_id") == stat.get("question_id")
                     and x.get("run_id") == stat.get("run_id") and x.get("schema_version") == 2), None)
        if not card or card.get("evaluation_kind") != "independent_holdout":
            return "잠김: 해당 질문의 독립 평가 없음"
        cell = next((x for x in card.get("cells", []) if x["label"] == stat.get("cell")), None)
        if not cell or not cell.get("level1_open"):
            return "잠김: 해당 셀의 M1 기준 미달 또는 평가 표본 부족"
        expected = cell.get("calibrated", {}).get(stat.get("answer"))
        if (stat.get("source") != "calibrated" or stat.get("metric") != "share"
                or expected is None or stat.get("value") != expected):
            return "잠김: 보정 집계와 주장 수치 불일치"
    return None


def verify_claims(con, case) -> dict:
    runs = load_runs(con, case.name)
    n, n_locked, n_level1, problems = 0, 0, 0, []
    for r in runs:
        cards = run_cards(r)
        for a in r["artifacts"]:
            for path, c in _claims(a["content"].get("output", a["content"])):
                n += 1
                if c["grade"] not in GRADE_RANK:
                    problems.append(f"{r['tool']}/{a['step']}{path}: 등급 '{c['grade']}' 아님")
                if not c.get("provenance"):
                    problems.append(f"{r['tool']}/{a['step']}{path}: 출처 없음")
                if int(c.get("level", 0) or 0) >= 2:
                    problems.append(f"{r['tool']}/{a['step']}{path}: level {c['level']} 주장은 생성 금지")
                if int(c.get("level", 0) or 0) == 1:
                    n_level1 += 1
                    if claim_lock(c, cards):
                        n_locked += 1
    if not runs:
        problems.append("실행 기록 없음")
    return {"passed": not problems, "problems": problems[:10], "n_claims": n, "n_runs": len(runs), "n_level1": n_level1, "n_locked": n_locked}


def _esc(x):
    return html.escape(str(x))


def _table(rows: list[dict], cols=None, raw_cols=frozenset()) -> str:
    if not rows:
        return "<p>(없음)</p>"
    cols = cols or list(rows[0])
    h = "".join(f"<th>{_esc(c)}</th>" for c in cols)
    b = "".join("<tr>" + "".join(f"<td>{r.get(c, '') if c in raw_cols else _esc(r.get(c, ''))}</td>" for c in cols) + "</tr>" for r in rows)
    return f"<table><tr>{h}</tr>{b}</table>"


def _agent_html(body: dict) -> str:
    rows = [{"열": c["attr"], "판정": ("✅ 통과" if c["passed"] else "❌ 불합격") + (f" · {c.get('evidence_level')}" if c.get("evidence_level") else ""),
             "커버리지": c["coverage"], "출처": f"{c['url']} ({c['year']})" if c["url"] else "-", "행": c["rows"],
             "문제": "; ".join(c["problems"]) or "-", "경고": "; ".join(c.get("warnings", [])) or "-", "모순": "; ".join(c.get("conflicts", [])) or "-"} for c in body.get("summary", [])]
    out = "<h4>후보 (게이트에 올라간 것)</h4>" + _table(rows)
    for c in body.get("candidates", []):
        tr = [{"회전": t["round"], "행동": t["action"], "내용": (", ".join(t.get("queries", [])) if t["action"] == "search" else
              f"found={t.get('found')} rows={t.get('n_rows')} url={t.get('url', '')[:60]} 커버리지={t.get('coverage')} 문제={'; '.join(t.get('problems', [])) or '-'}")} for t in c.get("trace", [])]
        out += f"<h4>추적 · {_esc(c['attr'])}</h4>" + _table(tr)
        t = c.get("table", {})
        if t.get("rows"):
            out += _table([{"축": json.dumps(r["by"], ensure_ascii=False), "원문 표기": r.get("source_label", ""), "비율": json.dumps(r["p"], ensure_ascii=False)} for r in t["rows"]])
            out += f"<p><small>표본 조건: {_esc(t.get('sample_condition') or '-')} · 매핑 note: {_esc(t.get('note') or '-')}</small></p>"
            if t.get("quote"):
                out += f"<pre>{_esc(t['quote'][:800])}</pre>"
    return out


def _l0_card_html(card: dict) -> str:
    if card.get("status") == "no_reference":
        return "<p class=lock>L0 정합성 카드 없음: 팩에 reference_distributions 없음</p>"
    rows = [{"축": a["col"], "TVD": a["tvd"], "판정": "✅" if a["passed"] else "❌", "가장 어긋난 값": a["worst_value"],
             "페르소나": f"{a['l0_share']:.1%}", "참조": f"{a['ref_share']:.1%}"} for a in card["axes"]]
    out = f"<h4>L0 정합성 카드 <small>참조 {_esc(card.get('source_ref', ''))} · 기준 TVD &lt; {card['max_tvd']}</small></h4>" + _table(rows)
    if card.get("reweight_suggestion"):
        out += f"<p><small>어긋난 축은 reweight(레이킹) 후보: {_esc(', '.join(card['reweight_suggestion']))}</small></p>"
    return out


def _card_html(card: dict) -> str:
    """보정 카드: 시뮬 분포가 실측에서 얼마나 떨어졌고(M1 전), 비율·혼합 보정 뒤 얼마나 남았나(M1 후). 수준 주장 잠금의 근거."""
    head = f"<h4>보정 카드 · 질문 {_esc(card.get('question_id'))} ↔ 실측 열 {_esc(card.get('observed_attr'))}</h4>"
    if card.get("status") != "ok":
        return head + f"<p class=lock>보정 없음: {_esc(card.get('reason', ''))}. 이 실행의 수준 주장은 잠김.</p>"
    st = "전체 셀 통과" if card.get("level1_open") and card.get("evaluation_kind") == "independent_holdout" else "일부 또는 전체 셀 잠김"
    evaluation = "독립 평가 오차" if card.get("evaluation_kind") == "independent_holdout" else "적합 오차 · 독립 검증 아님"
    head += f"<p><b>{evaluation}</b> · 학습 {card.get('n_fit', card.get('n_observed', 0))} / 평가 {card.get('n_evaluation', 0)}명</p>"
    summ = (f"<p>실측 {card['n_observed']}명(값 등급 {_esc(', '.join(card.get('value_grades') or [card.get('observed_grade') or '']))}) · M1 보정 전 <b>{card['m1_before']}</b> → 보정 후 <b>{card['m1_after']}</b> "
            f"(기준 &lt; {card['level1_max_m1']}) → 수준 주장 <b>{st}</b><br>"
            + (f"실측만 쓴 기준선 M1: 셀 직접 {card['baselines_m1']['direct']} · 전체 평균 {card['baselines_m1']['overall_mean']}"
               + (f" · 회귀 {card['baselines_m1']['regression']}" if "regression" in card["baselines_m1"] else "")
               + f" → 합성 보정이 <b>{'이기거나 같음' if card.get('beats_baselines') else '짐(잠금)'}</b><br>" if card.get("baselines_m1") else "")
            + (f"추정: 합성 사전 + 실측 베이지안 갱신, 사전 강도 α={card['eb_alpha']:g}(실측 {card['eb_alpha']:g}명 몫)<br>" if card.get("eb_alpha") is not None else "")
            + f"<small>비율 가중치 {_esc(json.dumps(card['weights'], ensure_ascii=False))} · λ={card['lam']} · {_esc(card['note'])}</small></p>")
    rows = [{"셀": x["label"], "n 시뮬/실측": f"{x['n_sim']}/{x['n_obs']}", "시뮬": json.dumps(x["sim"], ensure_ascii=False),
             "합성 사전": json.dumps(x.get("prior"), ensure_ascii=False), "보정 후": json.dumps(x.get("calibrated"), ensure_ascii=False), "보정용 실측": json.dumps(x["observed"], ensure_ascii=False), "평가용 실측": json.dumps(x.get("evaluation_observed"), ensure_ascii=False),
             "수치 검증": "통과" if x.get("level1_open") and card.get("evaluation_kind") == "independent_holdout" else "잠김",
             "TVD 전→후": f"{x.get('tvd_before', '-')} → {x.get('tvd_after', '-')}" if x.get("scored") else "표본 부족"} for x in card["cells"]]
    return head + summ + _table(rows)


def render_case_report(con, case, out_dir: Path = OUT) -> str:
    runs = load_runs(con, case.name)
    parts = [f"<h1>{_esc(case.title)}</h1>",
             f"<p><b>의사결정</b> {_esc(case.decision)}</p>",
             "<h2>1. 선택지와 조사 질문</h2>", _table([o.model_dump() for o in case.options]),
             _table([{"id": r.id, "질문": r.text, "결과→결정": "; ".join(f"{i.if_} → {i.then}" for i in r.implications)} for r in case.research_questions])]
    for r in runs:
        parts.append(f"<h2>{_esc(r['tool'])} <small>{_esc(r['run_id'])} · {r['status']} · 월드 {_esc(r['world_id'])}</small></h2>")
        for a in r["artifacts"]:
            c = a["content"]
            gs = " ".join(f"{k}×{v}" for k, v in a["grade_summary"].items())
            parts.append(f"<h3>{_esc(a['step'])} <small>{_esc(gs)}</small></h3>")
            if "shown" in c:
                d = next((x for x in r["decisions"] if x["gate"] == a["step"]), None)
                if d:
                    parts.append(f"<blockquote><b>★ 판단 {_esc(d['id'])}</b> {_esc(json.dumps(d['chosen'], ensure_ascii=False)[:400])}<br>이유: {_esc(d['rationale'])}</blockquote>")
                continue
            body = c.get("output", {k: v for k, v in c.items() if k not in ("prompt_snapshot", "inputs", "provenance")})
            cards = run_cards(r)
            claims = []
            for _, x in _claims(body):
                lock = claim_lock(x, cards)
                lvl = {0: "미검증 해석·가설", 1: "집계 수치"}.get(int(x.get("level", 0) or 0), str(x.get("level")))
                claims.append({"주장": _esc(x["text"]) if not lock else f"<s>{_esc(x['text'])}</s> <span class=lock>{_esc(lock)}</span>", "수준": lvl + (" 🔒" if lock else ""),
                               "등급": x["grade"], "출처": ", ".join(map(str, x.get("provenance", [])))})
            if claims:
                parts.append(_table(claims, raw_cols={"주장"}))
            for card in (c.get("calibration") or []):
                parts.append(_card_html(card))
            if "cells" in body and isinstance(body["cells"], list):
                parts.append(_table(body["cells"][:20]))
            if "aggregate" in body:
                rows = [{"셀": cell, "질문": q, "n": v["n"], "찬성률": v["pos_rate"], "분포": json.dumps(v["dist"], ensure_ascii=False)}
                        for cell, qs in body["aggregate"].items() for q, v in qs.items()]
                parts.append(_table(rows))
            if r["tool"] == "evidence_agent" and "candidates" in body:            # 근거 수집 에이전트: 후보·검사·추적
                parts.append(_agent_html(body))
            elif r["tool"] == "evidence_agent" and "applied" in body:
                parts.append(_table([{"열": x["attr"], "belief": x["belief_id"], "연산": f"{x['op_id']} · {x['status']}",
                                      "검사": ", ".join(f"{c['check']} {'✅' if c['passed'] else '❌'}" for c in x.get("checks", []))} for x in body["applied"]]) if body["applied"] else "<p>승인된 후보 없음 → 월드 변경 없음</p>")
            elif r["tool"] == "evidence_agent" and "max_tvd" in body:
                parts.append(_l0_card_html(body))
            elif not claims and "cells" not in body and "aggregate" not in body:
                parts.append(f"<pre>{_esc(json.dumps(body, ensure_ascii=False, indent=1)[:3000])}</pre>")
            if "provenance" in c and c["provenance"].get("decisions"):
                parts.append(f"<p><small>상속한 판단: {_esc(c['provenance']['decisions'])}</small></p>")
    css = "<style>body{font-family:system-ui;max-width:90ch;margin:32px auto;padding:0 16px;line-height:1.5}table{border-collapse:collapse;font-size:13px;width:100%}td,th{border-bottom:1px solid #ddd;padding:4px 8px;text-align:left;vertical-align:top}h2{margin-top:36px;border-top:1px solid #ddd;padding-top:10px}blockquote{border-left:3px solid #0E6B60;margin:0;padding-left:12px}small{color:#666;font-weight:normal}.lock{color:#999}s{color:#999}h4{margin:18px 0 6px}</style>"
    doc = f'<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="market-world-validation" content="v2"><meta name="robots" content="noindex"><title>{_esc(case.title)}</title>{css}</head><body>{"".join(parts)}</body></html>'
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{case.name}.html"
    p.write_text(doc, encoding="utf-8")
    return f"/report/{case.name}.html"
