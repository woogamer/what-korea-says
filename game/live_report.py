"""실제 플레이어 결과: 갤럽 18문항에서 AI만 / 플레이어만 / AI+플레이어(게임 정답판) / 나이 보정 겹침.

겹침 = Σ min(정답판%, 갤럽%) / Σ 갤럽% (게임 공개 화면과 같은 계산, 정답판 전체 칸). 플레이어 = 한국 거주라고 한 답, 장난 답(x:) 제외.
나이 보정: 나이대 비율을 성인 인구 비율(주민등록 2025 어림)에 맞춘 가중(full)과 그 제곱근(half). 나이 안 쓴 답은 가중 1.
실행(서버가 떠 있을 때): .venv/bin/python -m game.live_report [--url http://127.0.0.1:8077]
"""
import argparse
import json
import sqlite3
import urllib.request
from collections import Counter
from pathlib import Path

from game.bank import norm
from world.config import ROOT

POP = {"20s": .14, "30s": .16, "40s": .18, "50s": .20, "60+": .32}
AGE = {"30대": "30s"}   # 첫날 화면이 한국어 라벨을 값으로 보냄


def overlap(board_pct, items):
    got = Counter()
    for gi, p in board_pct:
        if gi in items:
            got[gi] += p
    return sum(min(v, items[k]) for k, v in got.items()) / sum(items.values())


def main(url, db):
    bank = json.loads((ROOT / "game" / "feud_bank.json").read_text())
    home = {q["id"]: {f"p:{a}": x for x in q["prior"] for a in [x["ko"], *x["alias_ko"]]} for q in bank["questions"]}
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    a = con.execute("SELECT COUNT(*), SUM(key LIKE 'x:%'), SUM(kr IS 0) FROM answers").fetchone()
    players = con.execute("SELECT COUNT(DISTINCT who) FROM (SELECT who FROM answers UNION SELECT who FROM guesses)").fetchone()[0]
    guesses = con.execute("SELECT COUNT(*) FROM guesses").fetchone()[0]
    ages = Counter(dict(con.execute("SELECT COALESCE(NULLIF(age,''),'-'), COUNT(DISTINCT who) FROM answers GROUP BY 1")))
    for k, v in AGE.items():
        ages[v] += ages.pop(k, 0)
    print(f"players {players} · answers {a[0]} (non-answers {a[1]}, outside Korea {a[2]}) · guesses {guesses}")
    print("players by age:", dict(sorted(ages.items())))

    rows = con.execute("SELECT qid, key, raw, age FROM answers WHERE kr IS NOT 0 AND key NOT LIKE 'x:%'").fetchall()
    res = []
    for q in bank["questions"]:
        if not q["reality"]:
            continue
        rv = json.load(urllib.request.urlopen(f"{url}/api/q/{q['id']}/reveal"))
        items = {i["label"]: i["pct"] for i in rv["gallup"]["items"]}
        n, kappa = rv["n"], rv["kappa"]
        rs = []
        for _, k, raw, age in (r for r in rows if r[0] == q["id"]):
            x = home[q["id"]].get(k)
            rs.append((f"p:{x['ko']}" if x else (k if k.startswith("n:") else f"n:{norm(raw)}"), AGE.get(age, age)))
        ag = [g for _, g in rs if g in POP]

        def weighted(power):
            w = {g: (POP[g] / (ag.count(g) / len(ag))) ** power if ag.count(g) else 1 for g in POP}
            c = Counter()
            for k, g in rs:
                c[k] += w.get(g, 1)
            scale = len(rs) / sum(c.values())
            return {k: v * scale for k, v in c.items()}

        row = {"id": q["id"], "q": q["question_en"], "n": n, "ai": rv["gallup"]["overlap_ai"], "board": rv["gallup"]["overlap_now"]}
        for name, power in [("players", 0), ("age_full", 1), ("age_half", .5)]:
            c = weighted(power)
            row[name] = overlap([(s["gallup_item"], 100 * c.get(s["key"], 0) / n) for s in rv["board"]], items)
            row[name + "_mix"] = overlap([(s["gallup_item"], (kappa * s["ai"] + 100 * c.get(s["key"], 0)) / (kappa + n)) for s in rv["board"]], items)
        res.append(row)

    avg = lambda k: sum(r[k] for r in res) / len(res)
    print(f"\n{len(res)} Gallup questions · players per question {min(r['n'] for r in res)}–{max(r['n'] for r in res)} (mean {avg('n'):.1f})")
    print(f"AI only {avg('ai'):.3f} · players only {avg('players'):.3f} · AI + players (live board) {avg('board'):.3f}")
    print(f"live board closer to Gallup than AI alone on {sum(r['board'] > r['ai'] for r in res)}/{len(res)}")
    for k in ["age_full", "age_half"]:
        print(f"{k}: AI + players {avg(k + '_mix'):.3f} (unweighted recomputed {avg('players_mix'):.3f}), worse on {sum(r[k + '_mix'] < r['players_mix'] - 1e-9 for r in res)}/{len(res)} · players only {avg(k):.3f}")
    print()
    for r in sorted(res, key=lambda r: r["board"] - r["ai"]):
        print(f"{r['ai']:.2f} → {r['board']:.2f}  players {r['players']:.2f}  n={r['n']:<3} {r['q']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8077")
    ap.add_argument("--db", default=str(ROOT / "data" / "feud_game.db"))
    a = ap.parse_args()
    main(a.url, a.db)
