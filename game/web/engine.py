"""게임 엔진: 문제 은행(game/feud_bank.json) + 플레이어 답 저장(SQLite) + 발전하는 정답판.

정답판 θ_k = (κ·p_k + c_k) / (κ + n)   (docs/feud-evolving-model.md 층2)
  p_k = AI 사전(직접 추정) 비율, c_k = 칸 k에 배정된 플레이어 답 수, n = 전체 플레이어 답 수, κ = 은행에서 추정(≈8)
칸 키: AI 칸 "p:<한국어 답>"(은행을 다시 빌드해도 유지, 묶음이 바뀌면 별칭으로 새 대표 칸을 찾음), 플레이어가 새로 만든 칸 "n:<정규화 문자열>" (MIN_NEW명 이상이어야 정답판에 오름)
답 → 칸 배정: 저장된 배정 → 글자 일치 → Super 판정(bank.same_slot). 맞힌 배정만 저장해 같은 답은 다시 묻지 않음.
"""
import json
import sqlite3
import time
from pathlib import Path

from game.bank import norm, same_slot
from llm.router import BudgetExceeded

BOARD = 8
MIN_NEW = 2      # 플레이어가 만든 칸은 2명 이상 같은 답이어야 정답판 후보(장난 한 명이 칸을 못 만들게)
MENU = 25        # 내 답 배정 시 비교할 후보 칸 수(사후 비율 상위)
MAX_LEN = 40


class Game:
    def __init__(self, bank_path: Path, db_path: Path, router):
        bank = json.loads(Path(bank_path).read_text())
        self.kappa = bank["kappa"]
        self.qs = {q["id"]: q for q in bank["questions"]}
        # 'p:<한국어>' → AI 칸. 별칭도 대표 칸으로(재빌드로 묶음이 바뀌어도 저장된 답이 맞는 칸에 들어가게)
        self.home = {qid: {f"p:{a}": x for x in q["prior"] for a in [x["ko"], *x["alias_ko"]]} for qid, q in self.qs.items()}
        self.router = router
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS answers (qid TEXT, raw TEXT, key TEXT, age TEXT, sex TEXT, kr INTEGER, ts REAL);
            CREATE TABLE IF NOT EXISTS resolved (qid TEXT, text TEXT, key TEXT, PRIMARY KEY (qid, text));
            CREATE TABLE IF NOT EXISTS gallup_map (qid TEXT, key TEXT, item TEXT, PRIMARY KEY (qid, key));
        """)

    # ---------- 정답판 ----------
    def _counts(self, qid):
        # 정답판은 '한국인의 답' → 한국 밖이라고 답한 플레이어(kr=0)는 저장만 하고 세지 않음
        rows = self.db.execute("SELECT key, COUNT(*), MIN(raw) FROM answers WHERE qid=? AND kr IS NOT 0 GROUP BY key", (qid,)).fetchall()
        cnt, raw = {}, {}
        for k, c, r in rows:
            x = self.home[qid].get(k)
            k = f"p:{x['ko']}" if x else (k if k.startswith("n:") else f"n:{norm(r)}")   # 은행에서 사라진 AI 칸 답은 플레이어 칸으로
            cnt[k] = cnt.get(k, 0) + c
            raw.setdefault(k, r)
        return cnt, raw

    def slots(self, qid):
        """모든 칸(AI 칸 + 플레이어 칸)의 사후 비율, 큰 순."""
        q = self.qs[qid]
        cnt, raw = self._counts(qid)
        n = sum(cnt.values())
        tot = sum(x["pct"] for x in q["prior"])
        out = []
        for x in q["prior"]:
            k = f"p:{x['ko']}"
            p = x["pct"] / tot
            out.append({"key": k, "label": x["en"], "ko": x["ko"], "aliases": [x["en"], x["ko"], *x["alias_ko"]],
                        "ai": 100 * p, "count": cnt.get(k, 0), "now": 100 * (self.kappa * p + cnt.get(k, 0)) / (self.kappa + n)})
        for k, c in cnt.items():
            if k.startswith("n:"):
                out.append({"key": k, "label": raw[k], "ko": raw[k], "aliases": [raw[k]], "ai": 0.0, "count": c,
                            "now": 100 * c / (self.kappa + n)})
        return sorted(out, key=lambda s: -s["now"]), n

    def board(self, qid):
        q = self.qs[qid]
        ss, n = self.slots(qid)
        ok = [s for s in ss if not s["key"].startswith("n:") or s["count"] >= MIN_NEW]
        size = min(BOARD, max(q["board_size"], sum(s["now"] >= 1 for s in ok)))
        return ok[:size], n

    def ai_weight(self, n):
        return self.kappa / (self.kappa + n)

    # ---------- 답 → 칸 ----------
    async def resolve(self, qid, text, menu):
        """menu(칸 목록) 중 text와 같은 답인 칸의 key, 없으면 None."""
        t = norm(text)
        if not t:
            return None
        keys = {s["key"] for s in menu}
        hit = self.db.execute("SELECT key FROM resolved WHERE qid=? AND text=?", (qid, t)).fetchone()
        if hit and hit[0] in keys:
            return hit[0]
        for s in menu:
            if any(t == norm(a) for a in s["aliases"]):
                return s["key"]
        try:
            i = await same_slot(self.router, self.qs[qid]["question_en"], [f"{s['label']} / {s['ko']}" for s in menu], text)
        except BudgetExceeded:
            return None   # 예산 초과 시 글자 일치만
        if i < 0:
            return None
        self.db.execute("INSERT OR REPLACE INTO resolved VALUES (?,?,?)", (qid, t, menu[i]["key"]))
        self.db.commit()
        return menu[i]["key"]

    async def answer(self, qid, text, age=None, sex=None, kr=None):
        """플레이어 자기 답 저장. 기존 칸과 같으면 그 칸, 아니면 새 칸 후보."""
        text = text.strip()[:MAX_LEN]
        ss, _ = self.slots(qid)
        key = await self.resolve(qid, text, ss[:MENU]) or f"n:{norm(text)}"
        self.db.execute("INSERT INTO answers VALUES (?,?,?,?,?,?,?)", (qid, text, key, age, sex, kr, time.time()))
        self.db.commit()
        return key

    async def guess(self, qid, text):
        b, _ = self.board(qid)
        key = await self.resolve(qid, text.strip()[:MAX_LEN], b)
        rank = next((i for i, s in enumerate(b) if s["key"] == key), -1)
        return rank, (b[rank] if rank >= 0 else None)

    # ---------- 공개 ----------
    async def gallup_item(self, qid, s):
        """칸 → 갤럽 항목 이름(없으면 None). AI 칸은 은행에 배정돼 있고, 플레이어 칸은 처음 볼 때 한 번 판정해 저장."""
        q = self.qs[qid]
        if not q["reality"]:
            return None
        x = self.home[qid].get(s["key"])
        if x and "reality_ko" in x:   # 은행이 정답판 칸만 배정해 둠, 나머지는 아래에서 처음 볼 때 판정
            return x["reality_ko"]
        hit = self.db.execute("SELECT item FROM gallup_map WHERE qid=? AND key=?", (qid, s["key"])).fetchone()
        if hit:
            return hit[0] or None
        names = [x["ko"] for x in q["reality"]["items"]] + q.get("reality_unpublished", [])
        try:
            i = await same_slot(self.router, q["question_ko"], names, s["label"])
        except BudgetExceeded:
            return None
        item = names[i] if i >= 0 else ""
        self.db.execute("INSERT OR REPLACE INTO gallup_map VALUES (?,?,?)", (qid, s["key"], item))
        self.db.commit()
        return item or None

    async def reveal(self, qid, mine=None):
        q = self.qs[qid]
        b, n = self.board(qid)
        real = {x["ko"]: x for x in q["reality"]["items"]} if q["reality"] else {}
        slots = []
        for s in b:
            item = await self.gallup_item(qid, s)
            slots.append({"key": s["key"], "label": s["label"], "now": round(s["now"], 1), "ai": round(s["ai"], 1), "count": s["count"],
                          "gallup": item and (real[item]["pct"] if item in real else "n/a"), "gallup_item": item and (real[item]["en"] if item in real else item)})
        out = {"id": qid, "q": q["question_en"], "n": n, "ai_weight": round(self.ai_weight(n), 3), "kappa": self.kappa, "board": slots}
        if mine:
            ss, _ = self.slots(qid)
            s = next((s for s in ss if s["key"] == mine), None)
            rank = next((i for i, x in enumerate(b) if x["key"] == mine), -1)
            raw = self.db.execute("SELECT raw FROM answers WHERE qid=? AND key=? LIMIT 1", (qid, mine)).fetchone()
            out["mine"] = {"label": s["label"] if s else (raw[0] if raw else None), "rank": rank, "count": s["count"] if s else 0}
        if q["reality"]:
            tot = sum(x["pct"] for x in q["reality"]["items"])

            def overlap(col):
                got = {}
                for s in slots:
                    if s["gallup_item"] and isinstance(s["gallup"], (int, float)):
                        got[s["gallup_item"]] = got.get(s["gallup_item"], 0) + s[col]
                return round(sum(min(v, next(x["pct"] for x in q["reality"]["items"] if x["en"] == k)) for k, v in got.items()) / tot, 3)
            on_board = {s["gallup_item"] for s in slots}
            out["gallup"] = {**q["source"], "items": [{"label": x["en"], "pct": x["pct"], "on_board": x["en"] in on_board} for x in q["reality"]["items"]],
                             "overlap_ai": overlap("ai"), "overlap_now": overlap("now")}
        return out
