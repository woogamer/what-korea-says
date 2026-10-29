"""게임 엔진: 문제 은행(game/feud_bank.json) + 플레이어 답 저장(SQLite) + 발전하는 정답판.

정답판 θ_k = (κ·p_k + c_k) / (κ + n)   (docs/feud-evolving-model.md 층2)
  p_k = AI 사전(직접 추정) 비율, c_k = 칸 k에 배정된 플레이어 답 수, n = 전체 플레이어 답 수, κ = 은행에서 추정(≈8)
칸 키: AI 칸 "p:<한국어 답>"(은행을 다시 빌드해도 유지, 묶음이 바뀌면 별칭으로 새 대표 칸을 찾음), 플레이어가 새로 만든 칸 "n:<정규화 문자열>" (MIN_NEW명 이상이어야 정답판에 오름)
답 → 칸 배정: 저장된 배정 → 글자 일치 → Super 판정(bank.same_slot). 맞힌 배정만 저장해 같은 답은 다시 묻지 않음.
"""
import json
import re
import sqlite3
import time
from pathlib import Path

from pydantic import BaseModel, Field

from game.bank import norm, same_slot
from llm.router import BudgetExceeded

BOARD = 8
MIN_NEW = 2      # 플레이어가 만든 칸은 2명 이상 같은 답이어야 정답판 후보(장난 한 명이 칸을 못 만들게)
MENU = 25        # 내 답 배정 시 비교할 후보 칸 수(사후 비율 상위)
MAX_LEN = 40
NOTES = json.loads((Path(__file__).parent / "judge_notes.json").read_text())   # 문항별 판정 규칙(실제 플레이어 답을 보고 씀)
JUNK = -2


class Verdict(BaseModel):
    why: str = Field(description="one short sentence: what the typed answer means, and which slot (if any) means exactly that")
    slot: int = Field(description="matching slot number; -1 = a real answer that matches no slot; -2 = not a real answer to this question")


class Game:
    def __init__(self, bank_path: Path, db_path: Path, router):
        bank = json.loads(Path(bank_path).read_text())
        self.kappa = bank["kappa"]
        self.qs = {q["id"]: q for q in bank["questions"]}
        # 'p:<한국어>' → AI 칸. 별칭도 대표 칸으로(재빌드로 묶음이 바뀌어도 저장된 답이 맞는 칸에 들어가게)
        self.home = {qid: {f"p:{a}": x for x in q["prior"] for a in [x["ko"], *x["alias_ko"]]} for qid, q in self.qs.items()}
        self.router = router
        self.db = sqlite3.connect(db_path, check_same_thread=False, timeout=15)   # 다른 쓰기가 잠깐 잡고 있어도 기다림
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS answers (qid TEXT, raw TEXT, key TEXT, age TEXT, sex TEXT, kr INTEGER, ts REAL);
            CREATE TABLE IF NOT EXISTS resolved (qid TEXT, text TEXT, key TEXT, PRIMARY KEY (qid, text));
            CREATE TABLE IF NOT EXISTS gallup_map (qid TEXT, key TEXT, item TEXT, PRIMARY KEY (qid, key));
            CREATE TABLE IF NOT EXISTS guesses (qid TEXT, raw TEXT, key TEXT, rank INTEGER, nth INTEGER, own INTEGER, who TEXT, ts REAL);
        """)
        if "who" not in {r[1] for r in self.db.execute("PRAGMA table_info(answers)")}:
            self.db.execute("ALTER TABLE answers ADD COLUMN who TEXT")   # 플레이어 IP 해시: 한 사람 한 문항 한 표

    # ---------- 정답판 ----------
    def _counts(self, qid):
        # 정답판은 '한국인의 답' → 한국 밖이라고 답한 플레이어(kr=0)·답이 아닌 것(x:)은 저장만 하고 세지 않음
        rows = self.db.execute("SELECT key, COUNT(*), MIN(raw) FROM answers WHERE qid=? AND kr IS NOT 0 AND key NOT LIKE 'x:%' GROUP BY key", (qid,)).fetchall()
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
    async def judge(self, qid, menu, text, careful=False):
        """text가 menu 중 몇 번 칸과 같은 답인지(-1 새 답, -2 답 아님). 숫자 답은 숫자가 같은 칸만(AI가 칸 번호와 헷갈림)."""
        digits = set(re.findall(r"\d+", text))
        if digits:
            return next((i for i, s in enumerate(menu) if set(re.findall(r"\d+", s["ko"])) == digits), -1)
        q = self.qs[qid]
        listing = "\n".join(f"{i}: {s['label']} / {s['ko']}" for i, s in enumerate(menu))
        note = NOTES.get(qid)
        r, _ = await self.router.acall("extract", [{"role": "user", "content":
            f"Survey question: {q['question_en']} / {q['question_ko']}\n" + (f"Rules for this question: {note}\n" if note else "") +
            f"Slots:\n{listing}\nPlayer typed: \"{text}\"\n"
            "Return the slot number only if the typed answer means the same specific thing as that slot (translation, spelling variant, typo, "
            "nickname, old name, extra adjective like 'snowy winter' = 'winter'). A different item of the same kind is NOT the same "
            "(수제비 is not 칼국수, 필라테스 is not 헬스). A broader word that could cover several slots is NOT the same (국물음식 is not 라면), "
            "but a short keyword that points to exactly one slot's topic is that slot (결혼 → 결혼은 언제 할 거야?, 유튜브 → 스마트폰). "
            "If two things are named, use the first. Return -1 if it is a sincere answer that matches no slot. "
            "Return -2 only if it is clearly not an answer to this question: a person's name (e.g. 'seungyeon', 'Yeon', '홍승연', 'sy'), "
            "test or random text, gibberish, or an answer to a different question (a month for a food question). An unusual but sincere answer is -1, not -2.\n"
            "First write why (what the answer means), then slot."}],
            schema=Verdict, thinking=careful, max_tokens=4000 if careful else 250, temperature=0, retries=2, tag="game-judge")
        return r.slot if r and (0 <= r.slot < len(menu) or r.slot == JUNK) else -1

    async def resolve(self, qid, text, careful=False):
        """text와 같은 답인 칸의 key, 새 답이면 None, 답이 아니면 'x:…'. 항상 모든 칸과 비교(정답판 8칸만 주면 AI가 가장 비슷한 칸에 억지로 붙임).
        careful = 정답판에 들어갈 내 답: 추론 켜고 판정해 저장. 추측은 빠르게(추론 끔), 저장 안 함(급한 판정이 내 답 배정을 오염시키지 않게)."""
        t = norm(text)
        if not t:
            return None
        menu = self.slots(qid)[0][:MENU]
        keys = {s["key"] for s in menu}
        hit = self.db.execute("SELECT key FROM resolved WHERE qid=? AND text=?", (qid, t)).fetchone()
        if hit and (hit[0] in keys or hit[0].startswith("x:")):
            return hit[0]
        for s in menu:
            if any(t == norm(p) for a in s["aliases"] for p in [a, *a.split("/")]):
                return s["key"]
        try:
            i = await self.judge(qid, menu, text, careful)
        except BudgetExceeded:
            return None   # 예산 초과 시 글자 일치만
        if i == -1:
            return None
        key = f"x:{t}" if i == JUNK else menu[i]["key"]
        if not careful:
            return key
        self.db.execute("INSERT OR REPLACE INTO resolved VALUES (?,?,?)", (qid, t, key))
        self.db.commit()
        return key

    async def answer(self, qid, text, age=None, sex=None, kr=None, who=None):
        """플레이어 자기 답 저장. 기존 칸과 같으면 그 칸, 아니면 새 칸 후보. 같은 사람(who)이 이 문항에 이미 답했으면 그 답만 돌려줌."""
        if who:
            hit = self.db.execute("SELECT key FROM answers WHERE qid=? AND who=? LIMIT 1", (qid, who)).fetchone()
            if hit:
                return hit[0]
        text = text.strip()[:MAX_LEN]
        key = await self.resolve(qid, text, careful=True) or f"n:{norm(text)}"
        self.db.execute("INSERT INTO answers VALUES (?,?,?,?,?,?,?,?)", (qid, text, key, age, sex, kr, time.time(), who))
        self.db.commit()
        return key

    async def guess(self, qid, text, nth=None, own=None, who=None):
        b, _ = self.board(qid)
        text = text.strip()[:MAX_LEN]
        key = await self.resolve(qid, text)
        rank = next((i for i, s in enumerate(b) if s["key"] == key), -1)
        # 추측 순서 = '남들이 많이 할 것 같은 답'(기록만, 정답판 계산엔 안 씀)
        try:
            self.db.execute("INSERT INTO guesses VALUES (?,?,?,?,?,?,?,?)", (qid, text, key, rank, nth, own, who, time.time()))
            self.db.commit()
        except sqlite3.Error:
            self.db.rollback()   # 기록 실패가 게임을 막지 않게
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
            slots.append({"key": s["key"], "label": s["label"], "ko": s["ko"], "now": round(s["now"], 1), "ai": round(s["ai"], 1), "count": s["count"],
                          "gallup": item and (real[item]["pct"] if item in real else "n/a"), "gallup_item": item and (real[item]["en"] if item in real else item), "gallup_item_ko": item})
        out = {"id": qid, "q": q["question_en"], "q_ko": q["question_ko"], "n": n, "ai_weight": round(self.ai_weight(n), 3), "kappa": self.kappa, "board": slots}
        if mine:
            ss, _ = self.slots(qid)
            s = next((s for s in ss if s["key"] == mine), None)
            rank = next((i for i, x in enumerate(b) if x["key"] == mine), -1)
            raw = self.db.execute("SELECT raw FROM answers WHERE qid=? AND key=? LIMIT 1", (qid, mine)).fetchone()
            out["mine"] = {"label": s["label"] if s else (raw[0] if raw else None), "ko": s["ko"] if s else (raw[0] if raw else None), "rank": rank, "count": s["count"] if s else 0}
        if q["reality"]:
            tot = sum(x["pct"] for x in q["reality"]["items"])

            def overlap(col):
                got = {}
                for s in slots:
                    if s["gallup_item"] and isinstance(s["gallup"], (int, float)):
                        got[s["gallup_item"]] = got.get(s["gallup_item"], 0) + s[col]
                return round(sum(min(v, next(x["pct"] for x in q["reality"]["items"] if x["en"] == k)) for k, v in got.items()) / tot, 3)
            on_board = {s["gallup_item"] for s in slots}
            out["gallup"] = {**q["source"], "items": [{"label": x["en"], "ko": x["ko"], "pct": x["pct"], "on_board": x["en"] in on_board} for x in q["reality"]["items"]],
                             "overlap_ai": overlap("ai"), "overlap_now": overlap("now")}
        return out
