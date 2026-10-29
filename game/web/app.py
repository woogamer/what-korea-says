"""What Korea Says — 웹 서버. 실행: .venv/bin/uvicorn game.web.app:app --port 8077"""
import hashlib
import os
import re
import secrets
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from game.web.engine import Game
from llm.router import Router
from world.config import ROOT

load_dotenv(ROOT / ".env")
HERE = Path(__file__).parent
game = Game(ROOT / "game" / "feud_bank.json", Path(os.environ.get("FEUD_DB", ROOT / "data" / "feud_game.db")),
            Router(budget_usd=float(os.environ.get("FEUD_BUDGET_USD", "5"))))
app = FastAPI(title="What Korea Says")


class Text(BaseModel):
    text: str = Field(min_length=1, max_length=80)
    age: str | None = Field(None, max_length=10)
    sex: str | None = Field(None, max_length=10)
    kr: bool | None = None
    nth: int | None = Field(None, ge=1, le=50)   # 추측: 이 판에서 몇 번째
    own: bool | None = None                      # 추측: 내 답을 첫 추측으로 넣은 것


def q_or_404(qid):
    if qid not in game.qs:
        raise HTTPException(404, "no such question")
    return game.qs[qid]


COOKIE = "wks"


@app.middleware("http")
async def browser_id(request: Request, call_next):
    # 브라우저마다 쿠키 하나 → 같은 와이파이 가족도 각자 한 표(전엔 IP 하나에 한 표)
    resp = await call_next(request)
    if not re.fullmatch(r"[0-9a-f]{32}", request.cookies.get(COOKIE, "")):
        resp.set_cookie(COOKIE, secrets.token_hex(16), max_age=400 * 86400, httponly=True, secure=True, samesite="lax")
    return resp


def player(request):
    """(플레이어, IP 해시). 플레이어 = 쿠키 해시, 쿠키가 없으면(첫 요청·쿠키 막힘) IP 해시. 원문 IP·쿠키는 저장 안 함."""
    ip = request.headers.get("cf-connecting-ip") or (request.client and request.client.host) or ""
    iph = hashlib.sha256(f"wks|{ip}".encode()).hexdigest()[:16] if ip else None
    c = request.cookies.get(COOKIE, "")
    return (hashlib.sha256(f"wks|c|{c}".encode()).hexdigest()[:16] if re.fullmatch(r"[0-9a-f]{32}", c) else iph), iph


@app.get("/api/questions")
def questions(request: Request):
    # done = 이 플레이어가 이미 답한 문제 → 브라우저 기록이 지워져도 같은 문제를 다시 안 냄.
    # IP로 기록된 답(쿠키 전·쿠키 받기 직전 요청)도 포함 → 원래 하던 사람이 쿠키를 받아도 같은 문제가 다시 안 나옴
    who, iph = player(request)
    done = {r[0] for r in game.db.execute("SELECT DISTINCT qid FROM answers WHERE who IN (?, ?)", (who, iph))} if who else set()
    return [{"id": q["id"], "q": q["question_en"], "q_ko": q["question_ko"], "gallup": bool(q["reality"]), "done": q["id"] in done} for q in game.qs.values()]


@app.get("/api/q/{qid}")
def question(qid: str):
    q = q_or_404(qid)
    b, n = game.board(qid)
    return {"id": qid, "q": q["question_en"], "q_ko": q["question_ko"], "slots": len(b), "n": n, "ai_weight": round(game.ai_weight(n), 3), "gallup": bool(q["reality"])}


@app.post("/api/q/{qid}/answer")
async def answer(qid: str, body: Text, request: Request):
    q_or_404(qid)
    who, iph = player(request)
    return {"key": await game.answer(qid, body.text, body.age, body.sex, body.kr, who, iph)}


@app.post("/api/q/{qid}/guess")
async def guess(qid: str, body: Text, request: Request):
    q_or_404(qid)
    rank, s = await game.guess(qid, body.text, body.nth, body.own, player(request)[0])
    return {"rank": rank, "key": s and s["key"], "label": s and s["label"], "ko": s and s["ko"], "now": s and round(s["now"], 1)}


@app.get("/api/q/{qid}/reveal")
async def reveal(qid: str, mine: str | None = None):
    q_or_404(qid)
    return await game.reveal(qid, mine)


@app.get("/")
def index():
    # 카톡 인앱 브라우저가 옛 페이지를 붙잡고 있어서(한국어 버튼 없음) 매번 새로 확인하게
    return FileResponse(HERE / "static" / "index.html", headers={"Cache-Control": "no-cache"})
