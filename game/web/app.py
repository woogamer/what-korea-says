"""What Korea Says — 웹 서버. 실행: .venv/bin/uvicorn game.web.app:app --port 8077"""
import hashlib
import os
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


def player(request):
    ip = request.headers.get("cf-connecting-ip") or (request.client and request.client.host) or ""
    return hashlib.sha256(f"wks|{ip}".encode()).hexdigest()[:16] if ip else None   # IP 원문은 저장 안 함


@app.get("/api/questions")
def questions():
    return [{"id": q["id"], "q": q["question_en"], "q_ko": q["question_ko"], "gallup": bool(q["reality"])} for q in game.qs.values()]


@app.get("/api/q/{qid}")
def question(qid: str):
    q = q_or_404(qid)
    b, n = game.board(qid)
    return {"id": qid, "q": q["question_en"], "q_ko": q["question_ko"], "slots": len(b), "n": n, "ai_weight": round(game.ai_weight(n), 3), "gallup": bool(q["reality"])}


@app.post("/api/q/{qid}/answer")
async def answer(qid: str, body: Text, request: Request):
    q_or_404(qid)
    who = player(request)
    return {"key": await game.answer(qid, body.text, body.age, body.sex, body.kr, who)}


@app.post("/api/q/{qid}/guess")
async def guess(qid: str, body: Text, request: Request):
    q_or_404(qid)
    rank, s = await game.guess(qid, body.text, body.nth, body.own, player(request))
    return {"rank": rank, "key": s and s["key"], "label": s and s["label"], "ko": s and s["ko"], "now": s and round(s["now"], 1)}


@app.get("/api/q/{qid}/reveal")
async def reveal(qid: str, mine: str | None = None):
    q_or_404(qid)
    return await game.reveal(qid, mine)


@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")
