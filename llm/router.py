"""모델 라우터: 역할 → Nemotron 모델, 구조화 출력(JSON 스키마), 재시도, 호출 로그·비용.

확인된 사실(2026-09-14 Token Factory 탐색):
- response_format json_schema(strict) 동작. Nano 0.8초.
- extra_body chat_template_kwargs.enable_thinking=false → 추론 토큰 없이 24 토큰·0.4초. 응답자 호출 기본값.
- 비동기 20건 병렬 3.6초.
"""
import asyncio
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel, ValidationError

from world.store import new_id

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

BASE_URL = os.environ.get("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/")

# 가격: $ / 1M 토큰 (입력, 출력)
MODELS = {
    "nano": {"id": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B", "in": 0.05, "out": 0.20},
    "super": {"id": "nvidia/nemotron-3-super-120b-a12b", "in": 0.085, "out": 0.40},
    "ultra": {"id": "nvidia/Nemotron-3-Ultra-550b-a55b", "in": 0.50, "out": 2.20},
}
# 역할 → (모델, 기본 thinking)
ROLES = {
    "respondent": ("nano", True),    # 시뮬레이션 응답자. 추론 켜면 동조율 M4 0.21→0.02 (2026-09-14 n=100 실험), 100명 ≈ $0.016·25초
    "classify": ("nano", False),     # impute_llm 분류
    "extract": ("super", True),      # 명제 추출
    "synthesize": ("super", True),   # 축 후보·비교표·모더레이터·이유 클러스터
    "critique": ("ultra", True),     # 게이트 직전 반박
    "conflict": ("ultra", True),     # 모순 검사
}


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class CallRecord:
    call_id: str
    run_id: str | None
    role: str
    model: str
    tag: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    ok: bool = True
    error: str | None = None
    attempts: int = 1
    raw: str = ""  # 마지막 응답 원문(디버그용, DB 미저장)
    ts: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


def cost_of(model_key: str, prompt_tokens: int, completion_tokens: int) -> float:
    m = MODELS[model_key]
    return (prompt_tokens * m["in"] + completion_tokens * m["out"]) / 1_000_000


def extract_json(text: str) -> str:
    """추론 잔여물·코드펜스를 걷어내고 첫 JSON 객체를 꺼낸다."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if m:
        return m.group(1)
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in response")
    _, end = json.JSONDecoder().raw_decode(text[start:])
    return text[start:start + end]


def compact_schema(schema: type[BaseModel]) -> str:
    def strip(x):
        if isinstance(x, dict):
            return {k: strip(v) for k, v in x.items() if k != "title"}
        if isinstance(x, list):
            return [strip(v) for v in x]
        return x
    return json.dumps(strip(schema.model_json_schema()), ensure_ascii=False, separators=(",", ":"))


MAX_TOKENS_CAP = 16000   # 잘림 재시도 시 늘릴 수 있는 상한


class Router:
    """con을 주면 호출마다 llm_call 테이블에 기록한다. budget_usd를 넘으면 BudgetExceeded."""

    def __init__(self, con=None, run_id: str | None = None, budget_usd: float | None = None,
                 api_key: str | None = None):
        key = api_key or os.environ["NEBIUS_API_KEY"]
        self._key = key
        self.client = OpenAI(base_url=BASE_URL, api_key=key)
        self._aclients = {}  # 이벤트 루프 → AsyncOpenAI (asyncio.run 이 루프를 매번 새로 만들므로)
        self.con, self.run_id, self.budget = con, run_id, budget_usd
        self.calls: list[CallRecord] = []

    # ---------- 공개 API ----------

    @property
    def total_cost(self) -> float:
        return sum(c.cost_usd for c in self.calls)

    def summary(self) -> dict:
        out = {}
        for c in self.calls:
            s = out.setdefault(c.model, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0, "errors": 0})
            s["calls"] += 1; s["prompt_tokens"] += c.prompt_tokens; s["completion_tokens"] += c.completion_tokens
            s["cost_usd"] = round(s["cost_usd"] + c.cost_usd, 6); s["errors"] += (not c.ok)
        return out

    def call(self, role: str, messages: list[dict], schema: type[BaseModel] | None = None,
             max_tokens: int = 800, temperature: float = 0.3, retries: int = 1, tag: str = "",
             thinking: bool | None = None, seed: int | None = None, validator=None):
        """동기 호출. schema를 주면 검증된 pydantic 객체, 아니면 문자열. (result, CallRecord)."""
        async def invoke():
            try:
                return await self.acall(role, messages, schema, max_tokens, temperature, retries, tag, thinking, seed, validator)
            finally:
                await self._close_current_client()
        return asyncio.run(invoke())

    async def acall(self, role: str, messages: list[dict], schema: type[BaseModel] | None = None,
                    max_tokens: int = 800, temperature: float = 0.3, retries: int = 1, tag: str = "",
                    thinking: bool | None = None, seed: int | None = None, validator=None):
        model_key, default_thinking = ROLES[role]
        kwargs = self._kwargs(model_key, schema, max_tokens, temperature,
                              default_thinking if thinking is None else thinking, seed)
        rec = CallRecord(new_id("c"), self.run_id, role, model_key, tag)
        msgs = list(messages)
        if schema is not None:  # 백엔드가 중첩 스키마를 강제하지 않는 경우가 있어 프롬프트에도 명시
            msgs[-1] = {**msgs[-1], "content": msgs[-1]["content"] + "\n\n출력 JSON 스키마(정확히 이 구조·타입, 다른 키 금지):\n" + compact_schema(schema)}
        t0 = time.time()
        last_err = None
        for attempt in range(retries + 1):
            rec.attempts = attempt + 1
            try:
                r = await self._aclient().chat.completions.create(messages=msgs, **kwargs)
                rec.prompt_tokens += r.usage.prompt_tokens
                rec.completion_tokens += r.usage.completion_tokens
                text = r.choices[0].message.content or ""
                rec.raw = text
                if getattr(r.choices[0], "finish_reason", None) == "length":
                    # 토큰 한도에서 잘림(추론 토큰 포함). 잘린 출력을 붙여 재요청하면 더 길어질 뿐 → 원래 메시지로, 한도를 늘려 다시.
                    last_err = f"truncated at max_tokens={kwargs['max_tokens']}"
                    kwargs = {**kwargs, "max_tokens": min(int(kwargs["max_tokens"] * 1.5), MAX_TOKENS_CAP)}
                    msgs = msgs[:len(messages)]
                    result = None
                    continue
                if schema is None:
                    result = text
                else:
                    result = schema.model_validate_json(extract_json(text))
                if validator is not None:
                    validator(result)
                rec.ok, rec.error = True, None
                break
            except (ValidationError, ValueError, json.JSONDecodeError) as e:
                if isinstance(e, ValidationError):
                    details = [{"path": ".".join(map(str, x["loc"])), "error": x["msg"]}
                               for x in e.errors(include_url=False, include_input=False)[:8]]
                    last_err = "ValidationError: " + json.dumps(details, ensure_ascii=False)
                else:
                    last_err = f"{type(e).__name__}: {str(e)[:1000]}"
                msgs = msgs + [{"role": "assistant", "content": text},
                               {"role": "user", "content": f"출력이 스키마에 맞지 않습니다: {last_err}\nJSON만 다시 출력하세요."}]
                result = None
            except Exception as e:  # API 오류
                last_err = f"{type(e).__name__}: {str(e)[:300]}"
                result = None
                await asyncio.sleep(1.0 * (attempt + 1))
        else:
            rec.ok, rec.error = False, last_err
        rec.latency_s = round(time.time() - t0, 3)
        rec.cost_usd = cost_of(model_key, rec.prompt_tokens, rec.completion_tokens)
        self._record(rec)
        if self.budget is not None and self.total_cost > self.budget:
            raise BudgetExceeded(f"cost {self.total_cost:.4f} > budget {self.budget}")
        return result, rec

    async def gather(self, jobs: list[dict], concurrency: int = 32):
        """독립 호출 N개 병렬. jobs = [{role, messages, schema, ...acall kwargs}]. 순서 보존."""
        sem = asyncio.Semaphore(concurrency)

        async def one(j):
            async with sem:
                return await self.acall(**j)
        try:
            return await asyncio.gather(*[one(j) for j in jobs])
        finally:
            await self._close_current_client()

    # ---------- 내부 ----------

    async def _close_current_client(self):
        client = self._aclients.pop(id(asyncio.get_running_loop()), None)
        if client is not None:
            await client.close()

    def _aclient(self):
        loop = asyncio.get_running_loop()
        c = self._aclients.get(id(loop))
        if c is None:
            self._aclients = {id(loop): (c := AsyncOpenAI(base_url=BASE_URL, api_key=self._key))}
        return c

    @staticmethod
    def _kwargs(model_key, schema, max_tokens, temperature, thinking, seed):
        kw = {"model": MODELS[model_key]["id"], "max_tokens": max_tokens, "temperature": temperature,
              "extra_body": {"chat_template_kwargs": {"enable_thinking": bool(thinking)}}}
        if seed is not None:
            kw["seed"] = seed
        if schema is not None:
            kw["response_format"] = {"type": "json_schema", "json_schema": {
                "name": schema.__name__, "schema": schema.model_json_schema(), "strict": True}}
        return kw

    def _record(self, rec: CallRecord):
        self.calls.append(rec)
        if self.con is not None:
            self.con.execute(
                "INSERT INTO llm_call VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [rec.call_id, rec.run_id, rec.role, MODELS[rec.model]["id"], rec.tag, rec.prompt_tokens,
                 rec.completion_tokens, rec.cost_usd, rec.latency_s, rec.ok, rec.error, rec.attempts, rec.ts])

    def records(self) -> list[dict]:
        return [asdict(c) for c in self.calls]
