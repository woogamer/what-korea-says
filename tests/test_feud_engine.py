import asyncio
import json

import pytest

from game.web.engine import Game


class NoLLM:
    """글자 일치로 끝나야 하는 경우 LLM을 부르면 실패."""
    async def acall(self, *a, **k):
        raise AssertionError("LLM called")


class FakeLLM:
    def __init__(self, slot):
        self.slot, self.calls = slot, 0

    async def acall(self, *a, **k):
        from game.feud import Match
        self.calls += 1
        return Match(slot=self.slot), None


BANK = {"kappa": 8, "questions": [{
    "id": "q", "question_en": "Favorite season?", "question_ko": "계절?", "board_size": 4,
    "prior": [{"en": "Spring", "ko": "봄", "alias_ko": ["벚꽃 피는 봄"], "pct": 40, "reality_ko": "봄"},
              {"en": "Autumn", "ko": "가을", "alias_ko": [], "pct": 30, "reality_ko": "가을"},
              {"en": "Summer", "ko": "여름", "alias_ko": [], "pct": 20, "reality_ko": "여름"},
              {"en": "Winter", "ko": "겨울", "alias_ko": [], "pct": 10, "reality_ko": "겨울"}],
    "reality": {"items": [{"ko": "봄", "en": "Spring", "pct": 40}, {"ko": "가을", "en": "Autumn", "pct": 35},
                          {"ko": "여름", "en": "Summer", "pct": 13}, {"ko": "겨울", "en": "Winter", "pct": 12}]},
    "source": {"name": "Gallup", "url": "u"}}]}


@pytest.fixture
def game(tmp_path):
    p = tmp_path / "bank.json"
    p.write_text(json.dumps(BANK))
    return lambda router: Game(p, tmp_path / "g.db", router)


def test_prior_only_board_is_ai(game):
    g = game(NoLLM())
    b, n = g.board("q")
    assert n == 0 and [s["label"] for s in b] == ["Spring", "Autumn", "Summer", "Winter"]
    assert b[0]["now"] == pytest.approx(40)


def test_posterior_formula_and_exact_match(game):
    g = game(NoLLM())
    for t in ["winter", "겨울", "WINTER!"] * 4:   # 12명 겨울, 글자 일치만으로 배정
        assert asyncio.run(g.answer("q", t)) == "p:겨울"
    b, n = g.board("q")
    winter = next(s for s in b if s["label"] == "Winter")
    assert n == 12 and winter["now"] == pytest.approx(100 * (8 * 0.1 + 12) / (8 + 12))
    assert g.ai_weight(n) == pytest.approx(0.4)


def test_new_slot_needs_two_players(game):
    g = game(FakeLLM(-1))
    asyncio.run(g.answer("q", "장마철"))
    assert all(s["key"] != "n:장마철" for s in g.board("q")[0])
    asyncio.run(g.answer("q", "장마철"))
    assert any(s["key"] == "n:장마철" for s in g.slots("q")[0])


def test_llm_resolution_is_cached(game):
    # 화면엔 빠른 판정, 뒤에서 신중 판정(refine)이 저장 → 같은 글자는 다음부터 LLM 안 부름
    llm = FakeLLM(0)
    g = game(llm)

    async def go():
        assert await g.answer("q", "cherry blossom season", who="a") == "p:봄"
        await asyncio.gather(*g._tasks)
        assert llm.calls == 2
        assert await g.answer("q", "cherry blossom season", who="b") == "p:봄"
        assert (await g.guess("q", "cherry blossom season"))[0] == 0
        await asyncio.gather(*g._tasks)
        assert llm.calls == 2
    asyncio.run(go())


def test_refine_fixes_fast_verdict(game):
    # 빠른 판정이 틀려도(새 답) 신중 판정이 칸을 고침
    class TwoLLM(FakeLLM):
        async def acall(self, *a, **k):
            self.slot = 0 if k.get("thinking") else -1
            return await super().acall(*a, **k)
    g = game(TwoLLM(-1))

    async def go():
        assert (await g.answer("q", "cherry blossom season", who="a")).startswith("n:")
        await asyncio.gather(*g._tasks)
    asyncio.run(go())
    assert g.db.execute("SELECT key FROM answers").fetchone()[0] == "p:봄"


def test_junk_answer_not_counted(game):
    g = game(FakeLLM(-2))
    assert asyncio.run(g.answer("q", "seungyeon", who="a")).startswith("x:")
    assert g.board("q")[1] == 0


def test_number_answer_never_asks_llm(game):
    g = game(NoLLM())
    assert asyncio.run(g.guess("q", "2"))[0] == -1


def test_reveal_overlap(game):
    g = game(NoLLM())
    r = asyncio.run(g.reveal("q"))
    assert r["gallup"]["overlap_ai"] == pytest.approx((40 + 30 + 13 + 10) / 100)


def test_non_korea_answers_not_counted(game):
    g = game(NoLLM())
    asyncio.run(g.answer("q", "winter", kr=False))
    assert g.board("q")[1] == 0
    asyncio.run(g.answer("q", "winter", kr=True))
    assert g.board("q")[1] == 1


def test_one_answer_per_player(game):
    g = game(FakeLLM(-1))
    asyncio.run(g.answer("q", "장마철", who="a"))
    asyncio.run(g.answer("q", "장마철", who="a"))   # 같은 사람이 두 번 → 한 표
    assert next(s["count"] for s in g.slots("q")[0] if s["key"] == "n:장마철") == 1
    asyncio.run(g.answer("q", "장마철", who="b"))
    assert next(s["count"] for s in g.slots("q")[0] if s["key"] == "n:장마철") == 2


def test_same_wifi_family_each_counts_up_to_cap(game):
    from game.web.engine import IP_CAP
    g = game(FakeLLM(-1))
    for i in range(IP_CAP + 2):   # 같은 IP, 브라우저(쿠키)는 각자
        asyncio.run(g.answer("q", "장마철", who=f"c{i}", ip="home"))
    assert next(s["count"] for s in g.slots("q")[0] if s["key"] == "n:장마철") == IP_CAP


def test_alias_key_survives_rebuild(game, tmp_path):
    g = game(NoLLM())
    g.db.execute("INSERT INTO answers VALUES ('q','벚꽃','p:벚꽃 피는 봄',NULL,NULL,1,0,NULL,NULL)")   # 예전 빌드에서 따로 칸이던 답
    g.db.commit()
    spring = next(s for s in g.board("q")[0] if s["label"] == "Spring")
    assert spring["count"] == 1
