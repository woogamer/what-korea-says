"""AI 100인 정답판 검증: 한국갤럽 <한국인이 좋아하는 40가지> 2019 (자유응답, 면접, 전국 만 13세 이상 1,700명)과 같은 문항으로 비교.

프롬프트 조정은 TRAIN 문항으로만 고르고, TEST 문항(조정에 안 쓴 것)으로 판정한다(문제 은행에 과적합 방지).
점수 = 겹침 = Σ min(실측%, AI%) / Σ 실측% (수치가 공개된 항목만). 1이면 공개된 분포를 그대로 재현.
출처: https://www.gallup.co.kr/gallupdb/reportContent.asp?seqNo=1048 (음식), 1050 (자연), 1052 (문화)
페르소나는 19세 이상뿐 → 13~18세 제외는 한계로 표시.
실행: .venv/bin/python -m game.validate_feud [--n 200] [--styles A,B,C]
"""
import argparse
import asyncio
import json
from collections import Counter

from game.feud import TARGET, Dist, judge, poll
from game.worldwide import load, sample
from llm.router import Router
from world.config import ROOT

U = dict.fromkeys   # 보고서에 "1% 이상"으로만 나온 항목(수치 미공개 → None, 점수에서 제외하되 매핑 대상)
NONE = "없음/모름"

# 보고서 본문 수치(%) 그대로
TRAIN = {
    "가장 좋아하는 한국음식은 무엇입니까?": {
        "김치찌개": 21, "된장찌개": 15, "불고기": 12, "김치": 10, "비빔밥": 6, "잡채": 4.2, "삼겹살": 3.6, "갈비": 2.7,
        "갈비찜": 1.2, "청국장": 1.2, "떡볶이": 1.2},
    "가장 좋아하는 취미는 무엇입니까?": {
        "등산": 11, "음악감상": 7, "운동/헬스": 7, "게임": 6, "독서": 5, "영화감상": 4, "TV시청": 4, "낚시": 3, "축구": 3, "여행": 3,
        **U(["노래 부르기", "골프", "걷기/산책", "당구", "뜨개질", "요리", "수영", "바둑", "꽃나무 기르기", "자전거/사이클", "그림 그리기", "볼링"])},
    "가장 함께하고 싶은 반려동물은 무엇입니까?": {"개": 60, "고양이": 8, "새": 2, "토끼": 1, NONE: 28},
    "가장 좋아하는 계절은 무엇입니까?": {"봄": 42, "가을": 40, "겨울": 10, "여름": 8},
}
TEST = {
    "가장 좋아하는 중국음식은 무엇입니까?": {
        "짜장면": 47, "짬뽕": 19, "탕수육": 19, "팔보채": 2.3, "양장피": 1.7, "깐풍기": 1.5, "유산슬": 1.3, "우동": 1.1, "볶음밥": 0.9},
    "직접 하는 운동/스포츠 중에서 가장 좋아하는 운동은 무엇입니까?": {
        "축구": 14, "걷기": 11, "야구": 8, "등산": 8, "수영": 6, "헬스": 4, "배드민턴": 4, "요가": 4, "조깅/달리기": 3, "농구": 3,
        **U(["탁구", "골프", "자전거/사이클", "테니스", "당구", "볼링", "댄스", NONE])},   # 없음: 남 10%·여 21%(전체 미공개)
    "가장 좋아하는 과일주스는 무엇입니까?": {
        "오렌지주스": 20, "딸기주스": 17, "사과주스": 14, "토마토주스": 8, "포도주스": 7, "키위주스": 6, "망고주스": 6, "바나나주스": 3.6,
        "수박주스": 3.4, "자몽주스": 2.9, "배주스": 1.9, "귤주스": 1.6, "복숭아주스": 1.3, "파인애플주스": 1.2, "참외주스": 1.1},
    "가장 좋아하는 꽃은 무엇입니까?": {
        "장미": 32, "벚꽃": 6, "안개꽃": 6, "국화": 6, "튤립": 5, "개나리": 4, "프리지어": 4, "코스모스": 4, "진달래": 3, "무궁화": 3,
        **U(["백합", "라일락", "수국", "해바라기", "아카시아꽃", "목련"])},
    "가장 좋아하는 나무는 무엇입니까?": {
        "소나무": 51, "벚나무": 7, "느티나무": 5, "단풍나무": 4, "편백나무": 3, "은행나무": 3, "동백나무": 2, "아카시아나무": 2, "대나무": 2,
        "잣나무": 2, **U(["전나무", "향나무", "버드나무"])},
    "가장 좋아하는 산은 무엇입니까?": {
        "설악산": 26, "지리산": 17, "한라산": 13, "북한산": 5, "백두산": 3, "태백산": 2, "팔공산": 2, "관악산": 2, "도봉산": 2,
        "무등산": 1.4, "내장산": 1.4, "소백산": 1.3, "계룡산": 1.1},
    "만약 개를 기른다면 어떤 개를 가장 길러보고 싶습니까?": {
        "푸들": 16, "몰티즈": 10, "진돗개": 8, "시츄": 5, "포메라니안": 4, "치와와": 4, "비숑프리제": 3, "리트리버": 2,
        "시베리안허스키": 1.8, "웰시코기": 1.2, NONE: 35},
    "만약 고양이를 기른다면 어떤 고양이를 가장 길러보고 싶습니까?": {
        "페르시안": 10, "러시안블루": 5, "샴": 4, "코리안숏헤어": 3, "벵갈": 1, "먼치킨": 1, "스코티시폴드": 1, NONE: 71},
}




async def direct(router, q, target=TARGET):
    """D: 개인 200명 표본 대신 Super에게 전국 분포를 직접 추정시킴(추론 켬). 반환은 poll과 같은 형식의 가상 응답 200개."""
    d, _ = await router.acall("extract", [{"role": "user", "content":
        f"{target} 보기 없이 물었다: \"{q}\"\n자유응답 결과 분포를 추정하라. "
        "고정관념이나 대표 답이 아니라 실제 사람들이 답했을 법한 분포로, 응답 상위 15개 정도와 '없음/모름' 비율을 % 로(합 100)."}],
        schema=Dist, thinking=True, max_tokens=6000, temperature=0.3, retries=2, tag="feud-direct")
    rows = []
    for sh in (d.shares if d else []):
        rows += [{"raw": sh.answer}] * round(sh.pct * 2)
    return rows


async def one(router, people, q, real, style, target=TARGET, rows=None):
    if rows is None:
        rows = await direct(router, q, target) if style == "D" else await poll(router, people, q, style)
    keys = list(real)
    uniq = sorted({r["raw"] for r in rows})
    to_real = dict(zip(uniq, await asyncio.gather(*[judge(router, q, keys, u) for u in uniq])))
    n = len(rows)
    ai = Counter(keys[to_real[r["raw"]]] for r in rows if to_real[r["raw"]] >= 0)
    ai_pct = {k: round(100 * ai.get(k, 0) / n, 1) for k in keys}
    known = [k for k in keys if real[k] is not None]
    score = sum(min(real[k], ai_pct[k]) for k in known) / sum(real[k] for k in known)
    real_top = sorted(known, key=lambda k: -real[k])
    ai_top = sorted(keys, key=lambda k: -ai_pct[k])
    outside = Counter(r["raw"] for r in rows if to_real[r["raw"]] < 0).most_common(4)
    return {"q": q, "style": style, "n": n, "score": round(score, 3), "top1": real_top[0] == ai_top[0],
            "top3": len(set(real_top[:3]) & set(ai_top[:3])), "ai": ai_pct, "outside": outside, "rows": rows}


def show(r, real):
    head = sorted([k for k in real if real[k] is not None], key=lambda k: -real[k])[:6]
    cells = " ".join(f"{k} {real[k]:g}/{r['ai'][k]:g}" for k in head)
    return f"  {r['style']} 겹침 {r['score']:.2f} · 1위 {'✓' if r['top1'] else '✗'} · 상위3 {r['top3']}/3 | 실측/AI: {cells} | 목록 밖: {r['outside'][:3]}"


async def main(n, seed, styles):
    router = Router(budget_usd=3.0)
    people = sample(load(), "Korea", n, seed)
    res = {}
    for split, bank in [("train", TRAIN), ("test", TEST)]:
        for q, real in bank.items():
            print(f"\n[{split}] {q}")
            for st in styles:
                r = await one(router, people, q, real, st)
                res[(split, q, st)] = r
                print(show(r, real), flush=True)
    print("\n=== 평균 겹침 ===")
    for split in ["train", "test"]:
        print(split, " ".join(f"{st} {sum(r['score'] for (sp, _, s), r in res.items() if sp == split and s == st) / len(TRAIN if split == 'train' else TEST):.3f}" for st in styles))
    (ROOT / "data" / "feud_validation.json").write_text(json.dumps([{**{k2: v for k2, v in r.items() if k2 != "rows"}, "split": k[0]} for k, r in res.items()], ensure_ascii=False, indent=1))
    print(f"비용 ${router.total_cost:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--styles", default="A,B,C,D")
    a = ap.parse_args()
    asyncio.run(main(a.n, a.seed, a.styles.split(",")))
