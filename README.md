# What Korea Says

**A Family Feud–style survey quiz whose answer board starts as an AI guess and becomes real public opinion as people play.**

Live demo: https://nebius.woogamer.win/play/ · License: MIT · Built on Nebius Token Factory + NVIDIA Nemotron

## The game

1. **Answer first.** "What is your favorite Korean food?" Type your own answer (English or Korean). It is saved *after* your round, so it never leaks into the board you are guessing.
2. **Guess the board.** Find the most common answers. Three wrong guesses and the round ends. Your score is the sum of the percentages you uncover.
3. **Reveal.** See the full board side by side with what the AI originally predicted and, for 18 of the 30 questions, what Gallup Korea actually measured in a face-to-face survey. A bar shows how much of today's board is still AI and how much is players.

## How the board evolves

Every board starts as a **prior**: NVIDIA Nemotron 3 Super's direct estimate of the national answer distribution (average of three runs). Player answers update it with a Dirichlet posterior:

```
board_k = (κ · AI_k + players_k) / (κ + n)
```

- **κ ≈ 8** means the AI estimate is trusted like 8 real respondents. We did not pick it by hand. We fit it by empirical Bayes (Dirichlet-multinomial marginal likelihood) on 30 Gallup question–survey pairs (2019, 2024 and 2026 releases; the game shows the latest 18).
- With no players the board is 100% AI. At 8 players it is half and half; at 30 players it is about 80% people.
- A brand-new answer that the AI never predicted needs at least 2 matching players before it can appear on the board.
- Only players who say they live in Korea are counted, because the board claims to be Korean opinion.

## Does it work? (simulated against Gallup)

We drew simulated players from the real Gallup distributions and scored each board by overlap with the survey (Σ min(board, survey) / Σ survey; 1.0 = identical). κ and every number below are leave-one-out: κ was fit only on the other questions, and the 2019 and 2024 versions of the same question were held out together, so no question is scored with a κ it helped fit. Simulated players are a representative sample; real players are self-selected (see Data and limits).

| Players per question | AI only | Players only | **AI + players** |
|---|---|---|---|
| 0 | 0.69 | – | 0.69 |
| 10 | 0.69 | 0.64 | **0.76** |
| 30 | 0.69 | 0.79 | **0.83** |
| 100 | 0.69 | 0.89 | **0.90** |

The combined board was never worse than either source alone. The AI is useful when few people have played. People overtake it at around 30 players.

## Real players (live, Oct 28–29, 2026)

We shared the link with friends and family for two days. As of Oct 29, 19:00 KST:
- **126 players**, 1,151 own answers and 3,479 guesses.
- 781 answers count toward the boards. The rest came from people who did not tick "I live in Korea" or were not answers.
- On the 18 Gallup questions, 22–32 Korea-resident players answered each (mean 26).

| Board on the 18 Gallup questions | Overlap with Gallup |
|---|---|
| AI only (Nemotron's first board) | 0.61 |
| Real players only | 0.65 |
| **AI + players (the live board)** | **0.68** |

- **Closer to Gallup on 12 of 18 questions.** The biggest gains were pet (0.56 → 0.90), spectator sport (0.69 → 0.85), tree (0.65 → 0.81), number (0.63 → 0.80) and mountain (0.77 → 0.92).
- **Further from Gallup on 6.** The largest drops were favorite baseball team (0.59 → 0.51) and flower (0.52 → 0.45). With about 25 players, a few fans of one team move the board a lot.
- **The same pattern as the simulation.** About 25 players alone are noisy, and the AI prior steadies them, so the combination beat either source alone.
- **Not comparable across tables.** Live scores use only the slots shown on the board, so they are lower than the simulation, which scores the AI's full answer list.

**Age reweighting did not help.** Our players skew to their 30s and 40s, while Korea's adult population skews older:

| Age | 20s | 30s | 40s | 50s | 60+ | not given |
|---|---|---|---|---|---|---|
| Players | 13 | 36 | 30 | 12 | 10 | 28 |

- Reweighting players to the adult population made the board worse: 0.68 → 0.63, worse on 13 of 18.
- Half-strength reweighting scored 0.675, still slightly worse.
- Older players did not answer more like Gallup than younger ones. Reweighting only put heavy weight on the few players over 50 in each question.
- At this size, more players help more than reweighting.

Reproduce with `.venv/bin/python -m game.live_report` while the server runs.

**What did not work:**
- Learning a global "how the AI is wrong" correction from other questions gained only +0.02, so it is not used.
- The model's 80% intervals covered the truth only 65% of the time at zero players, meaning the AI prior is overconfident.

## Why a direct estimate, not synthetic people

We first sampled Nemotron-Personas-Korea profiles and asked each one with Nemotron 3 Nano on Nebius. **More personas did not help.** We asked 1,000 personas all 18 Gallup questions and scored random subsets ($1.13 in total):

| Personas per question | 50 | 100 | 200 | 500 | 1,000 |
|---|---|---|---|---|---|
| Overlap with Gallup | 0.49 | 0.50 | 0.51 | 0.52 | 0.52 |

The error is bias, not sampling noise. A thousand different people converge on one answer, and none of them says "none":

| Question | Gallup | 1,000 personas |
|---|---|---|
| Favorite tree | pine 36% | cherry 84% |
| Leading AI company | Samsung 37% | Google 82% |
| Favorite number | 7 at 34% | 7 at 83% |
| Favorite weekday | Friday 34%, Saturday 33% | Saturday 75% |
| Favorite baseball team | none 46% | one team 70% |

About 25 real players per question (0.65) beat 1,000 synthetic ones (0.52).

Asking Super directly for the national distribution was more accurate on every held-out set (200 personas):

| Gallup set | Persona sample | Direct estimate |
|---|---|---|
| 2024, repeated questions (12) | 0.51 | **0.66** |
| 2024, new questions (4) | 0.65 | **0.67** |
| 2026, after model training (2) | 0.26 | **0.62** |

The direct estimate is also not a memorized copy of the older 2019 report. On 8 of 12 questions, its output is closer to the 2024 results than to the 2019 ones.

## How answers are matched

Free-text guesses like "kimchi jjigae", "김치찌개" or "Kimchi stew!" must land on the same slot.

1. **Text match:** a normalized exact match against slot names and aliases is tried first. Numbers match only the same number.
2. **Model judge:** otherwise Nemotron 3 Super judges which slot, if any, the guess belongs to. It plays a generous Family Feud host and follows per-question rules written after reading real player answers (`game/web/judge_notes.json`), for example "a chicken brand menu counts as its flavour" or "different mountains never match". Jokes and non-answers are stored but never counted.
3. **Fast, then careful:** the screen gets a fast verdict with reasoning off. A careful verdict with reasoning on runs in the background, fixes the slot if needed and caches it. Each distinct answer costs at most a few model calls. A spending cap falls back to text matching only.

**Building the question bank** (`game/bank.py`) uses Super too:
- It translates the board to English and merges duplicate answers.
- It maps AI slots onto Gallup categories.

Manual fixes are keyed by Korean text, so a rebuild cannot silently reshuffle stored player answers.

## Run it

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env            # add NEBIUS_API_KEY (Nebius Token Factory)
.venv/bin/uvicorn game.web.app:app --port 8077    # http://localhost:8077
.venv/bin/python -m pytest tests/test_feud_engine.py -q
```

- **Question bank:** the 30-question bank ships in `game/feud_bank.json`. To rebuild it, run `.venv/bin/python -m game.bank` (about $0.01).
- **Experiments:** `game/evolve.py` runs the evolving-model experiments, `game/validate_feud*.py` the persona-vs-direct comparisons, `game/validate_persona_n.py` the 1,000-persona curve, and `game/live_report.py` the real-player numbers.
- **Environment:** `FEUD_DB` sets where answers are stored (default `data/feud_game.db`). `FEUD_BUDGET_USD` caps model spending per server process (default 5).

## Repository

| Path | What |
|---|---|
| `game/web/` | FastAPI server, game engine (posterior board, matching, reveal), single-page UI |
| `game/bank.py`, `game/feud_bank.json` | Question bank builder and the shipped bank |
| `game/evolve.py`, `game/validate_*.py`, `game/feud.py`, `game/live_report.py` | Experiments and the live report behind the numbers above |
| `llm/router.py` | Token Factory client: role → Nemotron model, JSON-schema validation and retry, cost log, budget cap |
| `world/`, `sim/`, `tools/`, `evidence/`, … | An earlier prototype (a "market world model" for synthetic consumer research). Its central lesson carried over: treat synthetic answers as a prior and update them with real responses |

## Data and limits

- **Sources:** reality figures are cited from Gallup Korea's public reports:
  - *50 Things Koreans Like* (2024; 1,777 respondents, face-to-face, open-ended)
  - two 2026 Gallup releases (favorite baseball team, leading AI company)

  Source links are in the bank. 12 questions have no survey and show the AI and player boards only.
- **Model output:** AI boards are model estimates, not survey results.
- **Who plays:** players are self-selected friends and family, mostly in their 30s and 40s, so the board describes the people who played, adjusted only by the AI prior.
- **One vote per browser:** each browser counts once per question, with at most 6 votes per network per question.
- **Lenient matching:** for example, "kimchi" is accepted as "Kimchi stew", although Gallup counts them separately.

## Built with

Nebius Token Factory · NVIDIA Nemotron 3 Super · NVIDIA Nemotron 3 Nano · NVIDIA Nemotron-Personas-Korea (CC BY 4.0) · FastAPI · SQLite · Python
