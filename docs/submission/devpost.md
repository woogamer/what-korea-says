# Devpost submission — What Korea Says

Tagline: A Family Feud–style survey quiz where an AI writes the answer board and real players turn it into Korean public opinion.

Track: Best Apps and Agents · Demo: https://nebius.woogamer.win/play/?lang=en

## Inspiration

Ask an AI "What do Koreans like?" and you get a confident, plausible answer. But how close is it to what people actually say?

We had spent weeks building a synthetic-consumer research tool on NVIDIA's Korean persona data. The hardest lesson was that synthetic answers are a prior, not data. They become trustworthy only when real responses correct them.

We wanted to make that lesson something anyone can feel in two minutes. So we built a Family Feud–style quiz show where an AI writes the answer board, people play, and every round shows how far the AI was from reality.

## What it does

What Korea Says is a survey quiz game with 30 questions, such as favorite Korean food, a mom's most common nagging, or what you do on payday. It runs in English and Korean.

- **You answer first.** Your answer joins the board for the next players, and becomes your first guess.
- **You guess the board.** Three strikes and you are out. Matching understands English and Korean, typos, synonyms and brand names.
- **The reveal** puts three things side by side:
  - the board today
  - what Nemotron originally predicted
  - for 18 questions, what Gallup Korea measured (16 from its 2024 face-to-face survey of 1,777 people, 2 from 2026 releases)

  A bar shows how much of the board is still AI and how much is players.

The board is a Bayesian posterior: `(κ · AI + player counts) / (κ + players)`.
- **κ ≈ 8** was fit to 30 Gallup question–survey pairs (2019, 2024 and 2026 releases; the game shows the latest 18). The AI prediction is worth about 8 real respondents.
- **The first players** move the board a lot, and later players move it less.
- **New answers** the AI missed appear once two players give them.
- **Who counts:** only players living in Korea, one vote per browser per question.

## How we built it

- **Nebius Token Factory** serves every model call through its OpenAI-compatible API.
- **NVIDIA Nemotron 3 Super** does four jobs:
  - It writes each prior board as a direct estimate of the national answer distribution (reasoning on, three runs averaged).
  - It translates and merges answers while we build the bank.
  - It maps slots to Gallup categories.
  - It judges every answer and guess as a generous game-show host, following per-question rules we wrote after reading real player answers. The screen gets a fast verdict with reasoning off. A careful verdict with reasoning on runs in the background and corrects it.
- **NVIDIA Nemotron 3 Nano + Nemotron-Personas-Korea** powered the baseline we tested and rejected: up to 1,000 synthetic Koreans per question.
- **The server** is FastAPI and SQLite with a plain JavaScript front end. Confirmed matches are cached, so each distinct answer costs at most a few model calls. A budget cap falls back to exact text matching.

## Challenges

- **Synthetic people collapse onto one answer, and more of them does not help.** We asked 1,000 personas all 18 Gallup questions. Overlap with Gallup went 0.49 (50 personas) → 0.51 (200) → 0.52 (1,000), and then stopped improving. The error is bias, not noise:
  - Favorite tree: cherry 84% (Gallup: pine 36%).
  - Leading AI company: Google 82% (Gallup: Samsung 37%).
  - Favorite weekday: Saturday 75% (Gallup: Friday 34%, Saturday 33%).
  - Favorite baseball team: one team 70% (Gallup: 46% have none). Personas never say "none".

  Asking Super directly for the national distribution scored higher on every held-out Gallup set: 0.66 vs 0.51 on 2024 questions, and 0.62 vs 0.26 on 2026 questions released after the model's training.
- **Merging answers is where boards break.** One automatic merge pass put jajangmyeon, jjamppong and sweet-and-sour pork into a single 68% slot. We now match exact text first and let the judge decide the rest, with a careful second verdict that moves wrongly merged answers back.

## Accomplishments

**126 real players in two days, and they made the board better.** We only shared the link in group chats with friends and family, with no ads. Players stayed: they played 887 rounds, 9.5 questions each on average. 47 played 10 or more, and 11 played all 30. In total they gave 1,151 answers and 3,479 guesses (Oct 28–29, 2026), so 22–32 Korea-resident players answered each Gallup question.

Their answers moved the board from the AI's 0.61 to 0.68 overlap with Gallup:

| Board on the 18 Gallup questions | Overlap with Gallup (1.0 = identical) |
|---|---|
| 1,000 synthetic personas | 0.52 |
| AI direct estimate (the first board) | 0.61 |
| About 25 real players only | 0.65 |
| **AI + real players (the live board)** | **0.68** |

- **About 25 real people beat 1,000 synthetic ones.** The live board moved closer to Gallup on 12 of 18 questions. Pet went 0.56 → 0.90 and spectator sport 0.69 → 0.85.
- **It also moved away on 6.** Favorite baseball team dropped 0.59 → 0.51, because a few fans of one team move a 25-person board.
- **The same pattern held in simulation.** Overlap rose from 0.69 with no players to 0.76 with 10, 0.83 with 30 and 0.90 with 100, and the combined board beat both AI-only and players-only at every player count. κ and these numbers are leave-one-out: κ was fit only on the other questions.

We also report what failed:
- **Age reweighting.** Our players are mostly in their 30s and 40s. Reweighting them to Korea's adult population made the live board worse (0.68 → 0.63), because a few older players got very large weights. Older players did not answer more like Gallup than younger ones.
- **Global bias correction.** One correction learned across questions gained only +0.02.
- **Overconfidence.** The AI prior's 80% intervals covered the truth only 65% of the time.

## What we learned

The interesting number is not how good the AI is. It is how many real people the AI is worth.

For everyday preference questions, the answer is about 8. A thousand synthetic people are worth less than 25 real ones, because they all make the same mistake. AI is a good first draft of public opinion. People are the correction.

## What's next

- Reach older and more representative players. Then test age reweighting again with enough players in each age group.
- Let players submit their own questions, so the AI writes the first board instantly.
- Publish the live boards as an open dataset of Korean everyday preferences.

## Built with

Nebius Token Factory · NVIDIA Nemotron 3 Super · NVIDIA Nemotron 3 Nano · NVIDIA Nemotron-Personas-Korea · FastAPI · SQLite · Python · JavaScript

## Feedback on the tools

**Nebius Token Factory**
- The OpenAI-compatible endpoint made integration a few lines.
- The per-call token usage let us log the cost of every experiment. The whole research behind this game cost a few dollars. The 1,000-persona experiment, about 18,000 calls, cost $1.13.
- `json_schema` response format worked on Nano but was not enforced for nested schemas on Super. We validate and retry in the application.
- At about 50 concurrent connections we saw two process crashes during SSL handling. 25 was stable.

**NVIDIA Nemotron**
- Reasoning on/off (`chat_template_kwargs.enable_thinking`) changes behavior a lot. With it off, the judge became fast enough for live play. With it on, its verdicts were more careful, so we use both: fast on screen, careful in the background.
- Reasoning tokens count against `max_tokens`, so short answers need a generous budget.
- Run-to-run variance in distribution estimates is large. Averaging three runs helped.
- As a judge, the model sometimes read a bare number as a list index ("2" → slot #2). We match numbers without the model.

**Nemotron-Personas-Korea**
- It is a strong demographic scaffold.
- Asking each persona independently still produced a stereotyped consensus that did not improve from 200 to 1,000 personas. Guidance on when to sample personas versus asking for an aggregate estimate would help builders.
