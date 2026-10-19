# Devpost submission — What Korea Says

Track: Best Apps and Agents · Demo: https://nebius.woogamer.win/play/

## Inspiration

Ask an AI "What do Koreans like?" and you get a confident, plausible answer. But how close is it to what people actually say?

We had spent weeks building a synthetic-consumer research tool on NVIDIA's Korean persona data. The hardest lesson was that synthetic answers are a prior, not data. They become trustworthy only when real responses correct them.

We wanted to make that lesson something anyone can feel in two minutes. So we built a quiz show where an AI writes the answer board, people play, and every round shows how far the AI was from reality.

## What it does

What Korea Says is a survey quiz game with 30 questions, such as favorite Korean food, a mom's most common nagging, or what you do on payday.

- **You answer first.** Your answer is saved after the round, so it never helps you.
- **You guess the board.** Three strikes and you are out. Matching understands English and Korean, typos and synonyms.
- **The reveal** puts three things side by side:
  - the board today
  - what Nemotron originally predicted
  - for 18 questions, what Gallup Korea measured in a 1,777-person face-to-face survey

  A bar shows how much of the board is still AI and how much is players.

The board is a Bayesian posterior: `(κ · AI + player counts) / (κ + players)`.
- **κ ≈ 8** was fit to 30 Gallup question–survey pairs. The AI prediction is worth about 8 real respondents.
- **The first players** move the board a lot, and later players move it less.
- **New answers** the AI missed appear once two players give them.
- **Who counts:** only players living in Korea count.

## How we built it

- **Nebius Token Factory** serves every model call through its OpenAI-compatible API.
- **NVIDIA Nemotron 3 Super** does four jobs:
  - It writes each prior board as a direct estimate of the national answer distribution (reasoning on, three runs averaged).
  - It translates and merges answers while we build the bank.
  - It maps slots to Gallup categories.
  - It judges player guesses in real time (reasoning off, about 0.5 s).
- **NVIDIA Nemotron 3 Nano + Nemotron-Personas-Korea** powered the baseline we tested and rejected: sampling 200 synthetic Koreans per question.
- **The server** is FastAPI and SQLite with a plain JavaScript front end. Confirmed matches are cached, so each distinct answer costs at most one model call. A budget cap falls back to exact text matching.

## Challenges

- **Synthetic people collapse onto stereotypes.** With 200 personas, "Google" won the leading-AI-company question with 84% (Gallup: 15%, Samsung 37%). Asking Super directly for the national distribution scored higher on every held-out Gallup set: 0.66 vs 0.51 on 2024 questions, and 0.62 vs 0.26 on 2026 questions released after the model's training.
- **Merging answers is where boards break.** One automatic merge pass put jajangmyeon, jjamppong and sweet-and-sour pork into a single 68% slot, so we threw it out. The final pipeline:
  - exact match first
  - a bilingual judge
  - an alias re-check that sends wrongly merged answers back to the right slot
  - manual fixes keyed by Korean text, so rebuilds stay stable
- **Your own answer was a hint.** Early builds counted your answer before you guessed, so it now saves only at the end of the round.

## Accomplishments

In a simulation against Gallup, overlap with the real survey (1.0 = identical) rises with players:

| Players | Board overlap |
|---|---|
| 0 | 0.69 |
| 10 | 0.76 |
| 30 | 0.83 |
| 100 | 0.90 |

The combined board beat both AI-only and players-only at every player count.

We also report what failed:
- A global bias correction learned across questions gained only +0.02.
- The AI prior is overconfident: its 80% intervals covered the truth only 65% of the time.

## What we learned

The interesting number is not how good the AI is. It is how many real people the AI is worth.

For everyday preference questions, the answer is about 8. That is far fewer than in our earlier consumer-purchase study, where synthetic data stayed useful up to a few hundred respondents.

## What's next

- Collect real players and compare the live boards with Gallup.
- Add age and gender reweighting of players. It already works in simulation.
- Let players submit their own questions, so the AI writes the first board instantly.

## Built with

Nebius Token Factory · NVIDIA Nemotron 3 Super · NVIDIA Nemotron 3 Nano · NVIDIA Nemotron-Personas-Korea · FastAPI · SQLite · Python · JavaScript

## Feedback on the tools

**Nebius Token Factory**
- The OpenAI-compatible endpoint made integration a few lines.
- The per-call token usage let us log the cost of every experiment. The whole research behind this game cost a few dollars.
- `json_schema` response format worked on Nano but was not enforced for nested schemas on Super. We validate and retry in the application.
- At about 50 concurrent connections we saw two process crashes during SSL handling. 25 was stable.

**NVIDIA Nemotron**
- Reasoning on/off (`chat_template_kwargs.enable_thinking`) changes behavior a lot. With reasoning on, synthetic respondents gave far more varied and realistic answers. With it off, the judge became fast enough for real-time play.
- Reasoning tokens count against `max_tokens`, so short answers need a generous budget.
- Run-to-run variance in distribution estimates is large. Averaging three runs helped.

**Nemotron-Personas-Korea**
- It is a strong demographic scaffold.
- Asking each persona independently still produced a stereotyped consensus. Guidance on when to sample personas versus asking for an aggregate estimate would help builders.
