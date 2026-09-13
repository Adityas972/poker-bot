# poker-bot

A poker-playing agent built from scratch, following the classical poker-AI
research path (Kuhn -> Leduc -> Libratus/DeepStack-style heads-up No-Limit
Hold'em), using Counterfactual Regret Minimization (CFR) as the core solver,
with Deep CFR as a later stage once tabular CFR stops scaling.

## Decisions locked in

| Decision | Choice |
|---|---|
| Variant | Heads-up No-Limit Texas Hold'em |
| Players | 2 (heads-up) |
| Core algorithm | CFR family (vanilla CFR -> CFR+ -> Deep CFR) |
| Bet sizing | Discrete pot-fraction buckets |
| Hand evaluator | Built from scratch |
| Stack depth (final game) | 100 big blinds |
| Curriculum | Kuhn Poker -> Leduc Hold'em -> full HUNL |

## Roadmap / stages

- [x] **Stage 0** - project scaffolding
- [ ] **Stage 1** - Kuhn Poker + vanilla CFR (verify convergence to the known
      closed-form Nash equilibrium)
- [ ] **Stage 2** - Leduc Hold'em + CFR (adds a betting round, a community
      card, and a real (if tiny) card abstraction problem)
- [ ] **Stage 3** - Heads-up No-Limit Hold'em engine (deck, hand evaluator,
      betting engine, discrete bet-size action abstraction)
- [ ] **Stage 4** - CFR+ / Monte Carlo CFR on abstracted HUNL
- [ ] **Stage 5** - Deep CFR (neural function approximation replacing
      tabular regret/strategy tables)
- [ ] **Stage 6** - Evaluation: exploitability estimates, play vs bot via CLI

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running Stage 1 (Kuhn Poker)

```bash
python3 -m poker_bot.games.kuhn
```

This runs vanilla CFR self-play for N iterations and prints the learned
average strategy per information set, plus the average game value for
player 0 (should converge close to the known equilibrium value of -1/18).
