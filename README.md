# poker-bot

A poker-playing agent built from scratch for heads-up No-Limit Texas
Hold'em, using Counterfactual Regret Minimization (CFR) as the core
solver, with Deep CFR as a later stage once tabular CFR stops scaling.

## Decisions locked in

| Decision | Choice |
|---|---|
| Variant | Heads-up No-Limit Texas Hold'em |
| Players | 2 (heads-up) |
| Core algorithm | CFR family (vanilla CFR -> CFR+ -> Deep CFR) |
| Bet sizing | Discrete pot-fraction buckets: fold, check/call, 33%/75%/150% pot, all-in |
| Hand evaluator | Built from scratch, with human-readable descriptions |
| Stack model | Reset to a fresh 100bb every hand (no persistent bankroll) |

## Roadmap / stages

- [x] **Stage 0** - project scaffolding, CFR regret-matching node
      (validated on Kuhn Poker against its known closed-form Nash
      equilibrium, in `poker_bot/games/kuhn.py`)
- [x] **Stage 1** - card/deck primitives + from-scratch hand evaluator
      (verified exactly against the textbook 5-card hand frequency table)
- [x] **Stage 2** - heads-up No-Limit Hold'em betting engine: blinds,
      position rules, pot-fraction bet sizing, min-raise legality,
      all-in-for-less refunds, street/showdown transitions
- [ ] **Stage 3** - information-set abstraction for CFR on full HUNL
      (card bucketing; the raw game is too large for tabular CFR as-is)
- [ ] **Stage 4** - CFR+ / Monte Carlo CFR training loop on the abstracted
      game
- [ ] **Stage 5** - Deep CFR (neural function approximation replacing
      tabular regret/strategy tables) for RL-scale training
- [ ] **Stage 6** - Evaluation: exploitability/best-response estimates,
      play-vs-bot CLI

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running tests

```bash
python3 -m pytest            # fast tests
python3 -m pytest --runslow  # also runs the exhaustive evaluator check
```

## Package layout

```
poker_bot/
  cfr/
    node.py       # regret-matching info-set node, reused across games
  engine/
    card.py       # Card, Deck
    evaluator.py  # from-scratch 5-7 card hand evaluator
    game.py       # heads-up No-Limit Hold'em betting state machine
  games/
    kuhn.py       # Kuhn Poker + vanilla CFR (equilibrium sanity check)
```
