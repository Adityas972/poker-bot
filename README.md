# poker-bot

A heads-up No-Limit Hold'em bot built from scratch in Python. Core solver is Counterfactual Regret Minimization (CFR), starting tabular and moving to Deep CFR once the info-set count got out of hand. Comes with a CLI and a small local web UI so you can actually play against it.

## What's in here

- From-scratch hand evaluator, checked against all 2,598,960 possible 5-card hands
- Full heads-up No-Limit betting engine: blinds, position, pot-fraction bet sizing, min-raise rules, all-in handling, the works
- Vanilla CFR on Kuhn Poker as a sanity check (it converges to the known closed-form equilibrium, which is a nice way to confirm the regret-matching code isn't wrong before trusting it on anything bigger)
- Monte Carlo equity + percentile bucketing for a card abstraction
- External-sampling MCCFR on the full game — plain CFR can't even finish a single traversal here because No-Limit betting trees are unbounded, so this was a required switch, not a nice-to-have
- Deep CFR: two small PyTorch nets (advantage + policy) trained on reservoir-sampled self-play, so the bot generalizes across info sets instead of needing to visit each one individually — tabular CFR blew past 200k info sets after just 3000 hands, which is the wall this gets you past
- A terminal CLI and a browser UI for playing against the trained bot, with the bot's live action probabilities shown so you can see what it's actually thinking

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training a bot

```bash
python3 -m poker_bot.train_bot          # ~2 min, decent for testing things work
python3 -m poker_bot.train_bot --big    # ~25-30 min, plays noticeably better
```

Bigger network, more self-play hands, less noisy equity estimates. The trained model isn't checked into git (it's a binary blob and regenerates fine), so you'll need to run this before playing.

## Playing against it

```bash
python3 -m poker_bot.play        # terminal
python3 -m poker_bot.web.app     # browser, then open http://127.0.0.1:5050
```

## Tests

```bash
python3 -m pytest            # fast suite
python3 -m pytest --runslow  # also runs the exhaustive hand-evaluator check (~8s)
```

## Layout

```
poker_bot/
  cfr/
    node.py         # regret-matching info-set node, shared across games
    hunl_trainer.py # external-sampling MCCFR on the full game
  engine/
    card.py         # Card, Deck
    evaluator.py    # hand evaluator
    game.py         # betting state machine
  abstraction/
    equity.py       # Monte Carlo hand-equity estimation
    buckets.py      # percentile bucketing (tabular card abstraction)
  deep_cfr/
    features.py     # info-set -> fixed-size feature vector
    networks.py      # advantage/policy network
    memory.py        # reservoir buffer
    trainer.py        # Deep CFR training loop
  games/
    kuhn.py         # Kuhn Poker + vanilla CFR
  train_bot.py      # trains and saves a bot
  play.py           # terminal CLI
  web/
    app.py           # Flask backend for the browser UI
    templates/, static/
```

## Known limitations

- No exploitability/best-response measurement against the trained bot yet — for the full game that's approximate at best, but even that isn't wired up.
- Training is small-scale and CPU-only, so don't expect anything close to solver-strength play. A `--big` run gets you sensible, hand-strength-aware decisions, not a hard-to-beat opponent.
- Deep CFR training has real run-to-run variance at this scale — two runs with the same settings can land in noticeably different places.
