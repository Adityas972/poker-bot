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
| Raises per street | Capped at 3 (unbounded No-Limit raise wars make exhaustive-action CFR intractable regardless of card abstraction) |
| Hand evaluator | Built from scratch, with human-readable descriptions |
| Stack model | Reset to a fresh 100bb every hand (no persistent bankroll) |
| Card abstraction | Hand-strength percentile buckets from Monte Carlo equity |
| CFR variant | External-sampling Monte Carlo CFR (vanilla CFR's exhaustive traversal is intractable on real No-Limit betting trees) |

## Roadmap / stages

- [x] **Stage 0** - project scaffolding, CFR regret-matching node
      (validated on Kuhn Poker against its known closed-form Nash
      equilibrium, in `poker_bot/games/kuhn.py`)
- [x] **Stage 1** - card/deck primitives + from-scratch hand evaluator
      (verified exactly against the textbook 5-card hand frequency table)
- [x] **Stage 2** - heads-up No-Limit Hold'em betting engine: blinds,
      position rules, pot-fraction bet sizing, min-raise legality,
      all-in-for-less refunds, street/showdown transitions
- [x] **Stage 3** - hand-strength percentile bucketing (Monte Carlo
      equity-based card abstraction)
- [x] **Stage 4** - external-sampling Monte Carlo CFR training loop,
      wired to the real engine/evaluator/bucketer. Works correctly
      (validated qualitatively: stronger hands learn to fold less and
      raise more), but tabular CFR alone does not scale well here --
      even with only 6 hand-strength buckets, the number of distinct
      information sets exploded past 200,000 within 3,000 hands at
      ~10 hands/sec single-threaded Python, meaning most info sets get
      almost no visits. This is the standard wall real poker-AI work
      hits, and is exactly why Stage 5 (Deep CFR) exists.
- [x] **Stage 5** - Deep CFR (Brown et al. 2019): two small PyTorch MLPs
      per training run (an advantage network per player, predicting
      counterfactual regret from a fixed-size info-set feature vector,
      plus a policy network for the final average strategy), trained
      via reservoir-buffered samples collected from the same
      external-sampling traversal structure as Stage 4. Replaces
      discrete hand buckets with continuous Monte Carlo equity as a
      direct input feature, and generalizes across info sets instead of
      needing individual visits -- this is what actually fixes Stage 4's
      scaling wall, and is the RL-flavored piece of this project.
- [x] **Stage 6a** - play-vs-bot CLI (`poker_bot/play.py`): real hands
      against a trained Deep CFR policy network, with the bot's live
      strategy probabilities shown at each decision
- [ ] **Stage 6b** - exploitability / best-response estimates for the
      trained bot (approximate, since exact best response is
      intractable on full HUNL even abstracted)

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

## Playing against the bot

```bash
python3 -m poker_bot.train_bot   # trains a bot, ~2 minutes, saves to poker_bot/models/

python3 -m poker_bot.play        # play heads-up against it in the terminal
# or, for a browser UI:
python3 -m poker_bot.web.app     # then open http://127.0.0.1:5050
```

The web UI shows a real poker table (your cards always visible, the
bot's revealed only at showdown), clickable action buttons sized from
the actual legal actions at each decision, and a live "bot's reasoning"
panel showing the strategy probabilities the bot considered before each
of its moves.

## Package layout

```
poker_bot/
  cfr/
    node.py         # regret-matching info-set node, reused across games
    hunl_trainer.py # external-sampling MCCFR self-play on full HUNL
  engine/
    card.py         # Card, Deck
    evaluator.py    # from-scratch 5-7 card hand evaluator
    game.py         # heads-up No-Limit Hold'em betting state machine
  abstraction/
    equity.py       # Monte Carlo hand-equity estimation
    buckets.py      # hand-strength percentile bucketing (tabular card abstraction)
  deep_cfr/
    features.py     # fixed-size info-set feature encoding for the neural nets
    networks.py      # advantage/policy network architecture (shared MLP)
    memory.py        # reservoir-sampling buffer
    trainer.py        # Deep CFR training loop
  games/
    kuhn.py         # Kuhn Poker + vanilla CFR (equilibrium sanity check)
  train_bot.py      # trains a Deep CFR bot and saves its policy network
  play.py           # interactive CLI: play heads-up against the trained bot
  web/
    app.py           # Flask API + page routes for the browser UI
    templates/index.html
    static/style.css, app.js
```
