"""Local web UI for playing heads-up No-Limit Hold'em against the
trained Deep CFR bot.

Usage:
    python3 -m poker_bot.train_bot   # if you haven't already
    python3 -m poker_bot.web.app
    -> open http://127.0.0.1:5050 in a browser

Single-player, single-process local tool: game state lives in module-
level variables rather than a database, which is fine for one person
playing in one browser tab.
"""

import random

from flask import Flask, jsonify, render_template, request

from poker_bot.deep_cfr.trainer import load_policy_net, policy_strategy
from poker_bot.engine.evaluator import evaluate_hand
from poker_bot.engine.game import _action_amount, apply_action, legal_actions, start_hand
from poker_bot.play import BOT, HUMAN, STREET_NAMES
from poker_bot.train_bot import MODEL_PATH

app = Flask(__name__)

_policy_net = None
_policy_net_mtime = None
_state = None
_rng = random.Random()
_session_net = [0, 0]


def _get_policy_net():
    """Loads the policy network, and reloads it if the file on disk has
    changed since it was last loaded (e.g. a retrain finished) -- a
    long-running server process would otherwise keep serving whatever
    model happened to be loaded first, silently ignoring retrains."""
    global _policy_net, _policy_net_mtime
    if not MODEL_PATH.exists():
        return None
    mtime = MODEL_PATH.stat().st_mtime
    if _policy_net is None or mtime != _policy_net_mtime:
        _policy_net = load_policy_net(str(MODEL_PATH))
        _policy_net_mtime = mtime
    return _policy_net


def _action_button_label(state, action: str) -> str:
    if action == "FOLD":
        return "Fold"
    if action == "CALL":
        opponent = 1 - state.to_act
        to_call = state.street_contrib[opponent] - state.street_contrib[state.to_act]
        to_call = min(to_call, state.stacks[state.to_act])
        return "Check" if to_call <= 0 else f"Call {to_call}"
    total = _action_amount(state, action)
    committed = total - state.street_contrib[state.to_act]
    pretty = {"BET_33": "Bet 33%", "BET_75": "Bet 75%",
              "BET_150": "Bet 150%", "ALLIN": "All-in"}[action]
    return f"{pretty} ({committed})"


def _play_bot_turns(policy_net) -> list:
    """Auto-play the bot's turns until it's the human's turn or the hand
    ends. Returns a log of what the bot did, for display."""
    log = []
    while not _state.is_terminal and _state.to_act == BOT:
        strategy = policy_strategy(policy_net, _state, BOT, _rng, equity_rollouts=150)
        actions = list(strategy.keys())
        probs = [strategy[a] for a in actions]
        action = _rng.choices(actions, weights=probs, k=1)[0]
        log.append({
            "street": STREET_NAMES[_state.street],
            "strategy": {a: round(p, 3) for a, p in strategy.items() if p > 0.01},
            "chosen": action,
            "chosen_label": _action_button_label(_state, action),
        })
        apply_action(_state, action)
    return log


def _serialize(bot_log=None):
    state = _state
    data = {
        "model_loaded": _get_policy_net() is not None,
        "in_hand": state is not None,
        "terminal": bool(state and state.is_terminal),
        "session_net": {"you": _session_net[HUMAN], "bot": _session_net[BOT]},
        "bot_log": bot_log or [],
    }
    if state is None:
        return data

    data.update({
        "street": STREET_NAMES[state.street],
        "board": [repr(c) for c in state.board],
        "pot": state.pot,
        "your_stack": state.stacks[HUMAN],
        "bot_stack": state.stacks[BOT],
        "your_hole": [repr(c) for c in state.hole[HUMAN]],
        "to_act": "you" if (not state.is_terminal and state.to_act == HUMAN) else None,
        "legal_actions": [],
    })

    if not state.is_terminal and state.to_act == HUMAN:
        data["legal_actions"] = [
            {"action": a, "label": _action_button_label(state, a)}
            for a in legal_actions(state)
        ]

    if state.is_terminal:
        data["bot_hole"] = [repr(c) for c in state.hole[BOT]]
        data["full_board"] = [repr(c) for c in state.full_board]
        if state.folded_player is not None:
            data["outcome"] = "you_folded" if state.folded_player == HUMAN else "bot_folded"
        else:
            data["outcome"] = "showdown"
            data["your_hand_desc"] = evaluate_hand(list(state.hole[HUMAN]) + state.full_board).description
            data["bot_hand_desc"] = evaluate_hand(list(state.hole[BOT]) + state.full_board).description
        net = state.payouts()
        data["net"] = {"you": net[HUMAN], "bot": net[BOT]}

    return data


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/new_hand", methods=["POST"])
def new_hand():
    global _state
    policy_net = _get_policy_net()
    if policy_net is None:
        return jsonify({"error": f"No trained model found at {MODEL_PATH}. "
                                  f"Run `python3 -m poker_bot.train_bot` first."}), 400

    _state = start_hand(rng=_rng, starting_stack_bb=100, big_blind=100)
    bot_log = _play_bot_turns(policy_net) if _state.to_act == BOT else []
    return jsonify(_serialize(bot_log))


@app.route("/api/action", methods=["POST"])
def take_action():
    if _state is None or _state.is_terminal:
        return jsonify({"error": "no hand in progress"}), 400
    if _state.to_act != HUMAN:
        return jsonify({"error": "not your turn"}), 400

    action = request.json.get("action")
    if action not in [a["action"] for a in _serialize()["legal_actions"]]:
        return jsonify({"error": f"illegal action {action!r}"}), 400

    apply_action(_state, action)
    policy_net = _get_policy_net()
    bot_log = _play_bot_turns(policy_net) if not _state.is_terminal else []

    if _state.is_terminal:
        net = _state.payouts()
        _session_net[HUMAN] += net[HUMAN]
        _session_net[BOT] += net[BOT]

    return jsonify(_serialize(bot_log))


@app.route("/api/state")
def get_state():
    return jsonify(_serialize())


if __name__ == "__main__":
    app.run(port=5050, debug=False)
