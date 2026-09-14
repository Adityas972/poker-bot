"""Web UI for playing heads-up No-Limit Hold'em against the trained
Deep CFR bot.

Usage:
    python3 -m poker_bot.train_bot   # if you haven't already
    python3 -m poker_bot.web.app
    -> open http://127.0.0.1:5050 in a browser

Each browser gets its own game state, keyed off a random id in its
signed session cookie and held in the in-memory SESSIONS dict below --
enough for many concurrent players on a single low-traffic demo
instance without a database. This only works run as a single process
(see the Dockerfile's `-w 1`): splitting across multiple worker
processes would fragment SESSIONS and a player could lose their game
mid-hand depending on which worker handled the next request.
"""

import os
import random
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field

from flask import Flask, jsonify, render_template, request, session

from poker_bot.deep_cfr.trainer import load_policy_net, policy_strategy
from poker_bot.engine.evaluator import evaluate_hand
from poker_bot.engine.game import _action_amount, apply_action, legal_actions, start_hand
from poker_bot.play import BOT, HUMAN, STREET_NAMES
from poker_bot.train_bot import MODEL_PATH

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(24))

_policy_net = None
_policy_net_mtime = None

MAX_SESSIONS = 200


@dataclass
class SessionState:
    state: object = None
    session_net: list = field(default_factory=lambda: [0, 0])
    rng: random.Random = field(default_factory=random.Random)


SESSIONS: "OrderedDict[str, SessionState]" = OrderedDict()


def _get_session() -> SessionState:
    """Looks up (or creates) this browser's game state. Bounded to
    MAX_SESSIONS, evicting the oldest first, so a flood of one-off
    visitors can't grow this dict without limit."""
    sid = session.get("sid")
    if sid is None:
        sid = uuid.uuid4().hex
        session["sid"] = sid
    if sid not in SESSIONS:
        SESSIONS[sid] = SessionState()
    SESSIONS.move_to_end(sid)
    if len(SESSIONS) > MAX_SESSIONS:
        SESSIONS.popitem(last=False)
    return SESSIONS[sid]


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


def _play_bot_turns(sess: SessionState, policy_net) -> list:
    """Auto-play the bot's turns until it's the human's turn or the hand
    ends. Returns a log of what the bot did, for display."""
    state = sess.state
    log = []
    while not state.is_terminal and state.to_act == BOT:
        strategy = policy_strategy(policy_net, state, BOT, sess.rng, equity_rollouts=150)
        actions = list(strategy.keys())
        probs = [strategy[a] for a in actions]
        action = sess.rng.choices(actions, weights=probs, k=1)[0]
        log.append({
            "street": STREET_NAMES[state.street],
            "strategy": {a: round(p, 3) for a, p in strategy.items() if p > 0.01},
            "chosen": action,
            "chosen_label": _action_button_label(state, action),
        })
        apply_action(state, action)
    return log


def _serialize(sess: SessionState, bot_log=None):
    state = sess.state
    data = {
        "model_loaded": _get_policy_net() is not None,
        "in_hand": state is not None,
        "terminal": bool(state and state.is_terminal),
        "session_net": {"you": sess.session_net[HUMAN], "bot": sess.session_net[BOT]},
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
    _get_session()  # make sure the session cookie is set on first load
    return render_template("index.html")


@app.route("/api/new_hand", methods=["POST"])
def new_hand():
    sess = _get_session()
    policy_net = _get_policy_net()
    if policy_net is None:
        return jsonify({"error": f"No trained model found at {MODEL_PATH}. "
                                  f"Run `python3 -m poker_bot.train_bot` first."}), 400

    sess.state = start_hand(rng=sess.rng, starting_stack_bb=100, big_blind=100)
    bot_log = _play_bot_turns(sess, policy_net) if sess.state.to_act == BOT else []
    return jsonify(_serialize(sess, bot_log))


@app.route("/api/action", methods=["POST"])
def take_action():
    sess = _get_session()
    if sess.state is None or sess.state.is_terminal:
        return jsonify({"error": "no hand in progress"}), 400
    if sess.state.to_act != HUMAN:
        return jsonify({"error": "not your turn"}), 400

    action = request.json.get("action")
    if action not in [a["action"] for a in _serialize(sess)["legal_actions"]]:
        return jsonify({"error": f"illegal action {action!r}"}), 400

    apply_action(sess.state, action)
    policy_net = _get_policy_net()
    bot_log = _play_bot_turns(sess, policy_net) if not sess.state.is_terminal else []

    if sess.state.is_terminal:
        net = sess.state.payouts()
        sess.session_net[HUMAN] += net[HUMAN]
        sess.session_net[BOT] += net[BOT]

    return jsonify(_serialize(sess, bot_log))


@app.route("/api/state")
def get_state():
    sess = _get_session()
    return jsonify(_serialize(sess))


if __name__ == "__main__":
    app.run(port=5050, debug=False)
