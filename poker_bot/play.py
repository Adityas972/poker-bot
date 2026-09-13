"""Interactive CLI: play heads-up No-Limit Hold'em against the trained
Deep CFR bot.

Usage: python3 -m poker_bot.play
(train a bot first with: python3 -m poker_bot.train_bot)
"""

import random
import sys

from poker_bot.deep_cfr.trainer import load_policy_net, policy_strategy
from poker_bot.engine.evaluator import evaluate_hand
from poker_bot.engine.game import (
    Street, _action_amount, apply_action, legal_actions, start_hand,
)
from poker_bot.train_bot import MODEL_PATH

HUMAN = 0
BOT = 1
STREET_NAMES = {Street.PREFLOP: "Preflop", Street.FLOP: "Flop",
                 Street.TURN: "Turn", Street.RIVER: "River"}


def _cards_str(cards) -> str:
    return " ".join(repr(c) for c in cards) if cards else "(none)"


def _describe_action(state, action: str) -> str:
    if action == "FOLD":
        return "FOLD"
    if action == "CALL":
        opponent = 1 - state.to_act
        to_call = state.street_contrib[opponent] - state.street_contrib[state.to_act]
        return "CHECK" if to_call <= 0 else f"CALL ({min(to_call, state.stacks[state.to_act])} chips)"
    total = _action_amount(state, action)
    committed = total - state.street_contrib[state.to_act]
    label = {"BET_33": "bet 33% pot", "BET_75": "bet 75% pot",
             "BET_150": "bet 150% pot", "ALLIN": "ALL-IN"}[action]
    return f"{action} ({label}, +{committed} chips, total this street {total})"


def _print_state(state) -> None:
    print(f"\n--- {STREET_NAMES[state.street]} | pot: {state.pot} | "
          f"your stack: {state.stacks[HUMAN]} | bot stack: {state.stacks[BOT]} ---")
    if state.board:
        print(f"Board: {_cards_str(state.board)}")


def _prompt_for_action_index(actions) -> int:
    while True:
        choice = input("Choose an action number: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(actions):
            return int(choice) - 1
        print("Invalid choice, try again.")


def _human_turn(state, choose_index=_prompt_for_action_index) -> None:
    actions = legal_actions(state)
    print(f"Your hand: {_cards_str(state.hole[HUMAN])}")
    print("Your options:")
    for i, action in enumerate(actions):
        print(f"  {i + 1}) {_describe_action(state, action)}")
    apply_action(state, actions[choose_index(actions)])


def _bot_turn(state, policy_net, rng) -> None:
    strategy = policy_strategy(policy_net, state, BOT, rng, equity_rollouts=150)
    actions = list(strategy.keys())
    probs = [strategy[a] for a in actions]
    action = rng.choices(actions, weights=probs, k=1)[0]
    prob_str = ", ".join(f"{a}={p:.2f}" for a, p in sorted(strategy.items(), key=lambda kv: -kv[1]) if p > 0.01)
    print(f"Bot's strategy here: {prob_str}")
    print(f"Bot chooses: {_describe_action(state, action)}")
    apply_action(state, action)


def _play_one_hand(policy_net, rng, session_net, choose_index=_prompt_for_action_index) -> list:
    state = start_hand(rng=rng, starting_stack_bb=100, big_blind=100)
    print("\n" + "=" * 60)
    print("New hand. You post the small blind, bot posts the big blind.")

    last_street = None
    while not state.is_terminal:
        if state.street != last_street:
            _print_state(state)
            last_street = state.street
        if state.to_act == HUMAN:
            _human_turn(state, choose_index)
        else:
            _bot_turn(state, policy_net, rng)

    print(f"\n--- Result | final board: {_cards_str(state.board)} ---")
    if state.folded_player is not None:
        winner = "You" if state.folded_player == BOT else "Bot"
        print(f"{'Bot' if state.folded_player == BOT else 'You'} folded. {winner} win{'s' if winner == 'Bot' else ''} the pot.")
    else:
        your_hand = evaluate_hand(list(state.hole[HUMAN]) + state.full_board)
        bot_hand = evaluate_hand(list(state.hole[BOT]) + state.full_board)
        print(f"Your hand:  {_cards_str(state.hole[HUMAN])} -> {your_hand.description}")
        print(f"Bot's hand: {_cards_str(state.hole[BOT])} -> {bot_hand.description}")

    net = state.payouts()
    session_net[HUMAN] += net[HUMAN]
    session_net[BOT] += net[BOT]
    outcome = "won" if net[HUMAN] > 0 else ("lost" if net[HUMAN] < 0 else "chopped")
    print(f"You {outcome} {abs(net[HUMAN])} chips this hand. "
          f"Session total: you {session_net[HUMAN]:+d}, bot {session_net[BOT]:+d}")
    return net


def main():
    if not MODEL_PATH.exists():
        print(f"No trained model found at {MODEL_PATH}.")
        print("Train one first with: python3 -m poker_bot.train_bot")
        sys.exit(1)

    policy_net = load_policy_net(str(MODEL_PATH))
    rng = random.Random()
    session_net = [0, 0]

    print("Playing heads-up No-Limit Hold'em against the Deep CFR bot.")
    print("Stacks reset to 100bb every hand. Type Ctrl+C any time to quit.\n")

    try:
        while True:
            _play_one_hand(policy_net, rng, session_net)
            again = input("\nPlay another hand? [Y/n]: ").strip().lower()
            if again == "n":
                break
    except KeyboardInterrupt:
        pass

    print(f"\nFinal session total: you {session_net[HUMAN]:+d}, bot {session_net[BOT]:+d}")


if __name__ == "__main__":
    main()
