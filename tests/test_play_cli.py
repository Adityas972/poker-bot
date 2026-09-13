import random

from poker_bot.deep_cfr.networks import InfosetMLP
from poker_bot.engine.card import Card
from poker_bot.engine.game import Street, start_hand
from poker_bot.play import _cards_str, _describe_action, _play_one_hand


def test_cards_str_formats_and_handles_empty_board():
    cards = (Card.from_str("Ah"), Card.from_str("Td"))
    assert _cards_str(cards) == "Ah Td"
    assert _cards_str([]) == "(none)"


def test_describe_action_covers_every_action_label():
    state = start_hand(rng=random.Random(0))
    for action in ["FOLD", "CALL", "BET_33", "BET_75", "BET_150", "ALLIN"]:
        text = _describe_action(state, action)
        assert isinstance(text, str) and len(text) > 0


def test_describe_action_check_vs_call_wording():
    state = start_hand(rng=random.Random(0))
    # Facing the SB/BB gap: this is a real call, not a check.
    assert "CALL" in _describe_action(state, "CALL")

    state.street = Street.FLOP
    state.street_contrib = [0, 0]
    assert "CHECK" in _describe_action(state, "CALL")


def _always_call(actions):
    return actions.index("CALL")


def test_play_one_hand_end_to_end_with_scripted_human_and_untrained_bot():
    """Script the human to always check/call (never fold or raise), let
    an untrained (random-weight) network play the bot, and confirm a
    full hand runs all the way to showdown without crashing and produces
    a valid zero-sum result."""
    policy_net = InfosetMLP(hidden_dim=8)
    rng = random.Random(42)
    session_net = [0, 0]

    net = _play_one_hand(policy_net, rng, session_net, choose_index=_always_call)

    assert sum(net) == 0
    assert session_net[0] == net[0]
    assert session_net[1] == net[1]


def test_play_one_hand_with_folding_human():
    policy_net = InfosetMLP(hidden_dim=8)
    rng = random.Random(1)
    session_net = [0, 0]

    def fold_if_possible(actions):
        return actions.index("FOLD") if "FOLD" in actions else actions.index("CALL")

    net = _play_one_hand(policy_net, rng, session_net, choose_index=fold_if_possible)
    assert sum(net) == 0
    assert net[0] < 0  # human folded immediately, loses the small blind
