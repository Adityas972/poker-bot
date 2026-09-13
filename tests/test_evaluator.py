import pytest

from poker_bot.engine.card import Card
from poker_bot.engine.evaluator import HandCategory, evaluate_hand


def cards(text: str):
    """'Ah Kh Qh Jh Th' -> [Card, ...]"""
    return [Card.from_str(tok) for tok in text.split()]


@pytest.mark.parametrize("hand, expected_category", [
    ("Ah Kh Qh Jh Th", HandCategory.STRAIGHT_FLUSH),
    ("5h 4h 3h 2h Ah", HandCategory.STRAIGHT_FLUSH),  # wheel, straight flush
    ("Kc Kd Kh Ks 2c", HandCategory.FOUR_OF_A_KIND),
    ("Kc Kd Kh 2s 2c", HandCategory.FULL_HOUSE),
    ("Ah Kh 9h 5h 2h", HandCategory.FLUSH),
    ("9c 8d 7h 6s 5c", HandCategory.STRAIGHT),
    ("5c 4d 3h 2s Ac", HandCategory.STRAIGHT),  # wheel
    ("Kc Kd Kh 5s 2c", HandCategory.THREE_OF_A_KIND),
    ("Kc Kd 5h 5s 2c", HandCategory.TWO_PAIR),
    ("Kc Kd 9h 5s 2c", HandCategory.PAIR),
    ("Ac Kd 9h 5s 2c", HandCategory.HIGH_CARD),
])
def test_category_detection(hand, expected_category):
    assert evaluate_hand(cards(hand)).category == expected_category


def test_royal_flush_description():
    result = evaluate_hand(cards("Ah Kh Qh Jh Th"))
    assert result.description == "Royal Flush"


def test_wheel_straight_is_five_high_not_ace_high():
    result = evaluate_hand(cards("5c 4d 3h 2s Ac"))
    assert result.sort_key == (HandCategory.STRAIGHT, 5)


def test_seven_card_hand_picks_best_five():
    # Hole cards Ah Ad + board with a full house available among 7 cards.
    seven = cards("Ah Ad Ac 2h 2d 9s 4c")
    result = evaluate_hand(seven)
    assert result.category == HandCategory.FULL_HOUSE
    assert result.description == "Full House, Aces full of Twos"


@pytest.mark.parametrize("better, worse", [
    ("Ah Kh Qh Jh Th", "Kc Kd Kh Ks 2c"),   # straight flush > quads
    ("Kc Kd Kh Ks 2c", "Kc Kd Kh 2s 2c"),   # quads > full house
    ("Kc Kd Kh 2s 2c", "Ah Kh 9h 5h 2h"),   # full house > flush
    ("Ah Kh 9h 5h 2h", "9c 8d 7h 6s 5c"),   # flush > straight
    ("9c 8d 7h 6s 5c", "Kc Kd Kh 5s 2c"),   # straight > trips
    ("Kc Kd Kh 5s 2c", "Kc Kd 5h 5s 2c"),   # trips > two pair
    ("Kc Kd 5h 5s 2c", "Kc Kd 9h 5s 2c"),   # two pair > pair
    ("Kc Kd 9h 5s 2c", "Ac Kd 9h 5s 2c"),   # pair > high card
    ("Ac Kd 9h 5s 2c", "Kc Qd 9h 5s 2c"),   # A-high > K-high
])
def test_hand_category_ordering(better, worse):
    assert evaluate_hand(cards(better)) > evaluate_hand(cards(worse))


def test_kicker_breaks_tie_within_same_category():
    pair_ace_kicker = evaluate_hand(cards("Kc Kd Ah 5s 2c"))
    pair_king_kicker = evaluate_hand(cards("Kc Kd Qh 5s 2c"))
    assert pair_ace_kicker.category == pair_king_kicker.category == HandCategory.PAIR
    assert pair_ace_kicker > pair_king_kicker


def test_equal_hands_compare_equal():
    a = evaluate_hand(cards("Ah Kh Qh Jh Th"))
    b = evaluate_hand(cards("As Ks Qs Js Ts"))
    assert a.sort_key == b.sort_key
