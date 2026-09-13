"""Card and Deck primitives for a standard 52-card deck."""

import random
from dataclasses import dataclass

RANK_CHARS = "..23456789TJQKA"  # index 2-14; 0/1 unused
SUITS = "cdhs"  # clubs, diamonds, hearts, spades


@dataclass(frozen=True, order=True)
class Card:
    rank: int  # 2-14 (11=J, 12=Q, 13=K, 14=A)
    suit: str  # one of 'c', 'd', 'h', 's'

    def __post_init__(self):
        if not 2 <= self.rank <= 14:
            raise ValueError(f"rank must be 2-14, got {self.rank}")
        if self.suit not in SUITS:
            raise ValueError(f"suit must be one of {SUITS!r}, got {self.suit!r}")

    def __repr__(self):
        return f"{RANK_CHARS[self.rank]}{self.suit}"

    @classmethod
    def from_str(cls, text: str) -> "Card":
        """Parse e.g. 'Ah', 'Td', '9c'."""
        rank_char, suit_char = text[0].upper(), text[1].lower()
        return cls(RANK_CHARS.index(rank_char), suit_char)


def full_deck() -> list[Card]:
    return [Card(rank, suit) for rank in range(2, 15) for suit in SUITS]


class Deck:
    """A shuffled deck that deals cards without replacement."""

    def __init__(self, rng: random.Random = None):
        self._rng = rng or random.Random()
        self.cards = full_deck()
        self._rng.shuffle(self.cards)
        self._next = 0

    def deal(self, n: int = 1) -> list[Card]:
        if self._next + n > len(self.cards):
            raise ValueError("not enough cards left in the deck")
        dealt = self.cards[self._next:self._next + n]
        self._next += n
        return dealt

    @property
    def remaining(self) -> int:
        return len(self.cards) - self._next
