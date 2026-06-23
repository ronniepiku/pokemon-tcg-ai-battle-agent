"""Lazy, cached access to the simulator's card and attack metadata.

``all_card_data()`` / ``all_attack()`` call into the native library, so we load
them once and expose fast lookup tables. All access is defensive: if the native
library is unavailable for any reason the tables are empty and callers fall back
to neutral behaviour.
"""

from __future__ import annotations

from typing import Optional

from cg.api import Attack, CardData, all_attack, all_card_data


class CardDB:
    """Cached lookup tables for card and attack metadata."""

    def __init__(self) -> None:
        self.cards: dict[int, CardData] = {}
        self.attacks: dict[int, Attack] = {}
        self.card_count: int = 1
        self.attack_count: int = 1
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        try:
            cards = all_card_data()
            attacks = all_attack()
            self.cards = {c.cardId: c for c in cards}
            self.attacks = {a.attackId: a for a in attacks}
            self.card_count = (max(self.cards) if self.cards else 0) + 1
            self.attack_count = (max(self.attacks) if self.attacks else 0) + 1
        except Exception:
            # Leave tables empty; callers handle missing data gracefully.
            self.cards = {}
            self.attacks = {}
        self._loaded = True

    def card(self, card_id: Optional[int]) -> Optional[CardData]:
        if card_id is None:
            return None
        return self.cards.get(card_id)

    def attack(self, attack_id: Optional[int]) -> Optional[Attack]:
        if attack_id is None:
            return None
        return self.attacks.get(attack_id)


# Module-level singleton.
DB = CardDB()
