"""Deck loading utilities.

The deck file is a plain list of 60 integer card IDs, one per line. At runtime
on Kaggle the agent's files live under ``/kaggle_simulations/agent/`` so we
search both the current directory and that location.
"""

from __future__ import annotations

import os

DECK_SIZE = 60

_KAGGLE_PREFIX = "/kaggle_simulations/agent/"


def resolve_deck_path(path: str) -> str:
    """Return an existing path for ``path``, falling back to the Kaggle dir."""

    if os.path.exists(path):
        return path
    candidate = _KAGGLE_PREFIX + path
    if os.path.exists(candidate):
        return candidate
    # Last resort: resolve relative to this file's directory.
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidate = os.path.join(here, path)
    if os.path.exists(candidate):
        return candidate
    return path


def read_deck(path: str = "deck.csv") -> list[int]:
    """Read a 60-card deck from ``path``.

    Returns exactly ``DECK_SIZE`` card IDs. Raises ``ValueError`` if the file
    does not contain enough valid integers.
    """

    resolved = resolve_deck_path(path)
    with open(resolved, "r", encoding="utf-8") as fh:
        tokens = [line.strip() for line in fh.read().splitlines()]

    deck: list[int] = []
    for tok in tokens:
        if tok == "":
            continue
        deck.append(int(tok))

    if len(deck) < DECK_SIZE:
        raise ValueError(
            f"Deck file '{resolved}' has {len(deck)} cards; expected {DECK_SIZE}."
        )
    return deck[:DECK_SIZE]
