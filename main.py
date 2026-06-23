"""Kaggle submission entry point for the Pokémon TCG AI Battle (cabt) challenge.

The competition harness calls ``agent(obs_dict)`` repeatedly:

* On the very first call ``obs["select"]`` is ``None`` and the agent must return
  a 60-card deck (a list of card IDs).
* On every subsequent call it must return a list of option indices, with length
  in ``[select.minCount, select.maxCount]``, unique, and each ``0 <= i < len(option)``.

All heavy state (config, deck, card metadata, evaluator/model) is built once and
cached in a module-level ``AgentContext``.
"""

from __future__ import annotations

from bot.policy import AgentContext, decide

_CTX: AgentContext | None = None


def agent(obs_dict: dict) -> list[int]:
    """Robustly select a legal action for the given observation."""

    global _CTX
    if _CTX is None:
        _CTX = AgentContext()
    return decide(obs_dict, _CTX)


if __name__ == "__main__":
    # Minimal smoke test against the native simulator (no kaggle_environments
    # required). Runs a single self-play game with the real agent.
    from cg.api import to_observation_class
    from cg.game import battle_finish, battle_select, battle_start

    ctx = AgentContext()
    deck = ctx.deck
    obs, sd = battle_start(deck, deck)
    if sd.errorPlayer >= 0:
        raise SystemExit(f"Deck error (type {sd.errorType}).")

    steps = 0
    while obs["current"]["result"] < 0:
        sel = agent(obs)
        obs = battle_select(sel)
        steps += 1
        if steps > 20000:
            raise SystemExit("Game did not terminate.")
    result = obs["current"]["result"]
    battle_finish()
    print(f"Self-play finished in {steps} steps. Result: {result}")
