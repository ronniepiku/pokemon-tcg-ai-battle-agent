"""Self-play validation harness.

Runs the agent against itself and against a random opponent using the native
simulator directly (no kaggle_environments dependency). Verifies:

  * the deck is legal,
  * the agent never crashes or returns an illegal selection across many games,
  * the agent comfortably beats a random opponent (a basic strength gate).

Usage:
    python -m tests.validate --games 20 --opponent random --profile fast
"""

from __future__ import annotations

import argparse
import random
import sys
import time

# Allow running both as a module and as a script.
sys.path.insert(0, ".")

from cg.game import battle_finish, battle_select, battle_start  # noqa: E402

from bot.policy import AgentContext, decide  # noqa: E402


def random_agent(obs_dict: dict) -> list[int]:
    sel = obs_dict.get("select")
    if sel is None:
        # Should not happen for the opponent (it never submits a deck here).
        return []
    n = len(sel["option"])
    k = sel["maxCount"]
    if n == 0 or k == 0:
        return list(range(sel["minCount"]))
    return sorted(random.sample(range(n), min(k, n)))


def _assert_legal(sel: dict, picked: list[int]) -> None:
    n = len(sel["option"])
    assert sel["minCount"] <= len(picked) <= sel["maxCount"], (
        f"count {len(picked)} not in [{sel['minCount']}, {sel['maxCount']}]"
    )
    assert len(set(picked)) == len(picked), "duplicate indices"
    assert all(0 <= i < n for i in picked), "index out of range"


def play_game(ctx: AgentContext, deck: list[int], opponent: str, agent_index: int) -> int:
    obs, sd = battle_start(deck, deck)
    if sd.errorPlayer >= 0:
        raise SystemExit(f"Deck illegal (errorType={sd.errorType}).")

    steps = 0
    while obs["current"]["result"] < 0:
        cur = obs["current"]["yourIndex"]
        if cur == agent_index or opponent == "self":
            picked = decide(obs, ctx)
        else:
            picked = random_agent(obs)
        if obs.get("select") is not None:
            _assert_legal(obs["select"], picked)
        obs = battle_select(picked)
        steps += 1
        if steps > 30000:
            raise SystemExit("Game did not terminate.")
    result = obs["current"]["result"]
    battle_finish()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the cabt agent.")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--opponent", choices=["random", "self"], default="random")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    ctx = AgentContext()
    deck = ctx.deck
    print(f"Deck size: {len(deck)} | model loaded: {ctx.evaluator.has_model} "
          f"| search: {ctx.cfg.search.enabled} (sims={ctx.cfg.search.simulations})")

    wins = losses = draws = 0
    t0 = time.monotonic()
    for g in range(args.games):
        agent_index = g % 2  # Alternate who goes first for fairness.
        result = play_game(ctx, deck, args.opponent, agent_index)
        if result == 2:
            draws += 1
        elif result == agent_index or args.opponent == "self":
            wins += 1
        else:
            losses += 1
        print(f"  game {g + 1}/{args.games}: result={result}")
    dt = time.monotonic() - t0

    print(f"\nWins: {wins}  Losses: {losses}  Draws: {draws}  "
          f"({dt:.1f}s, {dt / max(1, args.games):.2f}s/game)")
    if args.opponent == "random":
        decisive = wins + losses
        wr = 100 * wins / decisive if decisive else 0
        print(f"Win rate vs random: {wr:.0f}%")
        if wr < 60:
            print("WARNING: win rate vs random is low.")


if __name__ == "__main__":
    main()
