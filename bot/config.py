"""Configuration for the Pokémon TCG agent.

Two profiles are provided:
  * ``fast``   – tiny search budget for quick local iteration / CI.
  * ``ladder`` – stronger search budget for competitive play.

Every field can be overridden through environment variables so behaviour is
reproducible without code changes, e.g. ``CABT_PROFILE=fast CABT_SEED=7``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class SearchConfig:
    """Parameters controlling the forward-model MCTS."""

    enabled: bool = True          # Use forward-model MCTS for MAIN decisions.
    simulations: int = 32         # MCTS simulations per decision.
    max_time_s: float = 2.0       # Hard wall-clock budget per decision (seconds).
    c_puct: float = 0.4           # Exploration constant in the PUCT formula.
    max_branch: int = 24          # Max candidate actions expanded at a node.
    only_main: bool = True        # Restrict search to SelectType.MAIN selections.
    min_options_for_search: int = 2  # Skip search when the choice is forced.
    rollouts: int = 1             # Greedy playouts per leaf (0 = static eval).
    rollout_max_steps: int = 400  # Safety cap on a single playout length.


@dataclass(frozen=True)
class AgentConfig:
    """Top-level agent configuration."""

    seed: int = 0
    deck_path: str = "deck.csv"
    model_path: str = "model.pth"
    use_model: bool = True        # Load model.pth for leaf evaluation if present.
    force_search: bool = False    # Use MCTS even without a trained model.
    search: SearchConfig = SearchConfig()
    profile: str = "ladder"


# Profile presets ---------------------------------------------------------------

_FAST = SearchConfig(enabled=True, simulations=12, max_time_s=0.5, rollouts=1)
_LADDER = SearchConfig(enabled=True, simulations=40, max_time_s=3.0, rollouts=1)

_PROFILES = {
    "fast": _FAST,
    "ladder": _LADDER,
}


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    try:
        return int(val) if val is not None else default
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    val = os.environ.get(name)
    try:
        return float(val) if val is not None else default
    except ValueError:
        return default


def load_config() -> AgentConfig:
    """Build the active configuration from defaults + environment overrides."""

    profile = os.environ.get("CABT_PROFILE", "ladder").strip().lower()
    search = _PROFILES.get(profile, _LADDER)

    # Allow fine-grained overrides on top of the chosen profile.
    search = replace(
        search,
        enabled=_env_bool("CABT_SEARCH", search.enabled),
        simulations=_env_int("CABT_SIMS", search.simulations),
        max_time_s=_env_float("CABT_TIME", search.max_time_s),
        max_branch=_env_int("CABT_BRANCH", search.max_branch),
    )

    return AgentConfig(
        seed=_env_int("CABT_SEED", 0),
        deck_path=os.environ.get("CABT_DECK", "deck.csv"),
        model_path=os.environ.get("CABT_MODEL", "model.pth"),
        use_model=_env_bool("CABT_USE_MODEL", True),
        force_search=_env_bool("CABT_FORCE_SEARCH", False),
        search=search,
        profile=profile,
    )
