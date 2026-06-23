"""Top-level decision policy.

``decide`` is the single entry point used by ``main.agent``. It is defensive by
construction: every layer has a fallback, and a legal selection is *always*
returned even if parsing, search, or heuristics fail.

Decision flow for a normal turn:
    1. Initial call (no select)         -> submit the deck.
    2. Forced/empty selection           -> trivial legal answer.
    3. Action-count safety valve        -> end the turn if it is running away.
    4. MAIN decision + search enabled   -> MCTS plan (validated).
    5. Otherwise / on any failure       -> heuristic greedy (validated).
    6. Last resort                       -> minimal legal selection.
"""

from __future__ import annotations

import random
from typing import Optional

from cg.api import Observation, OptionType, SelectData, SelectType, to_observation_class

from . import heuristics, search
from .carddb import DB
from .config import AgentConfig, load_config
from .deck_io import read_deck
from .evaluator import Evaluator

# Above this many actions in a single turn we force an END to avoid pathological
# loops (e.g. a repeatable ability the heuristic keeps re-selecting).
_TURN_ACTION_LIMIT = 60


class AgentContext:
    """Persistent per-process state: config, deck, card DB, evaluator."""

    def __init__(self, cfg: Optional[AgentConfig] = None) -> None:
        self.cfg = cfg or load_config()
        random.seed(self.cfg.seed)
        DB.load()
        self.deck = read_deck(self.cfg.deck_path)
        self.evaluator = Evaluator(self.cfg, self.deck)


def _is_legal(sel: SelectData, picked: list[int]) -> bool:
    if picked is None:
        return False
    n = len(sel.option)
    if not (sel.minCount <= len(picked) <= sel.maxCount):
        return False
    if len(set(picked)) != len(picked):
        return False
    return all(isinstance(i, int) and 0 <= i < n for i in picked)


def _safe_default(sel: SelectData) -> list[int]:
    """Smallest guaranteed-legal selection: the first ``minCount`` options."""

    n = len(sel.option)
    k = max(0, min(sel.minCount, n))
    return list(range(k))


def _find_option(sel: SelectData, opt_type: OptionType) -> Optional[int]:
    for i, opt in enumerate(sel.option):
        if opt.type == opt_type:
            return i
    return None


def _emergency_from_dict(obs_dict: dict, deck: list[int]) -> list[int]:
    """Fallback used when the observation cannot be parsed into dataclasses."""

    sel = obs_dict.get("select") if isinstance(obs_dict, dict) else None
    if sel is None:
        return deck
    try:
        n = len(sel.get("option", []))
        k = max(0, min(int(sel.get("minCount", 0)), n))
        return list(range(k))
    except Exception:
        return [0]


def decide(obs_dict: dict, ctx: AgentContext) -> list[int]:
    """Return a legal list of option indices for ``obs_dict``."""

    try:
        obs: Observation = to_observation_class(obs_dict)
    except Exception:
        return _emergency_from_dict(obs_dict, ctx.deck)

    # 1. Initial deck submission.
    if obs.select is None:
        return list(ctx.deck)

    sel = obs.select
    try:
        n = len(sel.option)
        if n == 0:
            return []

        # 3. Safety valve against runaway turns.
        state = obs.current
        if state is not None and state.turnActionCount > _TURN_ACTION_LIMIT:
            end_idx = _find_option(sel, OptionType.END)
            if end_idx is not None:
                return [end_idx]

        # 4. Forward-model planning for our MAIN decisions.
        #
        # MCTS is only worthwhile with a trained value/policy net: with hand-
        # crafted leaves it merely ties the (already strong) heuristic while
        # adding latency. So we gate search on a loaded model, unless explicitly
        # forced for experimentation (CABT_FORCE_SEARCH=1).
        use_search = (
            ctx.cfg.search.enabled
            and state is not None
            and sel.type == SelectType.MAIN
            and (ctx.evaluator.has_model or ctx.cfg.force_search)
        )
        if use_search:
            planned = search.plan(obs, ctx.cfg.search, ctx.evaluator, ctx.deck)
            if planned is not None and _is_legal(sel, planned):
                return planned

        # 5. Heuristic greedy policy.
        picked = heuristics.choose(obs)
        if _is_legal(sel, picked):
            return picked

        # 6. Last resort.
        return _safe_default(sel)
    except Exception:
        return _safe_default(sel)
