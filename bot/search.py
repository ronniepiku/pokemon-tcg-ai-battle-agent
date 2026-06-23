"""Monte-Carlo Tree Search over the simulator's forward model.

Design choices for robustness and tractability:

* The tree branches only on *our* MAIN decisions. Forced sub-selections and the
  opponent's turn are folded into transitions by resolving them with the greedy
  heuristic policy (``heuristics.choose``). The opponent is therefore modelled as
  a fixed greedy player, turning the game into an MDP from our perspective.
* Because every node is evaluated from our (``your_index``) perspective and the
  opponent is part of the environment, no minimax sign-flipping is needed — we
  simply maximise expected value via PUCT.
* Hidden information (our deck order, prizes, opponent deck/hand/prizes) is
  sampled once per decision (a single determinisation), mirroring the reference.
* The whole search is wrapped by the caller in try/except and is wall-clock
  bounded; on any failure the caller falls back to the heuristic policy.
"""

from __future__ import annotations

import math
import random
import time
from typing import Optional

from cg.api import (
    Observation,
    SearchState,
    SelectType,
    search_begin,
    search_end,
    search_step,
)

from . import heuristics
from .carddb import DB
from .config import SearchConfig
from .evaluator import Evaluator


class _Child:
    __slots__ = ("select", "prob", "node")

    def __init__(self, select: list[int], prob: float) -> None:
        self.select = select
        self.prob = prob
        self.node: Optional["_Node"] = None


class _Node:
    __slots__ = ("state", "terminal", "value", "total", "visit", "children")

    def __init__(self, state: SearchState) -> None:
        self.state = state
        self.value = 0.0
        self.total = 0.0
        self.visit = 0
        self.children: list[_Child] = []
        st = state.observation.current
        self.terminal = st is None or st.result >= 0


def _filler_basic_id() -> int:
    """A valid Basic Pokémon card ID to fill predicted opponent cards with."""

    DB.load()
    if 1072 in DB.cards and DB.cards[1072].basic:
        return 1072
    for cid, cd in DB.cards.items():
        from cg.api import CardType

        if cd.basic and cd.cardType == CardType.POKEMON:
            return cid
    return 1072


class MCTS:
    def __init__(self, cfg: SearchConfig, evaluator: Evaluator, deck: list[int]) -> None:
        self.cfg = cfg
        self.evaluator = evaluator
        self.deck = deck
        self.your_index = 0
        self._filler = _filler_basic_id()

    # -- transitions ------------------------------------------------------ #

    def _advance_to_decision(self, state: SearchState) -> SearchState:
        """Step greedily until our MAIN decision or a terminal state."""

        guard = 0
        s = state
        while guard < 600:
            obs = s.observation
            st = obs.current
            if st is None or st.result >= 0 or obs.select is None:
                return s
            if obs.select.type == SelectType.MAIN and st.yourIndex == self.your_index:
                return s
            pick = heuristics.choose(obs)
            s = search_step(s.searchId, pick)
            guard += 1
        return s

    def _expand(self, node: _Node) -> float:
        """Evaluate a leaf and create its children; returns the leaf value."""

        obs = node.state.observation
        if node.terminal:
            node.value = self.evaluator.value(obs, self.your_index)
            return node.value

        node.value = self._leaf_value(node)
        priors = self.evaluator.main_priors(obs)
        n = len(obs.select.option)
        for i in range(n):
            p = priors[i] if i < len(priors) else 1.0 / max(1, n)
            node.children.append(_Child([i], p))
        return node.value

    def _leaf_value(self, node: _Node) -> float:
        """Leaf value: NN if a model is loaded, else greedy rollouts / static eval."""

        # A trained network is a stronger, far cheaper leaf estimate than rollouts.
        if self.evaluator.has_model or self.cfg.rollouts <= 0:
            return self.evaluator.value(node.state.observation, self.your_index)
        total = 0.0
        for _ in range(self.cfg.rollouts):
            total += self._rollout(node.state)
        return total / self.cfg.rollouts

    def _rollout(self, state: SearchState) -> float:
        """Play greedily (both sides) to the end; return our terminal value.

        ``search_step`` forks immutable states, so stepping from a leaf's id does
        not disturb later child expansion of that same leaf.
        """

        s = state
        for _ in range(self.cfg.rollout_max_steps):
            obs = s.observation
            st = obs.current
            if st is None or st.result >= 0 or obs.select is None:
                break
            s = search_step(s.searchId, heuristics.choose(obs))
        return self.evaluator.value(s.observation, self.your_index)

    # -- search ----------------------------------------------------------- #

    def run(self, root_state: SearchState, your_index: int) -> Optional[list[int]]:
        self.your_index = your_index
        root = _Node(self._advance_to_decision(root_state))
        if root.terminal or root.state.observation.select is None:
            return None
        self._expand(root)
        root.total += root.value
        root.visit += 1
        if not root.children:
            return None

        deadline = time.monotonic() + self.cfg.max_time_s
        for _ in range(self.cfg.simulations):
            if time.monotonic() >= deadline:
                break
            self._simulate(root)

        # Choose the most-visited child (robust); tie-break on mean value.
        best = max(
            root.children,
            key=lambda c: (
                c.node.visit if c.node else 0,
                (c.node.total / c.node.visit) if c.node and c.node.visit else -1e9,
            ),
        )
        return best.select

    def _simulate(self, root: _Node) -> None:
        path: list[_Node] = [root]
        node = root
        while True:
            child = self._select_child(node)
            if child is None:
                break
            if child.node is None:
                nxt = search_step(node.state.searchId, child.select)
                child.node = _Node(self._advance_to_decision(nxt))
                value = self._expand(child.node)
                path.append(child.node)
                self._backprop(path, value)
                return
            node = child.node
            path.append(node)
            if node.terminal:
                self._backprop(path, node.value)
                return

    def _select_child(self, node: _Node) -> Optional[_Child]:
        if not node.children:
            return None
        c_puct = self.cfg.c_puct
        sqrt_parent = math.sqrt(max(1, node.visit))
        parent_q = node.total / node.visit if node.visit else 0.0
        best_child = None
        best_score = -1e18
        for child in node.children:
            if child.node is not None and child.node.visit > 0:
                q = child.node.total / child.node.visit
                visit = child.node.visit
            else:
                q = parent_q  # First-play urgency: optimistic parent estimate.
                visit = 0
            u = c_puct * child.prob * sqrt_parent / (1 + visit)
            score = q + u
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def _backprop(self, path: list[_Node], value: float) -> None:
        for node in path:
            node.total += value
            node.visit += 1


def _sample_like_deck(deck: list[int], k: int) -> list[int]:
    """Sample ``k`` card IDs as a stand-in for an unknown pile of our deck."""

    if k <= 0:
        return []
    if k <= len(deck):
        return random.sample(deck, k)
    return [random.choice(deck) for _ in range(k)]


def build_search_args(obs: Observation, deck: list[int], filler: int) -> dict:
    """Construct sampled hidden-information arguments for ``search_begin``.

    We model the opponent as playing a deck similar to ours (a "mirror"
    determinisation). This is a far more realistic prior than vanilla filler
    Pokémon: it lets the search plan against a genuine threat instead of a
    passive opponent, which is what makes the lookahead worthwhile.
    """

    state = obs.current
    your_index = state.yourIndex
    me = state.players[your_index]
    opp = state.players[1 - your_index]

    your_deck = _sample_like_deck(deck, me.deckCount)
    your_prize = _sample_like_deck(deck, len(me.prize))

    # Mirror the opponent's hidden cards from our own deck list. Guarantee at
    # least one Basic Pokémon in the predicted opponent deck (setup requirement).
    opp_deck = _sample_like_deck(deck, opp.deckCount)
    if opp_deck and filler not in opp_deck:
        opp_deck[0] = filler

    opp_active = opp.active[0] if opp.active else None
    need_active = len(opp.active) > 0 and opp_active is None

    return {
        "your_deck": your_deck,
        "your_prize": your_prize,
        "opponent_deck": opp_deck,
        "opponent_prize": _sample_like_deck(deck, len(opp.prize)),
        "opponent_hand": _sample_like_deck(deck, opp.handCount),
        "opponent_active": [filler] if need_active else [],
    }


def plan(obs: Observation, cfg: SearchConfig, evaluator: Evaluator, deck: list[int]) -> Optional[list[int]]:
    """Run MCTS for the current observation; returns a selection or ``None``.

    Returns ``None`` (caller falls back to heuristics) if search is disabled,
    not applicable, or fails for any reason.
    """

    state = obs.current
    if state is None or obs.select is None:
        return None
    if obs.select.type != SelectType.MAIN:
        return None
    if len(obs.select.option) < cfg.min_options_for_search:
        return None

    mcts = MCTS(cfg, evaluator, deck)
    try:
        args = build_search_args(obs, deck, mcts._filler)
        root_state = search_begin(obs, **args)
        result = mcts.run(root_state, state.yourIndex)
    except Exception:
        result = None
    finally:
        try:
            search_end()
        except Exception:
            pass
    return result
