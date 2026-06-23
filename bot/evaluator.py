"""Leaf evaluation for the planner.

Provides a single ``Evaluator`` that returns a scalar state value (and optional
action priors) for a search node. If a trained ``model.pth`` and torch are both
available it uses the neural network; otherwise it transparently falls back to
hand-crafted heuristics. Either way it never raises during normal play.
"""

from __future__ import annotations

import math
import os

from cg.api import Observation, SelectType

from . import heuristics
from .config import AgentConfig
from .deck_io import resolve_deck_path


class Evaluator:
    def __init__(self, cfg: AgentConfig, deck: list[int]) -> None:
        self.cfg = cfg
        self.deck = deck
        self._model = None
        self._model_tried = False

    # -- model loading ---------------------------------------------------- #

    def _ensure_model(self):
        if self._model_tried:
            return self._model
        self._model_tried = True
        if not self.cfg.use_model:
            return None
        path = resolve_deck_path(self.cfg.model_path)
        if not os.path.exists(path):
            return None
        try:
            import torch

            from .features import feature_dims
            from .model import build_model

            dims = feature_dims()
            model = build_model(dims)
            state_dict = torch.load(path, map_location="cpu")
            model.load_state_dict(state_dict)
            model.eval()
            self._model = model
        except Exception:
            self._model = None
        return self._model

    @property
    def has_model(self) -> bool:
        return self._ensure_model() is not None

    # -- evaluation ------------------------------------------------------- #

    def value(self, obs: Observation, your_index: int) -> float:
        """State value in [-1, 1] from ``your_index``'s perspective."""

        state = obs.current
        if state is not None and state.result >= 0:
            return heuristics.evaluate_state(state, your_index)

        model = self._ensure_model()
        if model is None or obs.select is None:
            return heuristics.evaluate_state(state, your_index)

        try:
            import torch

            from .features import (
                enumerate_actions,
                get_decoder_input,
                get_encoder_input,
            )
            from .model import eval_batch

            actions = enumerate_actions(obs)
            sv_e = get_encoder_input(obs, self.deck)
            sv_d = get_decoder_input(obs, actions)
            with torch.inference_mode():
                v, _ = eval_batch(model, sv_e, sv_d)
            if state.yourIndex != your_index:
                v = -v
            return float(v)
        except Exception:
            return heuristics.evaluate_state(state, your_index)

    def main_priors(self, obs: Observation) -> list[float]:
        """Normalised priors over the current MAIN options."""

        sel = obs.select
        n = len(sel.option)
        if n == 0:
            return []

        model = self._ensure_model()
        scores: list[float]
        if model is not None and sel.type == SelectType.MAIN:
            try:
                import torch

                from .features import (
                    enumerate_actions,
                    get_decoder_input,
                    get_encoder_input,
                )
                from .model import eval_batch

                actions = enumerate_actions(obs)
                sv_e = get_encoder_input(obs, self.deck)
                sv_d = get_decoder_input(obs, actions)
                with torch.inference_mode():
                    _, policy = eval_batch(model, sv_e, sv_d)
                scores = [policy[i] * 10.0 for i in range(min(n, len(policy)))]
                scores += [0.0] * (n - len(scores))
                return _softmax(scores)
            except Exception:
                pass

        # Heuristic priors: softmax over MAIN option scores (log-scaled).
        raw = [heuristics.score_main_option(obs, sel.option[i]) for i in range(n)]
        scores = [math.log1p(max(0.0, s)) for s in raw]
        return _softmax(scores)


def _softmax(scores: list[float]) -> list[float]:
    if not scores:
        return []
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    total = sum(exps) or 1.0
    return [e / total for e in exps]
