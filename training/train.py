"""AlphaZero-style self-play training for the value/policy network.

This is an adaptation of the reference training loop onto the modular
``bot.features`` / ``bot.model`` code. Running it produces ``model.pth``, which
the inference agent automatically picks up to enable NN-guided MCTS.

Training-time MCTS branches on *every* selection (using the network's policy),
which yields value targets for all state types — exactly what the inference-time
value head needs.

Quick smoke test (CPU/GPU, a few minutes):
    python -m training.train --iterations 1 --selfplay 4 --eval 4 --sims 8

Real training run (RTX 3060, long):
    python -m training.train --iterations 30 --selfplay 200 --eval 50 --sims 32 \
        --out model.pth
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from dataclasses import dataclass

sys.path.insert(0, ".")

import torch  # noqa: E402

from cg.game import battle_finish, battle_select, battle_start  # noqa: E402
from cg.api import to_observation_class  # noqa: E402

from bot import heuristics  # noqa: E402
from bot.carddb import DB  # noqa: E402
from bot.deck_io import read_deck  # noqa: E402
from bot.features import (  # noqa: E402
    SparseVector,
    enumerate_actions,
    feature_dims,
    get_decoder_input,
    get_encoder_input,
)
from bot.model import build_model, eval_batch  # noqa: E402
from cg.api import (  # noqa: E402
    SearchState,
    search_begin,
    search_end,
    search_step,
)


# --------------------------------------------------------------------------- #
# Self-play MCTS (NN guided) with training-target collection
# --------------------------------------------------------------------------- #

@dataclass
class LearnSample:
    value: float
    policy: list[float]
    sv_enc: SparseVector
    sv_dec: SparseVector


class Child:
    __slots__ = ("select", "prob", "node")

    def __init__(self, select: list[int], prob: float) -> None:
        self.select = select
        self.prob = prob
        self.node: "Node | None" = None


class Node:
    __slots__ = ("value", "total", "visit", "parent", "children", "state")

    def __init__(self, parent: "Node | None", state: SearchState) -> None:
        self.value = -2.0
        self.total = 0.0
        self.visit = 0
        self.parent = parent
        self.children: list[Child] = []
        self.state = state

    def backprop(self, value: float) -> None:
        node: "Node | None" = self
        while node is not None:
            node.total += value
            node.visit += 1
            node = node.parent


_FILLER_BASIC = 1072


def create_node(parent, state, your_index, your_deck, model):
    node = Node(parent, state)
    obs = state.observation
    st = obs.current
    if st.result >= 0:
        node.value = 0.0 if st.result == 2 else (1.0 if st.result == your_index else -1.0)
        node.backprop(node.value)
        return node, None

    actions = enumerate_actions(obs)
    sv_enc = get_encoder_input(obs, your_deck)
    sv_dec = get_decoder_input(obs, actions)
    value, policy = eval_batch(model, sv_enc, sv_dec)
    v = value if st.yourIndex == your_index else -value
    node.value = v
    node.backprop(v)

    total = 0.0
    for i in range(len(policy)):
        p = math.exp(policy[i] * 10.0)
        node.children.append(Child(actions[i], p))
        total += p
    for c in node.children:
        c.prob /= total or 1.0
    return node, LearnSample(value, policy, sv_enc, sv_dec)


def mcts_agent(obs_dict, your_deck, model, sims):
    obs = to_observation_class(obs_dict)
    st = obs.current
    your_index = st.yourIndex
    me = st.players[your_index]
    opp = st.players[1 - your_index]
    active = opp.active

    state = search_begin(
        obs,
        your_deck=random.sample(your_deck, me.deckCount),
        your_prize=random.sample(your_deck, len(me.prize)),
        opponent_deck=[_FILLER_BASIC] * opp.deckCount,
        opponent_prize=[1] * len(opp.prize),
        opponent_hand=[1] * opp.handCount,
        opponent_active=[_FILLER_BASIC] if len(active) > 0 and active[0] is None else [],
    )
    root, sample = create_node(None, state, your_index, your_deck, model)

    for _ in range(sims):
        current = root
        while True:
            best = None
            best_v = -1e18
            c = 0.4 * math.sqrt(current.visit)
            for child in current.children:
                visit = 0
                if child.node is None:
                    v = current.total / current.visit
                else:
                    v = child.node.total / child.node.visit
                    visit = child.node.visit
                if current.state.observation.current.yourIndex != your_index:
                    v = -v
                v += c * child.prob / (1 + visit)
                if v > best_v:
                    best_v = v
                    best = child
            if best is None:
                break
            if best.node is None:
                nxt = search_step(current.state.searchId, best.select)
                best.node, _ = create_node(current, nxt, your_index, your_deck, model)
                break
            current = best.node
            if current.state.observation.current.result >= 0:
                current.backprop(current.value)
                break

    max_child = None
    max_visit = -1
    min_value = 10.0
    for child in root.children:
        if child.node is not None:
            if child.node.visit > max_visit:
                max_child = child
                max_visit = child.node.visit
            v = child.node.total / child.node.visit
            min_value = min(min_value, v)

    if sample is not None:
        sample.value = root.total / root.visit
        for i, child in enumerate(root.children):
            base = sample.value
            if child.node is None:
                v = min_value - base - 0.03
            else:
                v = child.node.total / child.node.visit - base
            sample.policy[i] = max(-1.0, min(1.0, v))

    search_end()
    if max_child is None:
        return heuristics.choose(obs), sample
    return max_child.select, sample


# --------------------------------------------------------------------------- #
# Training plumbing
# --------------------------------------------------------------------------- #

class LearnInput:
    def __init__(self) -> None:
        self.index: list[int] = []
        self.value: list[float] = []
        self.offset: list[int] = []

    def add(self, sv: SparseVector) -> None:
        count = len(self.index)
        self.index.extend(sv.index)
        self.value.extend(sv.value)
        for o in sv.offset:
            self.offset.append(o + count)


def random_agent(obs_dict):
    obs = to_observation_class(obs_dict)
    n = len(obs.select.option)
    return sorted(random.sample(range(n), obs.select.maxCount)) if n else []


def evaluate(model, deck, games, sims, device):
    model.eval()
    results = [0, 0, 0]
    with torch.inference_mode():
        for i in range(games):
            obs, sd = battle_start(deck, deck)
            if sd.errorPlayer >= 0:
                raise SystemExit(f"Deck error type {sd.errorType}.")
            your_index = i % 2
            while obs["current"]["result"] < 0:
                if obs["current"]["yourIndex"] == your_index:
                    sel, _ = mcts_agent(obs, deck, model, sims)
                else:
                    sel = random_agent(obs)
                obs = battle_select(sel)
            r = obs["current"]["result"]
            battle_finish()
            results[2 if r == 2 else (0 if r == your_index else 1)] += 1
    decisive = results[0] + results[1]
    wr = 100 * results[0] // decisive if decisive else 0
    return wr, results


def selfplay(model, deck, games, sims):
    model.eval()
    samples: list[LearnSample] = []
    with torch.inference_mode():
        for _ in range(games):
            obs, _ = battle_start(deck, deck)
            per_player: list[list[LearnSample]] = [[], []]
            while obs["current"]["result"] < 0:
                sel, sample = mcts_agent(obs, deck, model, sims)
                if sample is not None:
                    per_player[obs["current"]["yourIndex"]].append(sample)
                obs = battle_select(sel)
            result = obs["current"]["result"]
            battle_finish()
            for i in range(2):
                lam = 0.9
                value = 1.0 if i == result else -1.0
                for s in reversed(per_player[i]):
                    label = (value + s.value) * 0.5
                    value = value * lam + s.value * (1.0 - lam)
                    s.value = label
                    samples.append(s)
    return samples


def train_epoch(model, optimizer, samples, device, batch_size=128):
    loss_enc_fn = torch.nn.HuberLoss(delta=0.2)
    loss_dec_fn = torch.nn.HuberLoss(reduction="none", delta=0.1)
    model.train()
    random.shuffle(samples)
    n_batches = len(samples) // batch_size
    for b in range(n_batches):
        in_enc, in_dec = LearnInput(), LearnInput()
        mask: list[float] = []
        lab_enc: list[float] = []
        lab_dec: list[float] = []
        for j in range(b * batch_size, (b + 1) * batch_size):
            s = samples[j]
            in_enc.add(s.sv_enc)
            in_dec.add(s.sv_dec)
            lab_enc.append(s.value)
            lab_dec.extend(s.policy)
            mask.extend([1.0] * len(s.policy))
            for _ in range(64 - len(s.policy)):
                mask.append(0.0)
                lab_dec.append(0.0)
                in_dec.offset.append(len(in_dec.index))

        mask_t = torch.tensor(mask, dtype=torch.float32, device=device).view(batch_size, -1)
        le = torch.tensor(lab_enc, dtype=torch.float32, device=device).view(batch_size, -1)
        ld = torch.tensor(lab_dec, dtype=torch.float32, device=device).view(batch_size, -1)
        optimizer.zero_grad()
        out_enc, out_dec = model(
            torch.tensor(in_enc.index, dtype=torch.int32, device=device),
            torch.tensor(in_enc.value, dtype=torch.float32, device=device),
            torch.tensor(in_enc.offset, dtype=torch.int32, device=device),
            torch.tensor(in_dec.index, dtype=torch.int32, device=device),
            torch.tensor(in_dec.value, dtype=torch.float32, device=device),
            torch.tensor(in_dec.offset, dtype=torch.int32, device=device),
        )
        loss = loss_enc_fn(out_enc, le) + (loss_dec_fn(out_dec, ld) * mask_t).sum() / batch_size
        loss.backward()
        optimizer.step()


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-play training.")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--selfplay", type=int, default=100)
    parser.add_argument("--eval", type=int, default=40)
    parser.add_argument("--sims", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deck", default="deck.csv")
    parser.add_argument("--out", default="model.pth")
    parser.add_argument("--ckpt-dir", default="out")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    DB.load()
    deck = read_deck(args.deck)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | deck: {len(deck)} cards")

    dims = feature_dims()
    model = build_model(dims).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    for it in range(args.iterations):
        t0 = time.monotonic()
        wr, res = evaluate(model, deck, args.eval, args.sims, device)
        print(f"[iter {it}] win-rate vs random: {wr}% {res}")
        samples = selfplay(model, deck, args.selfplay, args.sims)
        print(f"[iter {it}] collected {len(samples)} samples")
        if samples:
            train_epoch(model, optimizer, samples, device)
        ckpt = os.path.join(args.ckpt_dir, f"model{it}.pth")
        torch.save(model.state_dict(), ckpt)
        torch.save(model.state_dict(), args.out)
        print(f"[iter {it}] saved {ckpt} and {args.out} ({time.monotonic() - t0:.1f}s)")

    print("Training complete.")


if __name__ == "__main__":
    main()
