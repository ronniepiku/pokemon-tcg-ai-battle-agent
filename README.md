# pokemon-tcg-ai-battle-agent

An AI agent for the **Pokémon TCG AI Battle Challenge** (Kaggle, simulation
track). The agent plays full games through the `cabt` simulator's
`agent(obs_dict) -> list[int]` interface, choosing a legal action at every
decision point and submitting a 60-card deck at game start.

## Design at a glance

The agent is **robust-first, strong-second**:

* **Heuristic policy (default).** A typed, hand-crafted policy that evaluates the
  game state (prize race, board HP, conditions, readiness) and scores every legal
  option for the current selection context. It always returns a legal selection
  and never crashes on a well-formed observation. It wins ~100% vs a random
  opponent.
* **Forward-model MCTS (auto-upgrade).** When a trained `model.pth` is present,
  the agent runs Monte-Carlo Tree Search over the simulator's real forward model
  (`search_begin` / `search_step`), using the network for leaf value and action
  priors. The tree branches only on *our* MAIN decisions; forced sub-selections
  and the opponent's turn are folded into greedy transitions (opponent-as-MDP).
* **Value/policy network + self-play training.** An AlphaZero-style Transformer
  value/policy net, trainable via self-play on a local GPU (RTX 3060).

> **Why gate MCTS on a trained model?** Empirically, with hand-crafted leaf
> evaluation (static eval *or* greedy rollouts) MCTS only *ties* the heuristic
> while adding latency — the greedy heuristic is already close to the rollout
> policy. AlphaZero-style search only surpasses a strong heuristic once it has a
> *learned* value/policy. So the default ships the fast, strong heuristic and
> automatically switches to NN-guided MCTS when `model.pth` is available.

### Project structure

```python
main.py                 # Kaggle entry point: agent(obs_dict) -> list[int]
deck.csv                # 60-card deck (one card ID per line)
bot/
  config.py             # profiles, seeds, search budgets, env overrides
  deck_io.py            # deck loading (handles /kaggle_simulations/agent/)
  carddb.py             # cached card/attack metadata
  heuristics.py         # state evaluation + legal-action selection (all contexts)
  features.py           # sparse encoder/decoder features (torch-free)
  model.py              # Transformer value/policy network (torch)
  evaluator.py          # leaf eval: NN if model present, else heuristics
  search.py             # forward-model MCTS planner
  policy.py             # robust top-level decide() with layered fallbacks
training/
  train.py              # AlphaZero-style self-play training loop
tests/
  validate.py           # self-play validation + strength gate
build_submission.py     # assembles dist/ and packs submission.tar.gz
```

The deck is the proven Water **Mega Abomasnow ex** archetype: a 350 HP /
200-damage Stage-1 wall (4 Snover → 4 Mega Abomasnow ex), Kyogre as a secondary
attacker, a search/draw engine (Ultra Ball, Mega Signal, Team Rocket's Petrel,
Lillie's Determination, Secret Box), Powerglass for energy acceleration from the
discard, Surfing Beach for free switching, and 31 Water Energy. Exactly one ACE
SPEC (Secret Box) keeps the list legal.

## Run locally (Windows or Linux)

Requirements: Python ≥ 3.10. The native simulator (`cg/cg.dll`, `cg/libcg.so`)
is already in the repo. `numpy` and (for training) `torch` with CUDA.

```bash
# One self-play game with the real agent (no kaggle_environments needed):
python main.py

# Validation: 20 games vs a random opponent + a strength gate.
python -m tests.validate --games 20 --opponent random

# Self-play sanity (agent vs agent):
python -m tests.validate --games 10 --opponent self
```

### Configuration (reproducible via env vars)

| Variable            | Meaning                                  | Default     |
| ------------------- | ---------------------------------------- | ----------- |
| `CABT_PROFILE`      | `fast` (CI) or `ladder` (strong)         | `ladder`    |
| `CABT_SEED`         | RNG seed                                 | `0`         |
| `CABT_SEARCH`       | enable MCTS (`0`/`1`)                    | `1`         |
| `CABT_SIMS`         | MCTS simulations per decision            | profile     |
| `CABT_TIME`         | wall-clock budget per decision (s)       | profile     |
| `CABT_USE_MODEL`    | load `model.pth` if present              | `1`         |
| `CABT_FORCE_SEARCH` | run MCTS even without a trained model    | `0`         |
| `CABT_MODEL`        | path to model weights                    | `model.pth` |
| `CABT_DECK`         | path to deck file                        | `deck.csv`  |

```bash
# Fast heuristic-only iteration:
CABT_PROFILE=fast CABT_SEARCH=0 python -m tests.validate --games 50
```

## Train the value/policy network (RTX 3060)

```bash
# Quick smoke test (a few minutes):
python -m training.train --iterations 1 --selfplay 4 --eval 4 --sims 8

# Real run (long; writes model.pth + out/modelN.pth checkpoints):
python -m training.train --iterations 30 --selfplay 200 --eval 50 --sims 32 --out model.pth
```

Once `model.pth` exists, the agent (and the bundle) automatically uses NN-guided
MCTS — no code changes required.

## Create the submission

```bash
python build_submission.py          # stages dist/ then runs:
                                     #   tar -czvf submission.tar.gz *
```

The bundle has `main.py` and `deck.csv` at the **top level**, plus the `cg`
native package and the `bot` package (and `model.pth` if present). Upload
`submission.tar.gz` to Kaggle.

To package manually (the exact required command), from a staged directory:

```bash
cd dist && tar -czvf ../submission.tar.gz *
```

## Key design decisions & tradeoffs

* **Robustness over brittle complexity.** `policy.decide` is wrapped in layered
  fallbacks (parse → safety valve → search → heuristic → minimal legal answer);
  it always returns a legal selection. Search and model loading are fully
  optional and fail closed to the heuristic.
* **Opponent-as-MDP search.** Modelling the opponent as a fixed greedy player and
  determinising hidden info as a *mirror* of our own deck makes the lookahead
  realistic and tractable, instead of planning against passive filler Pokémon.
* **Hidden information & stochasticity.** Hidden piles are sampled once per
  decision (single determinisation); coin flips/shuffles are resolved by the real
  forward model.
* **Reproducibility.** Seeds and all search budgets are configurable; `fast` and
  `ladder` profiles trade speed for strength.

## Known limitations & next improvements

* Without a trained `model.pth`, play is heuristic (strong, but no deep
  lookahead). Training the net is the primary strength upgrade.
* Single determinisation per decision; multiple determinisations (or particle
  filtering on the opponent's revealed cards) would reduce variance.
* The opponent model is greedy and assumes a mirror deck; tracking the opponent's
  revealed cards to refine the prediction would sharpen the search.
* Heuristic option scoring is archetype-aware but generic across contexts; a few
  rare selection contexts fall back to sensible defaults rather than bespoke
  logic.
