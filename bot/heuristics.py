"""Hand-crafted heuristics: state evaluation and legal-action selection.

These functions never raise on well-formed observations and always produce a
legal selection. They are used directly as the robust fallback policy and as the
leaf evaluation / action priors that guide the MCTS planner.

Value convention: ``evaluate_state`` returns a score in roughly ``[-1, 1]`` from
``your_index``'s perspective (positive = good for that player).
"""

from __future__ import annotations

from typing import Optional

from cg.api import (
    AreaType,
    CardType,
    Observation,
    Option,
    OptionType,
    Pokemon,
    SelectContext,
    SelectType,
    State,
)

from .carddb import DB

# Selection contexts where selecting *more* high-value cards is good
# (drawing, searching to hand, healing, putting Pokémon into play, ...).
_BENEFICIAL = {
    SelectContext.SETUP_ACTIVE_POKEMON,
    SelectContext.SETUP_BENCH_POKEMON,
    SelectContext.SWITCH,
    SelectContext.TO_ACTIVE,
    SelectContext.TO_BENCH,
    SelectContext.TO_FIELD,
    SelectContext.TO_HAND,
    SelectContext.HEAL,
    SelectContext.REMOVE_DAMAGE_COUNTER,
    SelectContext.REMOVE_DAMAGE_COUNTER_COUNT,
    SelectContext.DRAW_COUNT,
    SelectContext.EVOLVES_FROM,
    SelectContext.EVOLVES_TO,
}

# Contexts where we give up resources: select as few / least valuable as allowed.
_SACRIFICE = {
    SelectContext.DISCARD,
    SelectContext.TO_DECK,
    SelectContext.TO_DECK_BOTTOM,
    SelectContext.TO_PRIZE,
    SelectContext.DISCARD_ENERGY,
    SelectContext.DISCARD_ENERGY_CARD,
    SelectContext.DISCARD_TOOL_CARD,
    SelectContext.DISCARD_CARD_OR_ATTACHED_CARD,
    SelectContext.DETACH_FROM,
    SelectContext.DEVOLVE,
    SelectContext.TO_HAND_ENERGY,
    SelectContext.TO_DECK_ENERGY,
    SelectContext.NOT_MOVE,
}

# Contexts that target the opponent: prefer the Pokémon we are most likely to KO.
_OFFENSIVE_TARGET = {
    SelectContext.DAMAGE,
    SelectContext.DAMAGE_COUNTER,
    SelectContext.DAMAGE_COUNTER_ANY,
}

# YesNo contexts where "Yes" is the sensible default.
_YES_DEFAULT = {
    SelectContext.IS_FIRST,        # Going first suits a setup/wall deck.
    SelectContext.ACTIVATE,        # Effects offered to us are usually beneficial.
    SelectContext.FIRST_EFFECT,
    SelectContext.COIN_HEAD,       # Arbitrary 50/50.
}


# --------------------------------------------------------------------------- #
# Card valuation
# --------------------------------------------------------------------------- #

def card_value(card_id: Optional[int]) -> float:
    """Rough strategic value of holding/keeping a card (higher = keep)."""

    cd = DB.card(card_id)
    if cd is None:
        return 4.0
    if cd.cardType == CardType.POKEMON:
        if cd.megaEx or cd.ex:
            return 11.0          # Primary attackers / win conditions.
        if cd.stage1 or cd.stage2:
            return 9.0
        if cd.basic:
            return 7.5           # Basics enable setup and avoid mulligans.
        return 7.0
    if cd.cardType == CardType.SUPPORTER:
        return 6.0
    if cd.cardType in (CardType.ITEM, CardType.TOOL):
        return 5.5
    if cd.cardType == CardType.STADIUM:
        return 5.0
    if cd.cardType in (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY):
        return 3.0               # Plentiful; safest thing to discard.
    return 4.0


def _pokemon_hp(p: Optional[Pokemon]) -> int:
    return p.hp if p is not None else 0


def attack_damage(obs: Observation, attacker: Optional[Pokemon], attack_id: int) -> int:
    """Estimate an attack's damage including weakness/resistance."""

    atk = DB.attack(attack_id)
    if atk is None:
        return 0
    dmg = atk.damage
    state = obs.current
    if state is None or attacker is None:
        return dmg
    opp = state.players[1 - state.yourIndex]
    opp_active = opp.active[0] if opp.active else None
    if opp_active is None:
        return dmg
    attacker_cd = DB.card(attacker.id)
    target_cd = DB.card(opp_active.id)
    if attacker_cd is None or target_cd is None or dmg <= 0:
        return dmg
    if target_cd.weakness is not None and target_cd.weakness == attacker_cd.energyType:
        dmg *= 2
    if target_cd.resistance is not None and target_cd.resistance == attacker_cd.energyType:
        dmg = max(0, dmg - 30)
    return dmg


# --------------------------------------------------------------------------- #
# State evaluation
# --------------------------------------------------------------------------- #

def _count_pokemon(player) -> int:
    n = len(player.bench)
    if player.active and player.active[0] is not None:
        n += 1
    return n


def evaluate_state(state: State, your_index: int) -> float:
    """Heuristic value of ``state`` from ``your_index``'s perspective."""

    if state is None:
        return 0.0

    # Terminal result dominates everything.
    if state.result >= 0:
        if state.result == 2:
            return 0.0
        return 1.0 if state.result == your_index else -1.0

    me = state.players[your_index]
    opp = state.players[1 - your_index]

    score = 0.0

    # Prize race is the primary objective (first to 0 prizes wins).
    my_taken = 6 - len(me.prize)
    opp_taken = 6 - len(opp.prize)
    score += 0.16 * (my_taken - opp_taken)

    # Active Pokémon HP balance.
    my_active = me.active[0] if me.active else None
    opp_active = opp.active[0] if opp.active else None
    score += 0.20 * (_pokemon_hp(my_active) - _pokemon_hp(opp_active)) / 350.0

    # Having an Active Pokémon at all matters (no Active = loss condition).
    score += 0.10 * ((1 if my_active else 0) - (1 if opp_active else 0))

    # Board development.
    score += 0.04 * (_count_pokemon(me) - _count_pokemon(opp))

    # Resources in hand.
    score += 0.02 * (me.handCount - opp.handCount)

    # Special conditions on our Active are bad; on theirs are good.
    my_cond = me.poisoned + me.burned + me.asleep + me.paralyzed + me.confused
    opp_cond = opp.poisoned + opp.burned + opp.asleep + opp.paralyzed + opp.confused
    score += 0.03 * (opp_cond - my_cond)

    # Energy attached to our active attacker (readiness).
    if my_active is not None:
        score += 0.02 * min(len(my_active.energies), 4)

    return max(-0.98, min(0.98, score))


# --------------------------------------------------------------------------- #
# Option scoring
# --------------------------------------------------------------------------- #

def _hand_card_id(state: State, index: int) -> Optional[int]:
    me = state.players[state.yourIndex]
    if me.hand is not None and 0 <= index < len(me.hand):
        return me.hand[index].id
    return None


def _option_target_card(obs: Observation, option: Option) -> Optional[int]:
    """Best-effort lookup of the card ID a CARD-type option refers to."""

    state = obs.current
    if state is None:
        return None
    area = option.area
    idx = option.index
    pidx = option.playerIndex if option.playerIndex is not None else state.yourIndex
    if area is None or idx is None:
        return None
    try:
        if area == AreaType.DECK and obs.select and obs.select.deck is not None:
            return obs.select.deck[idx].id
        player = state.players[pidx]
        if area == AreaType.HAND and player.hand is not None:
            return player.hand[idx].id
        if area == AreaType.DISCARD:
            return player.discard[idx].id
        if area == AreaType.ACTIVE and player.active and player.active[idx] is not None:
            return player.active[idx].id
        if area == AreaType.BENCH:
            return player.bench[idx].id
        if area == AreaType.PRIZE and player.prize[idx] is not None:
            return player.prize[idx].id
        if area == AreaType.STADIUM and state.stadium:
            return state.stadium[idx].id
        if area == AreaType.LOOKING and state.looking:
            card = state.looking[idx]
            return card.id if card is not None else None
    except (IndexError, TypeError):
        return None
    return None


def _retreat_score(state: State) -> float:
    me = state.players[state.yourIndex]
    active = me.active[0] if me.active else None
    if active is None or not me.bench:
        return -100.0
    danger = me.poisoned or me.burned or me.asleep or me.paralyzed or me.confused
    low_hp = active.maxHp > 0 and active.hp <= active.maxHp * 0.4
    if danger or low_hp:
        return 760.0
    return -50.0


def score_main_option(obs: Observation, option: Option) -> float:
    """Score a SelectType.MAIN option. Higher = do this first."""

    state = obs.current
    if state is None:
        return 0.0
    me = state.players[state.yourIndex]
    opp = state.players[1 - state.yourIndex]
    active = me.active[0] if me.active else None
    t = option.type

    if t == OptionType.ATTACK:
        dmg = attack_damage(obs, active, option.attackId)
        opp_active = opp.active[0] if opp.active else None
        if opp_active is not None and dmg >= opp_active.hp:
            return 10000.0 + dmg          # Secure the KO / prize.
        return 120.0 + dmg
    if t == OptionType.EVOLVE:
        return 900.0
    if t == OptionType.ABILITY:
        return 840.0
    if t == OptionType.ATTACH:
        return 800.0
    if t == OptionType.PLAY:
        cd = DB.card(_hand_card_id(state, option.index))
        if cd is None:
            return 600.0
        if cd.cardType == CardType.ITEM:
            return 700.0
        if cd.cardType == CardType.SUPPORTER:
            return 680.0
        if cd.cardType == CardType.TOOL:
            return 620.0
        if cd.cardType == CardType.POKEMON:
            return 600.0
        if cd.cardType == CardType.STADIUM:
            return 560.0
        return 600.0
    if t == OptionType.RETREAT:
        return _retreat_score(state)
    if t == OptionType.DISCARD:
        return 40.0
    if t == OptionType.END:
        return 1.0
    return 10.0


def _select_priority(obs: Observation, option: Option, context: SelectContext) -> float:
    """How much we want to *include* this option in a multi/single selection."""

    state = obs.current
    t = option.type

    # Simple option types first.
    if t == OptionType.YES:
        return 1.0 if context in _YES_DEFAULT else 0.0
    if t == OptionType.NO:
        return 1.0 if context not in _YES_DEFAULT else 0.0
    if t == OptionType.END:
        return -1.0
    if t == OptionType.NUMBER:
        n = option.number if option.number is not None else 0
        # Draw/heal/remove: more is better; otherwise prefer the minimum.
        if context in _BENEFICIAL:
            return float(n)
        return float(-n)
    if t == OptionType.SPECIAL_CONDITION:
        return 1.0
    if t == OptionType.ATTACK:
        active = None
        if state is not None:
            me = state.players[state.yourIndex]
            active = me.active[0] if me.active else None
        dmg = attack_damage(obs, active, option.attackId)
        opp_active = None
        if state is not None:
            opp = state.players[1 - state.yourIndex]
            opp_active = opp.active[0] if opp.active else None
        if opp_active is not None and dmg >= opp_active.hp:
            return 10000.0 + dmg
        return 100.0 + dmg

    # Card-like options: value the underlying card, considering context.
    cid = _option_target_card(obs, option)
    val = card_value(cid)

    if context in _OFFENSIVE_TARGET:
        # Target the opponent Pokémon with the least HP (easiest KO).
        if state is not None:
            target_hp = _target_hp(obs, option)
            return 1000.0 - target_hp
        return val
    if context == SelectContext.SETUP_ACTIVE_POKEMON or context == SelectContext.TO_ACTIVE:
        # Prefer a sturdy, ready attacker in the Active Spot.
        cd = DB.card(cid)
        return float(cd.hp) if cd is not None else val
    if context in _SACRIFICE:
        # Selecting means giving it up: prefer the least valuable card.
        return -val
    # Beneficial / default: prefer the most valuable card.
    return val


def _target_hp(obs: Observation, option: Option) -> int:
    state = obs.current
    if state is None:
        return 0
    pidx = option.playerIndex if option.playerIndex is not None else 1 - state.yourIndex
    try:
        player = state.players[pidx]
        if option.area == AreaType.ACTIVE and player.active:
            p = player.active[option.index]
            return p.hp if p is not None else 0
        if option.area == AreaType.BENCH:
            return player.bench[option.index].hp
    except (IndexError, TypeError):
        return 0
    return 0


# --------------------------------------------------------------------------- #
# Top-level selection
# --------------------------------------------------------------------------- #

def _pick_count(context: SelectContext, k_min: int, k_max: int, n: int) -> int:
    """Decide how many options to select for this context."""

    upper = min(k_max, n)
    if upper <= k_min:
        return max(0, min(k_min, n))
    if context in _BENEFICIAL:
        return upper                 # Take the maximum allowed of good things.
    return max(0, min(k_min, n))     # Give up the minimum of bad things.


def choose(obs: Observation) -> list[int]:
    """Return a legal selection for ``obs`` using heuristics only.

    Guarantees: result length in ``[minCount, maxCount]``, indices unique and in
    range. Never raises for a well-formed observation.
    """

    sel = obs.select
    if sel is None:
        return []

    n = len(sel.option)
    k_min = max(0, sel.minCount)
    k_max = sel.maxCount

    if n == 0:
        return []

    if sel.type == SelectType.MAIN:
        scores = [score_main_option(obs, sel.option[i]) for i in range(n)]
        best = max(range(n), key=lambda i: scores[i])
        return [best]

    context = sel.context
    priorities = [_select_priority(obs, sel.option[i], context) for i in range(n)]
    order = sorted(range(n), key=lambda i: priorities[i], reverse=True)

    count = _pick_count(context, k_min, k_max, n)
    if count <= 0:
        return []
    chosen = sorted(order[:count])
    return chosen
