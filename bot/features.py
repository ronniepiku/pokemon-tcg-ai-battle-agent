"""Sparse feature extraction for the value/policy network.

This is a cleaned port of the reference encoder/decoder featurisation. It is
deliberately torch-free so it can be imported even when torch is unavailable;
``model.py`` consumes the resulting :class:`SparseVector` objects.

Feature layout constants depend on the card/attack tables and are computed
lazily from :data:`bot.carddb.DB`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

from cg.api import (
    AreaType,
    Card,
    Observation,
    OptionType,
    PlayerState,
    Pokemon,
    SelectContext,
    State,
)

from .carddb import DB

NUM_WORDS_ENCODER = 24
ENCODER_SIZE = 22000           # Input vocab size (exceeds card vocab).
DECODER_MAIN_FEATURE = 8       # Number of SelectContext.MAIN sub-features.
DECODER_ATTACK_OFFSET = 14     # First index of attack features in decoder vocab.


@dataclass
class FeatureDims:
    card_count: int
    attack_count: int
    decoder_card_offset: int
    decoder_size: int


def feature_dims() -> FeatureDims:
    """Compute decoder/encoder dimensions from the loaded card tables."""

    DB.load()
    card_count = DB.card_count
    attack_count = DB.attack_count
    decoder_card_offset = DECODER_ATTACK_OFFSET + attack_count
    decoder_size = decoder_card_offset + (
        1 + DECODER_MAIN_FEATURE + int(SelectContext.RECOVER_SPECIAL_CONDITION)
    ) * card_count
    return FeatureDims(card_count, attack_count, decoder_card_offset, decoder_size)


CardLike = Union[Card, Pokemon, None]


@dataclass
class SparseVector:
    """Input to ``torch.nn.EmbeddingBag`` (index/value/offset triplets)."""

    index: list[int] = field(default_factory=list)
    value: list[float] = field(default_factory=list)
    offset: list[int] = field(default_factory=list)
    pos: int = 0

    def add(self, index: int, value: float | int | bool) -> None:
        v = float(value)
        if v != 0.0:
            self.index.append(self.pos + index)
            self.value.append(v)

    def add_pos(self, pos: int) -> None:
        self.pos += pos

    def add_single(self, value: float | int | bool) -> None:
        v = float(value)
        if v != 0.0:
            self.index.append(self.pos)
            self.value.append(v)
        self.pos += 1

    def word_start(self) -> None:
        self.offset.append(len(self.index))


# --------------------------------------------------------------------------- #
# Encoder features
# --------------------------------------------------------------------------- #

def _add_card(sv: SparseVector, card: CardLike, card_count: int) -> None:
    if card is not None:
        sv.add(card.id, 1)
    sv.add_pos(card_count)


def _add_cards(sv: SparseVector, cards: list[Card] | None, value: float, card_count: int) -> None:
    if cards is not None:
        for card in cards:
            sv.add(card.id, value)
    sv.add_pos(card_count)


def _add_pokemon(sv: SparseVector, poke: Pokemon | None, card_count: int) -> None:
    if poke is None:
        sv.add_single(1)
        sv.add_pos(1 + 3 * card_count)
    else:
        sv.add_single(0)
        sv.add_single(poke.hp / 400)
        _add_card(sv, poke, card_count)
        _add_cards(sv, poke.tools, 1.0, card_count)
        _add_cards(sv, poke.energyCards, 0.5, card_count)


def _add_player(sv: SparseVector, ps: PlayerState, card_count: int) -> None:
    sv.add_single(ps.deckCount / 60)
    sv.add_single(len(ps.discard) / 60)
    sv.add_single(ps.handCount / 8)
    sv.add_single(len(ps.bench) / 5)
    sv.add(len(ps.prize), 1)
    sv.add_pos(7)
    sv.add_single(ps.poisoned)
    sv.add_single(ps.burned)
    sv.add_single(ps.asleep)
    sv.add_single(ps.paralyzed)
    sv.add_single(ps.confused)
    _add_cards(sv, ps.discard, 0.25, card_count)


def get_encoder_input(obs: Observation, your_deck: list[int]) -> SparseVector:
    dims = feature_dims()
    cc = dims.card_count
    state: State = obs.current
    your_index = state.yourIndex

    sv = SparseVector()
    for i in range(2):
        ps = state.players[i ^ your_index]
        for j in range(8):
            sv.word_start()
            pos = sv.pos
            if j < len(ps.bench):
                _add_pokemon(sv, ps.bench[j], cc)
            else:
                _add_pokemon(sv, None, cc)
            if j != 7:
                sv.pos = pos

    for i in range(2):
        ps = state.players[i ^ your_index]
        sv.word_start()
        if len(ps.active) > 0:
            _add_pokemon(sv, ps.active[0], cc)
        else:
            _add_pokemon(sv, None, cc)

    for i in range(2):
        ps = state.players[i ^ your_index]
        sv.word_start()
        _add_player(sv, ps, cc)

    sv.word_start()
    _add_cards(sv, state.players[your_index].hand, 0.25, cc)

    sv.word_start()
    for cid in your_deck:
        sv.add(cid, 0.25)
    sv.add_pos(cc)

    sv.word_start()
    _add_cards(sv, state.stadium, 1.0, cc)

    sv.word_start()
    sv.add_single(1)
    sv.add_single(state.turn / 10)
    sv.add_single(state.firstPlayer == your_index)
    return sv


# --------------------------------------------------------------------------- #
# Decoder features
# --------------------------------------------------------------------------- #

def _get_card(obs: Observation, area: AreaType, index: int, player_index: int) -> CardLike:
    ps = obs.current.players[player_index]
    if area == AreaType.DECK:
        return obs.select.deck[index]
    if area == AreaType.HAND:
        return ps.hand[index]
    if area == AreaType.DISCARD:
        return ps.discard[index]
    if area == AreaType.ACTIVE:
        return ps.active[index]
    if area == AreaType.BENCH:
        return ps.bench[index]
    if area == AreaType.PRIZE:
        return ps.prize[index]
    if area == AreaType.STADIUM:
        return obs.current.stadium[index]
    if area == AreaType.LOOKING:
        return obs.current.looking[index]
    return None


def _decoder_main(sv: SparseVector, feature_index: int, card: CardLike, dims: FeatureDims) -> None:
    if card is not None:
        sv.add(dims.decoder_card_offset + feature_index * dims.card_count + card.id, 1)


def _decoder_card_id(sv: SparseVector, context: SelectContext, card_id: int, dims: FeatureDims) -> None:
    sv.add(
        dims.decoder_card_offset
        + (DECODER_MAIN_FEATURE + int(context)) * dims.card_count
        + card_id,
        1,
    )


def _decoder_card(sv: SparseVector, context: SelectContext, card: CardLike, dims: FeatureDims) -> None:
    if card is not None:
        _decoder_card_id(sv, context, card.id, dims)


def get_decoder_input(obs: Observation, actions: list[list[int]]) -> SparseVector:
    dims = feature_dims()
    sv = SparseVector()
    your_index = obs.current.yourIndex
    ps = obs.current.players[your_index]
    context = obs.select.context

    for action in actions:
        sv.word_start()
        if len(action) == 0:
            sv.add(0, 1)
            continue
        for i in action:
            o = obs.select.option[i]
            t = o.type
            if t == OptionType.END:
                sv.add(1, 1)
            elif t == OptionType.YES:
                sv.add(2, 1)
            elif t == OptionType.NO:
                sv.add(3, 1)
            elif t == OptionType.SPECIAL_CONDITION:
                sv.add(4 + int(o.specialConditionType), 1)
            elif t == OptionType.NUMBER:
                sv.add(9 + min(o.number, 4), 1)
            elif t == OptionType.ATTACK:
                sv.add(DECODER_ATTACK_OFFSET + o.attackId, 1)
            elif t == OptionType.PLAY:
                _decoder_main(sv, 0, ps.hand[o.index], dims)
            elif t == OptionType.ATTACH:
                _decoder_main(sv, 1, _get_card(obs, o.area, o.index, your_index), dims)
                _decoder_main(sv, 2, _get_card(obs, o.inPlayArea, o.inPlayIndex, your_index), dims)
            elif t == OptionType.EVOLVE:
                _decoder_main(sv, 3, _get_card(obs, o.area, o.index, your_index), dims)
                _decoder_main(sv, 4, _get_card(obs, o.inPlayArea, o.inPlayIndex, your_index), dims)
            elif t == OptionType.ABILITY:
                _decoder_main(sv, 5, _get_card(obs, o.area, o.index, your_index), dims)
            elif t == OptionType.DISCARD:
                _decoder_main(sv, 6, _get_card(obs, o.area, o.index, your_index), dims)
            elif t == OptionType.RETREAT:
                _decoder_main(sv, 7, ps.active[0], dims)
            elif t == OptionType.CARD:
                _decoder_card(sv, context, _get_card(obs, o.area, o.index, o.playerIndex), dims)
            elif t == OptionType.TOOL_CARD:
                card = _get_card(obs, o.area, o.index, o.playerIndex)
                _decoder_card(sv, context, card.tools[o.toolIndex], dims)
            elif t in (OptionType.ENERGY_CARD, OptionType.ENERGY):
                card = _get_card(obs, o.area, o.index, o.playerIndex)
                _decoder_card(sv, context, card.energyCards[o.energyIndex], dims)
            elif t == OptionType.SKILL:
                _decoder_card_id(sv, context, o.cardId, dims)

    return sv


def enumerate_actions(obs: Observation, cap: int = 64) -> list[list[int]]:
    """Enumerate candidate selections (combinations of option indices).

    Mirrors the reference: fixed selection size ``maxCount`` over the option
    list, capped to ``cap`` combinations.
    """

    sel = obs.select
    n = len(sel.option)
    k = sel.maxCount
    if k <= 0 or n == 0:
        return [[]]
    indices = list(range(k))
    actions: list[list[int]] = []
    for _ in range(cap):
        actions.append(indices.copy())
        for i in range(k):
            index = k - i - 1
            if indices[index] < n - i - 1:
                indices[index] += 1
                for j in range(index + 1, k):
                    indices[j] = indices[j - 1] + 1
                break
        else:
            break
    return actions
