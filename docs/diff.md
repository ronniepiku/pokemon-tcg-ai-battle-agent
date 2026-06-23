# Differences Between the Official Pokémon TCG Rules and the Simulator Behavior

The simulator used in this competition is designed for AI-vs-AI battles, and its rules and behavior may differ in some respects from those of the official Pokémon Trading Card Game. Below is a summary of the differences we are currently aware of.

Some attacks may not be selectable in the simulator even when they could be declared under the official rules In the official Pokémon TCG, there are cases where a player is allowed to declare an attack, but the effect cannot be fully resolved, and the turn simply ends after the attack declaration. In the simulator, such attacks may instead be treated as not selectable from the beginning. Examples include the following cases: Using an attack with an effect that puts a Basic Pokémon from the deck onto the Bench when there is no open Bench space Using an attack with an effect that draws cards when the player’s deck has 0 cards remaining Using an attack with an effect that interacts with the opponent’s hand when the opponent has 0 cards in hand Although the handling is different, we believe the end result is the same, and the impact on gameplay is minimal.

About Nullifying Zero, the attack of Mega Zygarde ex For Mega Zygarde ex’s attack, Nullifying Zero, under the official Pokémon TCG rules, the player using the attack may choose the order in which damage is assigned to the targets. In the simulator, however, the target order cannot be chosen, and coins are flipped automatically from left to right. This differs from the official rules, but since Knock Out processing is handled simultaneously, we believe this does not affect the competition.

Prize-taking order when both players’ Pokémon are Knocked Out at the same time When both players’ Pokémon are Knocked Out at the same time, the order of taking Prize cards differs between the official Pokémon TCG rules and the simulator.

Official Pokémon TCG order

The player whose turn is next chooses their Prize cards
The opposing player chooses their Prize cards
Both players take their Prize cards at the same time
The player whose turn is next puts a Pokémon into the Active Spot first
Simulator order used in this competition

The player whose turn is next chooses their Prize cards
That player takes their Prize cards
The opposing player chooses their Prize cards
The opposing player takes their Prize cards
The player whose turn is next puts a Pokémon into the Active Spot first
This is a different processing order from the official rules. However, in this competition, even if both players ultimately take all of their Prize cards, the result is treated as a draw, so we believe this does not affect match outcomes.

In this competition, please note that the simulator behavior will be treated as the correct behavior. If we identify any additional points that should be announced, we will share them in the Discussion forum as needed.
