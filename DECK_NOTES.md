# Deck Notes: Consistency-Focused Dual-Type Evolution Line

## Archetype Choice
We selected a **dual-type evolution archetype** prioritizing consistency and bench
development. This fits the ISMCTS agent because:

1. **Evolution lines create decision trees** — the agent can evaluate whether to
   develop the bench vs. attack now, and ISMCTS naturally explores both paths.
2. **Dual-type coverage** mitigates hidden-information risk: if the opponent's
   face-down active has a type advantage, we can pivot to our secondary line.
3. **Moderate retreat costs** give the agent tactical flexibility to retreat and
   re-attack, creating richer action spaces for the tree search.
4. **Consistency Supporters** (draw/search) ensure the agent sees more of its own
   deck per turn, reducing variance in simulations — the belief model has better
   information about our own remaining cards.

## Card Count Rationale

| Category | Count | Reasoning |
|----------|-------|-----------|
| Primary Basic | 4 | Maximize chance of opening with it |
| Primary Stage 1 | 4 | Ensure evolution is available when needed |
| Primary Stage 2 | 3 | Win condition; 3 is enough with draw support |
| Secondary Basic | 4 | Backup line; also strong opener |
| Secondary Stage 1 | 3 | Evolution support for secondary |
| Secondary Stage 2 | 2 | Alternate win condition |
| Draw Supporters | 4 | Consistency engine |
| Search Supporters | 3 | Find evolution pieces |
| Rare Candy equiv. | 3 | Skip Stage 1 for speed |
| Switch/Retreat | 2 | Tactical flexibility |
| Pokémon Tool | 2 | Damage/protection boost |
| Ace Spec | 1 | One powerful utility card |
| Primary Energy | 14 | Fuel for main attacker |
| Secondary Energy | 4 | Fuel for backup line |
| **Total** | **60** | |

## Synergy Logic

- **Primary attacker** has an attack that scales with attached energy, rewarding
  the energy-heavy count (14). It serves as the main damage dealer.
- **Secondary line** covers a different type, providing an answer when the primary
  is at a type disadvantage or has been knocked out.
- **Rare Candy** accelerates Stage 2 deployment, letting the agent threaten high
  damage as early as turn 2 — the ISMCTS search will naturally discover this line.
- **Draw/search Supporters** thin the deck, increasing the probability that the
  belief model's determinizations are accurate (fewer unseen cards = less variance).
- **Switch cards** + low retreat costs enable the agent to cycle between attackers,
  which ISMCTS can exploit by finding optimal retreat-and-attack sequences.

## Energy Allocation

14 primary + 4 secondary = 18 energy (30% of deck). This is slightly above the
standard 15-card energy base because:
- Our primary attacker benefits from high energy attachment.
- Extra energy ensures we can recover from knockouts without stalling.
- ISMCTS can evaluate energy-attach decisions intelligently, so having options
  is better than being energy-starved.

## Why Not Aggressive Low-Retreat?

An aggressive single-type deck was considered but rejected because:
- Single-type decks are vulnerable to hidden-information type disadvantage.
- Low-retreat aggressive decks have fewer meaningful decision branches, reducing
  the advantage of ISMCTS over simple heuristics.
- Evolution decks have a higher skill ceiling that rewards forward-thinking search.
