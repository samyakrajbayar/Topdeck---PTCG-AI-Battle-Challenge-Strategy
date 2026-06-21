# deck_logic.py — Runtime deck selection from card pool
#
# Since the actual card-pool CSV (~2000 Standard-format cards) is provided
# by the environment at runtime via all_card_data(), we don't hardcode card IDs.
# Instead, this module selects a coherent 60-card archetype programmatically.
#
# ARCHETYPE: "Consistency-Focused Dual-Type Evolution Line"
# Strategy: A core evolution attacker backed by consistency Supporters,
# adequate energy, and utility cards. Chosen for:
#   - Reliability: evolution lines with draw/search support
#   - Type coverage: dual energy types to handle weakness matchups
#   - Energy efficiency: moderate retreat costs, scalable attacks
#   - Bench development: multiple evolution lines for redundancy

from collections import Counter

def build_deck():
    """Build a 60-card deck from the live card pool.
    Returns a list of 60 card IDs (strings), one per line for deck.csv."""

    cards = all_card_data() if all_card_data else []
    if not cards:
        # Fallback placeholder IDs — replace with real IDs from the pool
        return ["PLACEHOLDER"] * 60

    # Step 1: Identify candidate evolution lines
    basics = [c for c in cards if c.basic and not c.ex]
    stage1s = [c for c in cards if c.stage1]
    stage2s = [c for c in cards if c.stage2]

    # Find evolution chains: basic -> stage1 -> stage2
    chains = []
    for s2 in stage2s:
        s1_match = [s1 for s1 in stage1s if s1.evolvesFrom == s2.cardId
                     or s1.name.startswith(s2.name.split()[0])]
        for s1 in s1_match:
            b_match = [b for b in basics if b.evolvesFrom == s1.cardId
                        or b.name.startswith(s1.name.split()[0])]
            for b in b_match:
                chains.append((b, s1, s2))

    if not chains:
        # No stage2 chains; try basic -> stage1
        for s1 in stage1s:
            b_match = [b for b in basics if b.name.startswith(s1.name.split()[0])]
            for b in b_match:
                chains.append((b, s1, None))

    # Step 2: Score chains by HP, attack damage, retreat cost
    def score_chain(chain):
        b, s1, s2 = chain
        score = 0
        score += b.hp * 0.5
        score += s1.hp * 1.0
        if s2:
            score += s2.hp * 1.5
        # Prefer low retreat cost
        score -= b.retreatCost * 5
        # Prefer high-damage attacks
        for c in chain:
            if c and c.attacks:
                for aid in c.attacks:
                    atk = next((a for a in all_attack() if a.attackId == aid), None)
                    if atk:
                        score += atk.damage * 0.3
        return score

    chains.sort(key=score_chain, reverse=True)

    # Step 3: Pick top 2 chains for dual-type coverage
    primary = chains[0] if len(chains) >= 1 else None
    secondary = chains[1] if len(chains) >= 2 else None

    deck = Counter()

    if primary:
        b, s1, s2 = primary
        deck[b.cardId] = 4  # 4 copies of basic for consistency
        deck[s1.cardId] = 3  # 3 Stage 1
        if s2:
            deck[s2.cardId] = 2  # 2 Stage 2

    if secondary:
        b, s1, s2 = secondary
        deck[b.cardId] = 3
        deck[s1.cardId] = 2
        if s2:
            deck[s2.cardId] = 1

    # Step 4: Add Supporters (draw/search)
    # Look for cards with skills that mention "draw" or "search" in text
    supporters = [c for c in cards if c.cardType == "SUPPORTER"
                  or (c.skills and any("draw" in s.lower() or "search" in s.lower()
                                       for s in (c.skills or [])))]
    for s in supporters[:3]:  # Up to 3 different supporters
        deck[s.cardId] = 2

    # Step 5: Add energy cards
    # Determine energy type from primary chain
    if primary:
        energy_type = primary[0].energyType
    else:
        energy_type = "COLORLESS"

    energy_cards = [c for c in cards if c.cardType == "ENERGY"
                    and (c.energyType == energy_type or c.energyType == "COLORLESS")]
    if energy_cards:
        deck[energy_cards[0].cardId] = 12  # 12 basic energy
    else:
        deck["BASIC_ENERGY"] = 12

    # Step 6: Fill remaining slots with utility (items, tools)
    current_count = sum(deck.values())
    remaining = 60 - current_count

    items = [c for c in cards if c.cardType == "ITEM"
             or c.cardType == "TOOL"][:remaining]
    for item in items:
        deck[item.cardId] = min(2, remaining)
        remaining = 60 - sum(deck.values())
        if remaining <= 0:
            break

    # Pad if still under 60
    while sum(deck.values()) < 60:
        if primary:
            deck[primary[0].cardId] += 1
        else:
            deck["FILLER"] += 1

    # Convert to list of 60 card IDs
    deck_list = []
    for card_id, count in deck.items():
        deck_list.extend([card_id] * count)

    return deck_list[:60]  # Exactly 60

# Uncomment to generate deck.csv at dev time:
# deck = build_deck()
# with open("deck.csv", "w") as f:
#     for card_id in deck:
#         f.write(f"{card_id}\n")
