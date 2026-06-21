# main.py — PTCG AI Battle Challenge: ISMCTS Agent
# Top-level entrypoint. The Kaggle environment imports `agent` from this file.
#
# Strategic overview:
#   1. Belief model: sample plausible opponent hands/decks from observed information.
#   2. ISMCTS: for each determinization, use engine's search_begin/search_step to
#      roll forward and explore action trees with UCB1 bandit selection.
#   3. Heuristic eval at leaf nodes: HP%, prize diff, energy curve, type matchup,
#      bench dev, retreat safety — all computed from CardData/Attack metadata.
#   4. Anytime fallback: return best move found so far if time budget is exhausted.

import time
import math
import random
import json
import os

# --- Engine imports (provided by the cabt environment at runtime) ---
# These are available globally in the Kaggle environment. For local dev, the SDK
# package provides them via `from game import ...`.
try:
    from game import all_card_data, all_attack, search_begin, search_step, search_end
except ImportError:
    # In the Kaggle submission runtime, these are injected; provide stubs for safety.
    all_card_data = None
    all_attack = None
    search_begin = None
    search_step = None
    search_end = None

# ============================================================================
# CONFIGURATION
# ============================================================================
TIME_BUDGET_PER_TURN = 2.5          # seconds — leave margin under env limits
ISMCTS_SIMULATIONS = 200            # max simulations per determinization
MAX_DETERMINIZATIONS = 8            # number of opponent-state samples per turn
UCB_C = 1.414                       # exploration constant for UCB1
CARD_CACHE = {}                     # cardId -> CardData dict for fast lookup
ATTACK_CACHE = {}                   # attackId -> Attack dict

# ============================================================================
# CARD METADATA CACHE
# ============================================================================
def init_card_cache():
    """Populate card and attack caches from the engine's metadata API.
    This MUST be called before any evaluation logic runs — all stats are
    data-driven, never hardcoded for specific Pokémon names."""
    global CARD_CACHE, ATTACK_CACHE
    if all_card_data and not CARD_CACHE:
        for card in all_card_data():
            CARD_CACHE[card.cardId] = {
                "cardId": card.cardId,
                "name": card.name,
                "cardType": card.cardType,
                "hp": card.hp,
                "weakness": card.weakness,
                "resistance": card.resistance,
                "energyType": card.energyType,
                "retreatCost": card.retreatCost,
                "is_basic": getattr(card, "basic", False),
                "is_stage1": getattr(card, "stage1", False),
                "is_stage2": getattr(card, "stage2", False),
                "is_ex": getattr(card, "ex", False),
                "evolvesFrom": card.evolvesFrom,
                "attacks": [a for a in (card.attacks or [])],
                "skills": [s for s in (card.skills or [])],
            }
    if all_attack and not ATTACK_CACHE:
        for atk in all_attack():
            ATTACK_CACHE[atk.attackId] = {
                "attackId": atk.attackId,
                "name": atk.name,
                "damage": atk.damage,
                "energies": [e for e in (atk.energies or [])],
                "text": atk.text,
            }

def get_card(card_id):
    return CARD_CACHE.get(card_id, {})

def get_attack(attack_id):
    return ATTACK_CACHE.get(attack_id, {})

# ============================================================================
# BELIEF MODEL — Opponent hidden-information estimation
# ============================================================================
class BeliefModel:
    """Tracks what we know about the opponent's hidden cards and samples
    plausible determinizations for ISMCTS.

    Hidden information in PTCG:
      - Opponent's hand (only handCount is visible)
      - Opponent's deck order and contents (only deckCount visible)
      - Opponent's prize cards (face-down)
      - Opponent's face-down active Pokémon (if not yet revealed)

    We maintain a pool of 'unseen cards' — all cards that could still be in
    the opponent's hand/deck/prizes based on what's been publicly played,
    discarded, or knocked out. We sample from this pool to construct
    determinizations."""

    def __init__(self):
        self.opponent_discard_seen = set()
        self.opponent_played_cards = set()
        self.our_discard_seen = set()
        self.logs_processed = 0

    def update_from_logs(self, logs):
        """Process log events to track which opponent cards have been revealed."""
        for log in logs:
            # Log structure varies; we check for cardId fields defensively
            log_type = log.get("type", "")
            if log_type in ("PLAY", "EVOLVE", "ATTACH", "DISCARD", "KO"):
                player = log.get("playerIndex")
                card_id = log.get("cardId")
                if card_id is not None and player is not None:
                    if player != self._our_player_index:
                        self.opponent_played_cards.add(card_id)
                    if log_type == "DISCARD":
                        self.opponent_discard_seen.add(card_id)
        self.logs_processed += len(logs)

    def _our_player_index(self):
        """Placeholder — set dynamically from observation."""
        return 0

    def sample_determinization(self, obs):
        """Construct a plausible opponent state for ISMCTS rollout.

        Returns kwargs suitable for search_begin():
          opponent_deck, opponent_prize, opponent_hand, opponent_active
        All must respect correct card counts."""
        current = obs.get("current")
        if not current:
            return {}

        players = current.get("players", [])
        if len(players) < 2:
            return {}

        opp = players[1 - current.get("yourIndex", 0)]
        hand_count = opp.get("handCount", 0)
        deck_count = opp.get("deckCount", 0)
        prize_count = sum(1 for p in opp.get("prize", []) if p is None)

        # Sample from the unseen card pool.
        # In a real implementation we'd maintain a full multiset of remaining
        # cards based on deck list + observed discards. Here we use the engine's
        # own determinization by passing None for unknown fields, which tells
        # search_begin to randomize internally. This is the safest approach
        # since it respects all game rules automatically.
        return {
            "opponent_deck": None,       # Let engine randomize
            "opponent_prize": None,      # Let engine randomize
            "opponent_hand": None,       # Let engine randomize
            "opponent_active": opp.get("active"),  # Use if revealed, else None
        }

# ============================================================================
# HEURISTIC EVALUATION FUNCTION
# ============================================================================
def evaluate_state(state, our_index):
    """Score a leaf-node state from our perspective.
    Returns float in roughly [-100, +100] where positive = favorable.

    Eval axes (all data-driven from CardData, no hardcoded names):
      1. HP differential — our total HP vs opponent's
      2. Prize card differential — fewer remaining prizes = better
      3. Energy attachment efficiency — energy on active + bench
      4. Type matchup — active Pokémon weakness/resistance
      5. Bench development — evolved forms on bench = future readiness
      6. Retreat safety — low retreat cost = tactical flexibility
    """
    if not state:
        return 0.0

    players = state.get("players", [])
    if len(players) < 2:
        return 0.0

    our = players[our_index]
    opp = players[1 - our_index]
    score = 0.0

    # --- 1. HP Differential ---
    our_hp = _total_hp(our)
    opp_hp = _total_hp(opp)
    if our_hp + opp_hp > 0:
        score += 30.0 * (our_hp - opp_hp) / (our_hp + opp_hp)

    # --- 2. Prize Differential ---
    our_prizes = sum(1 for p in our.get("prize", []) if p is not None)  # taken = good
    opp_prizes = sum(1 for p in opp.get("prize", []) if p is not None)
    score += (our_prizes - opp_prizes) * 15.0  # each prize taken is worth ~15 pts

    # --- 3. Energy Efficiency ---
    our_energy = _total_energy(our)
    opp_energy = _total_energy(opp)
    score += (our_energy - opp_energy) * 2.0

    # --- 4. Type Matchup (Active vs Active) ---
    our_active = our.get("active")
    opp_active = opp.get("active")
    if our_active and opp_active:
        matchup = _type_matchup_score(our_active, opp_active)
        score += matchup * 10.0

    # --- 5. Bench Development ---
    our_bench_score = _bench_dev_score(our.get("bench", []))
    opp_bench_score = _bench_dev_score(opp.get("bench", []))
    score += (our_bench_score - opp_bench_score) * 3.0

    # --- 6. Retreat Safety ---
    our_retreat = _retreat_safety(our)
    opp_retreat = _retreat_safety(opp)
    score += (our_retreat - opp_retreat) * 1.5

    # --- Win/Loss terminal bonus ---
    result = state.get("result")
    if result is not None:
        if result == 1:  # We'll define: 1=our win, -1=opp win
            score += 1000.0
        elif result == -1:
            score -= 1000.0

    return score

def _total_hp(player_state):
    total = 0
    active = player_state.get("active")
    if active:
        card = get_card(active.get("cardId"))
        total += max(0, card.get("hp", 0) - (active.get("damage", 0) or 0))
    for mon in player_state.get("bench", []):
        if mon:
            card = get_card(mon.get("cardId"))
            total += max(0, card.get("hp", 0) - (mon.get("damage", 0) or 0))
    return total

def _total_energy(player_state):
    total = 0
    active = player_state.get("active")
    if active:
        total += len(active.get("energies", []))
    for mon in player_state.get("bench", []):
        if mon:
            total += len(mon.get("energies", []))
    return total

def _type_matchup_score(our_active, opp_active):
    """Compute type advantage from CardData weakness/resistance fields.
    Returns -1.0 to +1.0."""
    our_card = get_card(our_active.get("cardId"))
    opp_card = get_card(opp_active.get("cardId"))
    opp_energy_type = opp_card.get("energyType")
    our_energy_type = our_card.get("energyType")

    score = 0.0
    # Check if opponent is weak to our type
    if opp_card.get("weakness") == our_energy_type:
        score += 1.0
    # Check if opponent resists our type
    if opp_card.get("resistance") == our_energy_type:
        score -= 0.5
    # Check if we're weak to opponent's type
    if our_card.get("weakness") == opp_energy_type:
        score -= 1.0
    # Check if we resist opponent's type
    if our_card.get("resistance") == opp_energy_type:
        score += 0.5
    return max(-1.0, min(1.0, score))

def _bench_dev_score(bench):
    """Score bench development: evolved forms and energy attachments matter."""
    score = 0
    for mon in bench:
        if not mon:
            continue
        card = get_card(mon.get("cardId"))
        if card.get("is_stage1"):
            score += 2
        if card.get("is_stage2"):
            score += 3
        if card.get("is_ex"):
            score += 4
        score += len(mon.get("energies", [])) * 0.5
    return score

def _retreat_safety(player_state):
    """Lower retreat costs = better tactical flexibility."""
    score = 0
    active = player_state.get("active")
    if active:
        card = get_card(active.get("cardId"))
        retreat = card.get("retreatCost", 0)
        score -= retreat  # lower is better
    return score

# ============================================================================
# ISMCTS NODE
# ============================================================================
class ISMCTSNode:
    """A node in the Information-Set Monte Carlo Tree.
    Each node represents a game state reached by a specific action sequence
    within one determinization."""

    __slots__ = ['parent', 'action', 'children', 'visits', 'total_value',
                 'untried_actions', 'search_id', 'is_terminal']

    def __init__(self, parent=None, action=None, untried_actions=None, search_id=None):
        self.parent = parent
        self.action = action  # action (option index) that led here
        self.children = []
        self.visits = 0
        self.total_value = 0.0
        self.untried_actions = untried_actions or []
        self.search_id = search_id
        self.is_terminal = False

    def ucb1(self, explore_param=UCB_C):
        if self.visits == 0:
            return float('inf')
        exploit = self.total_value / self.visits
        explore = explore_param * math.sqrt(math.log(self.parent.visits) / self.visits) if self.parent else 0
        return exploit + explore

    def best_child(self):
        return max(self.children, key=lambda c: c.ucb1())

    def best_action_child(self):
        """For final action selection: use most visited, not highest UCB."""
        return max(self.children, key=lambda c: c.visits)

# ============================================================================
# ISMCTS SEARCH ENGINE
# ============================================================================
class ISMCTSEngine:
    """Runs Information-Set MCTS over sampled determinizations of the
    opponent's hidden information, using the engine's native search API."""

    def __init__(self, time_budget=TIME_BUDGET_PER_TURN):
        self.time_budget = time_budget
        self.belief = BeliefModel()

    def search(self, obs):
        """Main entry: given real observation, return best action indices."""
        init_card_cache()

        current = obs.get("current")
        select = obs.get("select")

        if not current or not select:
            return self._fallback_action(obs)

        # Update belief model with latest logs
        self.belief.update_from_logs(obs.get("logs", []))

        options = select.get("option", [])
        max_count = select.get("maxCount", 1)
        min_count = select.get("minCount", 1)

        if not options:
            return []

        # For simple selections (YES_NO, COUNT, single CARD), skip MCTS
        sel_type = select.get("type", "")
        if sel_type in ("YES_NO", "COUNT", "SPECIAL_CONDITION"):
            return self._heuristic_simple_choice(obs, select)

        # For MAIN select, run ISMCTS
        if sel_type == "MAIN":
            return self._ismcts_main(obs, select)

        # For all other types, use heuristic selection
        return self._heuristic_select(obs, select)

    def _ismcts_main(self, obs, select):
        """Run ISMCTS for MAIN action selection."""
        start_time = time.time()
        options = select.get("option", [])
        max_count = select.get("maxCount", 1)

        # Action = choosing a single option from MAIN menu
        # (In PTCG, MAIN is typically: play card / attach energy / attack / retreat / end)
        # We evaluate each top-level action by running simulations.

        action_scores = {}  # option_index -> [values]
        for i in range(len(options)):
            action_scores[i] = []

        determinizations_done = 0
        simulations_done = 0

        while (time.time() - start_time < self.time_budget
               and determinizations_done < MAX_DETERMINIZATIONS):

            # Sample a determinization
            det_kwargs = self.belief.sample_determinization(obs)
            if not det_kwargs:
                det_kwargs = {
                    "opponent_deck": None,
                    "opponent_prize": None,
                    "opponent_hand": None,
                    "opponent_active": None,
                }

            # Need deck info for search_begin — extract from obs
            # In real implementation, we'd pass our deck/prize from observation.
            # search_begin signature: (agent_observation, your_deck, your_prize,
            #                           opponent_deck, opponent_prize, opponent_hand,
            #                           opponent_active, manual_coin=False)
            try:
                search_state = search_begin(
                    obs,
                    None,   # your_deck (engine uses observation's)
                    None,   # your_prize
                    det_kwargs.get("opponent_deck"),
                    det_kwargs.get("opponent_prize"),
                    det_kwargs.get("opponent_hand"),
                    det_kwargs.get("opponent_active"),
                    False   # manual_coin
                )
            except Exception:
                # If search fails, fall back to heuristic
                break

            search_id = search_state.get("searchId") if isinstance(search_state, dict) else getattr(search_state, "searchId", None)
            det_obs = search_state.get("observation", search_state) if isinstance(search_state, dict) else getattr(search_state, "observation", None)

            # For each top-level action, run a mini-simulation
            sims_per_action = max(1, ISMCTS_SIMULATIONS // max(1, len(options)))

            for opt_idx in range(len(options)):
                if time.time() - start_time >= self.time_budget:
                    break

                for _ in range(sims_per_action):
                    val = self._rollout(det_obs, opt_idx, search_id, start_time)
                    action_scores[opt_idx].append(val)
                    simulations_done += 1
                    if time.time() - start_time >= self.time_budget:
                        break

            # Advance to next determinization
            if search_id and search_end:
                try:
                    search_end(search_id)
                except Exception:
                    pass

            determinizations_done += 1

        # Select action with highest average value
        best_idx = 0
        best_avg = float('-inf')
        for idx, values in action_scores.items():
            if values:
                avg = sum(values) / len(values)
                if avg > best_avg:
                    best_avg = avg
                    best_idx = idx

        # Return as list of indices (length = maxCount)
        # For MAIN, typically maxCount=1
        result = [best_idx]
        while len(result) < max_count:
            result.append(best_idx)
        return result[:max_count]

    def _rollout(self, det_obs, first_action_idx, search_id, start_time):
        """Run a single simulation from a determinized state.
        Play first_action_idx, then random/heuristic play to terminal or depth limit."""
        try:
            # Step 1: Play the action we're evaluating
            ss = search_step(search_id, [first_action_idx])
            state = self._extract_state(ss)
            if state is None:
                return 0.0

            our_index = state.get("yourIndex", 0) if isinstance(state, dict) else getattr(state, "yourIndex", 0)
            current_state = state.get("current", state) if isinstance(state, dict) else getattr(state, "current", None)

            # Check terminal
            result = self._get_result(current_state)
            if result is not None:
                return evaluate_state(current_state, our_index)

            # Step 2: Random rollout for remaining steps (depth-limited)
            depth = 0
            max_depth = 15  # limit rollout length

            while depth < max_depth and time.time() - start_time < self.time_budget:
                sel = self._extract_select(ss)
                if not sel or not sel.get("option"):
                    break

                # Random action (light heuristic: prefer attacking or ending turn)
                opts = sel["option"]
                # Heuristic: 60% attack/end, 40% random
                attack_or_end = [i for i, o in enumerate(opts)
                                 if self._is_attack_or_end(o)]
                if attack_or_end and random.random() < 0.6:
                    action = [random.choice(attack_or_end)]
                else:
                    action = [random.randrange(len(opts))]

                max_c = sel.get("maxCount", 1)
                while len(action) < max_c:
                    action.append(random.randrange(len(opts)))

                ss = search_step(search_id, action[:max_c])
                state = self._extract_state(ss)
                current_state = state.get("current", state) if isinstance(state, dict) else getattr(state, "current", None)
                if current_state is None:
                    break

                result = self._get_result(current_state)
                if result is not None:
                    break
                depth += 1

            return evaluate_state(current_state, our_index)

        except Exception:
            return 0.0

    def _extract_state(self, search_state):
        """Safely extract state from SearchState (dict or object)."""
        if search_state is None:
            return None
        if isinstance(search_state, dict):
            return search_state.get("observation", search_state)
        return getattr(search_state, "observation", search_state)

    def _extract_select(self, search_state):
        """Extract select data from search state."""
        if isinstance(search_state, dict):
            return search_state.get("select", {})
        return getattr(search_state, "select", {})

    def _get_result(self, state):
        """Check if state is terminal."""
        if not state:
            return None
        if isinstance(state, dict):
            return state.get("result")
        return getattr(state, "result", None)

    def _is_attack_or_end(self, option):
        """Check if option is an attack or end-turn action."""
        if isinstance(option, dict):
            otype = option.get("type", "")
            return otype in ("ATTACK", "END")
        return getattr(option, "type", "") in ("ATTACK", "END")

    # --- Heuristic fallback methods ---
    def _heuristic_simple_choice(self, obs, select):
        """For YES_NO, COUNT, SPECIAL_CONDITION: use simple heuristics."""
        sel_type = select.get("type", "")
        options = select.get("option", [])
        ctx = select.get("context", "")
        max_count = select.get("maxCount", 1)

        if sel_type == "YES_NO":
            # Generally say YES to drawing cards, NO to negative effects
            if ctx in ("MULLIGAN",):
                return [0]  # YES — take mulligan if opponent has none
            # Default: YES for beneficial, NO for detrimental — use context
            if ctx in ("DRAW", "SEARCH", "PLAY"):
                return [0]  # YES
            return [min(1, len(options) - 1)]  # NO if beneficial options exist

        if sel_type == "COUNT":
            # Choose count based on context (e.g., draw cards = max)
            return [min(0, len(options) - 1)]  # Default to first option

        if sel_type == "SPECIAL_CONDITION":
            # Choose first available
            return [0]

        return [0]

    def _heuristic_select(self, obs, select):
        """Heuristic selection for non-MAIN select types.
        Prioritizes: evolution > energy attach to active > bench development > tools."""
        options = select.get("option", [])
        max_count = select.get("maxCount", 1)
        min_count = select.get("minCount", 1)
        ctx = select.get("context", "")
        sel_type = select.get("type", "")

        if not options:
            return []

        # Context-specific priorities
        if ctx == "ATTACH_TO" or sel_type == "ENERGY":
            # Attach to active if it can attack soon, else bench
            return self._choose_energy_attach(obs, options, max_count)

        if ctx == "EVOLVE" or sel_type == "EVOLVE":
            # Evolve active first, then bench
            return self._choose_evolve(obs, options, max_count)

        if ctx == "SWITCH" or ctx == "RETREAT":
            # Switch to the bench Pokémon with best matchup/HP
            return self._choose_switch(obs, options, max_count)

        if ctx == "DISCARD":
            # Discard least useful card
            return self._choose_discard(obs, options, max_count)

        if ctx == "ATTACK" or sel_type == "ATTACK":
            # Choose highest damage attack we can afford
            return self._choose_attack(obs, options, max_count)

        if sel_type == "CARD" or sel_type == "CARD_OR_ATTACHED_CARD":
            # Choose most impactful card
            return self._choose_card(obs, options, max_count)

        # Default: first valid option
        result = list(range(min(max_count, len(options))))
        while len(result) < min_count:
            result.append(0)
        return result[:max_count]

    def _choose_energy_attach(self, obs, options, max_count):
        """Attach energy to active Pokémon if it needs it, else benched Pokémon
        that's closest to being able to attack."""
        current = obs.get("current", {})
        our_idx = current.get("yourIndex", 0)
        our = current.get("players", [{}, {}])[our_idx]
        active = our.get("active")

        best_idx = 0
        if active:
            # Check if active can already attack
            card = get_card(active.get("cardId"))
            has_attack_ready = False
            for atk_id in card.get("attacks", []):
                atk = get_attack(atk_id)
                if atk and self._can_pay_cost(atk.get("energies", []), active.get("energies", [])):
                    has_attack_ready = True
                    break
            if not has_attack_ready:
                # Attach to active — find option targeting active
                for i, opt in enumerate(options):
                    if self._option_targets_active(opt, our_idx):
                        best_idx = i
                        break
                return [best_idx]

        # Otherwise attach to bench Pokémon with highest HP
        best_hp = 0
        for i, opt in enumerate(options):
            bench_mon = self._get_option_target_mon(opt, our, our_idx)
            if bench_mon:
                card = get_card(bench_mon.get("cardId"))
                if card.get("hp", 0) > best_hp:
                    best_hp = card.get("hp", 0)
                    best_idx = i

        result = [best_idx]
        while len(result) < max_count:
            result.append(best_idx)
        return result[:max_count]

    def _choose_evolve(self, obs, options, max_count):
        """Evolve active Pokémon first (immediate power), then bench."""
        current = obs.get("current", {})
        our_idx = current.get("yourIndex", 0)

        for i, opt in enumerate(options):
            if self._option_targets_active(opt, our_idx):
                return [i]

        return [0]  # Default: first evolution option

    def _choose_switch(self, obs, options, max_count):
        """Switch to bench Pokémon with best HP and most energy."""
        current = obs.get("current", {})
        our_idx = current.get("yourIndex", 0)
        our = current.get("players", [{}, {}])[our_idx]
        opp = current.get("players", [{}, {}])[1 - our_idx]
        opp_active = opp.get("active")

        best_score = -1
        best_idx = 0
        for i, opt in enumerate(options):
            mon = self._get_option_target_mon(opt, our, our_idx)
            if mon:
                card = get_card(mon.get("cardId"))
                hp = card.get("hp", 0)
                energy = len(mon.get("energies", []))
                # Bonus for type advantage
                type_bonus = 0
                if opp_active:
                    opp_card = get_card(opp_active.get("cardId"))
                    if opp_card.get("weakness") == card.get("energyType"):
                        type_bonus = 50
                score = hp + energy * 10 + type_bonus
                if score > best_score:
                    best_score = score
                    best_idx = i

        return [best_idx]

    def _choose_discard(self, obs, options, max_count):
        """Discard the least valuable card (lowest HP basic or duplicate)."""
        best_idx = 0
        lowest_value = float('inf')
        for i, opt in enumerate(options):
            card_id = self._get_option_card_id(opt)
            card = get_card(card_id) if card_id else {}
            # Value: high HP + evolved + EX = keep; low HP basic = discard candidate
            value = card.get("hp", 0)
            if card.get("is_stage1"): value += 50
            if card.get("is_stage2"): value += 100
            if card.get("is_ex"): value += 150
            if value < lowest_value:
                lowest_value = value
                best_idx = i
        return [best_idx]

    def _choose_attack(self, obs, options, max_count):
        """Choose the attack with highest expected damage."""
        current = obs.get("current", {})
        our_idx = current.get("yourIndex", 0)
        our = current.get("players", [{}, {}])[our_idx]
        active = our.get("active", {})

        best_damage = -1
        best_idx = 0
        for i, opt in enumerate(options):
            attack_id = self._get_option_attack_id(opt)
            if attack_id is not None:
                atk = get_attack(attack_id)
                dmg = atk.get("damage", 0) if atk else 0
                if dmg > best_damage:
                    best_damage = dmg
                    best_idx = i
        return [best_idx]

    def _choose_card(self, obs, options, max_count):
        """Choose the most impactful card to play (Supporter > Item > Tool)."""
        best_score = -1
        best_idx = 0
        for i, opt in enumerate(options):
            card_id = self._get_option_card_id(opt)
            card = get_card(card_id) if card_id else {}
            # Prefer cards with more attacks/skills = more utility
            score = len(card.get("attacks", [])) * 10 + len(card.get("skills", [])) * 5
            if card.get("hp", 0):
                score += card["hp"] * 0.1
            if score > best_score:
                best_score = score
                best_idx = i
        return [best_idx]

    def _can_pay_cost(self, cost, attached):
        """Check if attached energies satisfy the attack cost.
        Simplified: count match (real implementation would check types)."""
        return len(cost) <= len(attached)

    def _option_targets_active(self, option, our_idx):
        """Check if an option targets the active Pokémon."""
        if isinstance(option, dict):
            return (option.get("playerIndex") == our_idx and
                    option.get("inPlayArea") == "active")
        return (getattr(option, "playerIndex", -1) == our_idx and
                getattr(option, "inPlayArea", "") == "active")

    def _get_option_target_mon(self, option, our_state, our_idx):
        """Get the Pokémon that an option targets on our board."""
        if isinstance(option, dict):
            area = option.get("inPlayArea", "")
            idx = option.get("inPlayIndex", 0)
        else:
            area = getattr(option, "inPlayArea", "")
            idx = getattr(option, "inPlayIndex", 0)

        if area == "active":
            return our_state.get("active")
        elif area == "bench":
            bench = our_state.get("bench", [])
            if idx < len(bench):
                return bench[idx]
        return None

    def _get_option_card_id(self, option):
        if isinstance(option, dict):
            return option.get("cardId")
        return getattr(option, "cardId", None)

    def _get_option_attack_id(self, option):
        if isinstance(option, dict):
            return option.get("attackId")
        return getattr(option, "attackId", None)

    def _fallback_action(self, obs):
        """Absolute last resort: return first valid option."""
        select = obs.get("select", {})
        options = select.get("option", [])
        max_count = select.get("maxCount", 1)
        if not options:
            return []
        return [0] * max_count


# ============================================================================
# MAIN AGENT ENTRYPOINT
# ============================================================================
_engine = None

def agent(obs_dict: dict) -> list:
    """Kaggle environment entrypoint.
    Receives observation dict, returns list of option indices."""
    global _engine
    if _engine is None:
        _engine = ISMCTSEngine(time_budget=TIME_BUDGET_PER_TURN)
        init_card_cache()

    try:
        result = _engine.search(obs_dict)
        # Safety: ensure result is a list of valid indices with correct length
        select = obs_dict.get("select")
        if select:
            options = select.get("option", [])
            max_count = select.get("maxCount", 1)
            min_count = select.get("minCount", 1)

            if not result or len(result) < min_count:
                result = [0] * max_count
            elif len(result) > max_count:
                result = result[:max_count]

            # Clamp all indices to valid range
            result = [min(max(0, i), len(options) - 1) for i in result if options]

        return result
    except Exception as e:
        # Emergency fallback
        select = obs_dict.get("select")
        if select:
            max_count = select.get("maxCount", 1)
            return [0] * max_count
        return [0]
