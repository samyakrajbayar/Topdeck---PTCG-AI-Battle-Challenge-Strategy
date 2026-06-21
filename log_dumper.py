# log_dumper.py — Wraps battle execution to dump JSON match logs
# for the visualizer.
#
# Usage:
#   from log_dumper import run_and_dump
#   run_and_dump(deck_path="deck.csv", agent_path="main.py",
#                output="match_log.json")

import json
from kaggle_environments import make

def run_and_dump(deck_path="deck.csv", output="match_log.json"):
    """Run a match and save every observation + log event as JSON."""
    # Read deck
    with open(deck_path) as f:
        deck = [line.strip() for line in f if line.strip()]

    env = make("cabt", configuration={"decks": [deck, deck]})
    env.run(["main.py", "main.py"])

    # Extract steps
    steps = []
    for i, step in enumerate(env.steps):
        step_data = {
            "step": i,
            "observations": [],
            "actions": [],
        }
        for player_idx in range(len(step)):
            obs = step[player_idx].get("observation", {})
            action = step[player_idx].get("action", None)
            step_data["observations"].append({
                "current": obs.get("current"),
                "select": obs.get("select"),
                "logs": obs.get("logs", []),
            })
            step_data["actions"].append(action)
        steps.append(step_data)

    with open(output, "w") as f:
        json.dump({"steps": steps, "config": env.configuration}, f, indent=2, default=str)

    print(f"Match log saved to {output} ({len(steps)} steps)")

if __name__ == "__main__":
    run_and_dump()
