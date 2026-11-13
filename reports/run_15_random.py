"""Run the student agent on 15 distinct random seeds with the standard budget."""

from __future__ import annotations

import json
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from game2048.benchmark import load_agent, run_game  # noqa: E402
from game2048.metrics import summarize  # noqa: E402


output = Path(__file__).with_name("benchmark_15_random.json")
seeds = secrets.SystemRandom().sample(range(1, 2**31), 15)
agent_class = load_agent("student.agent:Agent")
records = []

for index, seed in enumerate(seeds, start=1):
    record = run_game(agent_class(), seed, decision_budget_ms=None, total_budget_seconds=60)
    records.append(record)
    payload = {"settings": {"agent": "student.agent:Agent", "total_time_sec": 60,
                            "per_move_time_ms": None, "seeds": seeds},
               "summary": summarize(records), "games": records}
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"{index:2}/15 seed={seed} score={record['score']} "
          f"tile={record['max_tile']} moves={record['moves']} "
          f"budget_exhausted={record['budget_exhausted']}", flush=True)

print(f"Saved {output}", flush=True)
