"""Check a Full-Duplex-Bench run before scoring it.

A sample only measures the agent if the agent actually joined the room and
heard the user. When LiveKit credits run out, or the agent process hangs,
the harness still saves a result for every sample, so a broken run looks
complete. This script matches each result to the agent's trace and lists the
samples where the agent heard nothing.

Run from the project root:

    uv run python scripts/check_fdb_run.py Full-Duplex-Bench/v3/fdb_subset_48
"""

import json
import sys
from pathlib import Path

results_dir = Path(
    sys.argv[1] if len(sys.argv) > 1 else "Full-Duplex-Bench/v3/fdb_subset_48"
)
traces_dir = Path("traces")

rows = []
for result_file in results_dir.glob("*/result_interruptible.json"):
    result = json.loads(result_file.read_text())
    trace = traces_dir / f"trace_{result.get('room_name')}.jsonl"
    if not trace.exists():
        status = "agent never joined"
    elif any(json.loads(line).get("type") == "USER_MESSAGE" for line in trace.open()):
        status = "ok"
    else:
        status = "agent heard nothing"
    rows.append((result.get("evaluated_at", ""), result["example_id"], status))

rows.sort()
bad = [r for r in rows if r[2] != "ok"]
print(f"{len(rows)} results, {len(rows) - len(bad)} ok")
print("".join("." if r[2] == "ok" else "x" for r in rows), "(in run order)")
for _, example_id, status in bad:
    print(f"  {example_id}: {status}")
if bad:
    print(
        "\nA run of x's at the end usually means credits ran out (agent log shows "
        "429) or the agent hung. Fix that, delete those result_interruptible.json "
        "files and rerun without --force."
    )
sys.exit(1 if bad else 0)
