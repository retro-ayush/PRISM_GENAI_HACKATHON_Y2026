"""Build the balanced 48-sample Full-Duplex-Bench v3 subset.

12 scenarios per domain (one recording each), spread across easy / medium /
hard. Selection is seeded, so every machine gets the same 48 samples. The
subset folders link to the released audio instead of copying it.

Run from Full-Duplex-Bench/v3 (after downloading fdb_v3_data_released/):

    .venv/bin/python ../../scripts/make_fdb_subset.py
"""

import argparse
import collections
import json
import os
import random
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--src", default="fdb_v3_data_released")
parser.add_argument("--dst", default="fdb_subset_48")
parser.add_argument("--per-domain", type=int, default=12)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

src, dst = Path(args.src), Path(args.dst)
random.seed(args.seed)

by_example: dict[str, tuple[Path, dict]] = {}
for folder in sorted(src.iterdir()):
    if (folder / "metadata.json").exists():
        meta = json.loads((folder / "metadata.json").read_text())
        by_example.setdefault(meta["id"], (folder, meta))  # one recording each

domains = collections.defaultdict(list)
for example_id, (folder, meta) in by_example.items():
    domains[meta["domain"]].append((meta["difficulty"], example_id, folder))

chosen = []
for _domain, items in sorted(domains.items()):
    random.shuffle(items)
    tiers = collections.defaultdict(list)
    for item in items:
        tiers[item[0]].append(item)
    picked = []
    while len(picked) < args.per_domain and any(tiers.values()):
        for tier in ("easy", "medium", "hard"):
            if tiers[tier] and len(picked) < args.per_domain:
                picked.append(tiers[tier].pop())
    chosen += picked

dst.mkdir(exist_ok=True)
for _tier, _example_id, folder in chosen:
    out = dst / folder.name
    out.mkdir(exist_ok=True)
    for name in ("input.wav", "metadata.json"):
        if not (out / name).exists():
            os.symlink((folder / name).resolve(), out / name)

print(f"{len(chosen)} samples in {dst}/")
