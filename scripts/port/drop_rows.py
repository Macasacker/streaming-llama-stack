#!/usr/bin/env python3
"""Drop envvars.json rows whose last reader was deleted, and report what was dropped.

Deleting a backend deletes its `getenv` sites, and a row whose only reads were those
sites becomes documentation of a variable nothing reads - which is what the plan's rule
forbids. `slstack.envvars.inventory()` is the only honest source of "still read": it is
the same scan `envdocs --check` uses, so a row it no longer reports is a row the gate
would call stale.

Reads must be scanned across the **whole tree**, not only the subtree being pruned:
`GGML_CUDA_GRAPHS` names no deleted backend but `GGML_CUDA_SYCL_GRAPHS` does, and the
former is read in `ggml-cuda/`. And `inventory()` scans `engines/` and `vendor/` only, so
a launcher-owned name read in `slstack/` looks readerless here and must be checked
separately before it is dropped.

    python3 scripts/port/drop_rows.py --names build/dropped-names-41.json
    python3 scripts/port/drop_rows.py --names build/dropped-names-41.json --apply

`reads` are re-anchored afterwards by `reanchor_env_reads.py`, and
`python3 -B -m slstack.envdocs --refresh-defaults` regenerates the evidence and
`docs/VARIABLES.md`.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from slstack.envvars import ROOT, TABLE, inventory


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--names", required=True, type=Path,
                    help="JSON list of variable names to drop")
    ap.add_argument("--apply", action="store_true", help="rewrite envvars.json")
    ap.add_argument("--keep", action="store_true",
                    help="report only, and refuse to drop a name inventory() still reads")
    args = ap.parse_args()

    names = set(json.loads(args.names.read_text(encoding="utf-8")))
    rows = json.loads(TABLE.read_text(encoding="utf-8"))
    by_var = {row["var"]: row for row in rows}
    live = inventory()

    still_read = sorted(names & set(live))
    if args.keep and still_read:
        print("refusing: still read somewhere in the scanned tree:", ", ".join(still_read))
        return 1
    absent = sorted(names - set(by_var))
    if absent:
        print("not in the table:", ", ".join(absent))

    dropped = sorted(names & set(by_var))
    for var in dropped:
        row = by_var[var]
        print(f"drop {var}  ({row['classification']}, {len(row['reads'])} read(s): "
              f"{row['reads'][0] if row['reads'] else 'none'})")
    print(f"dropped {len(dropped)} rows, {len(rows)} -> {len(rows) - len(dropped)}")

    if args.apply:
        kept = [row for row in rows if row["var"] not in names]
        TABLE.write_text(json.dumps(kept, indent=1) + "\n", encoding="utf-8")
        print(f"applied: {TABLE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
