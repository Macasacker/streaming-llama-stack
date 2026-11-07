#!/usr/bin/env python3
"""Re-anchor `envvars.json` read locations to the line they are actually on now.

Every read in the table is `path:line`, and `slstack.envdocs --check` compares those
line numbers against the tree, so editing a source file above a read breaks the gate
even when the read itself is untouched. This recomputes the live reads with
`slstack.envvars.inventory()` and, for each variable, maps a recorded reference onto
the live read in the same file by order: when a file's read count is unchanged, the
n-th recorded line becomes the n-th live line, and only genuinely moved reads are
rewritten. A file whose read count changed is reported instead of guessed at, because
that means reads were added or removed - which is a table decision, not a line fix.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from slstack.envvars import ROOT, TABLE, inventory


def by_file(refs):
    out = defaultdict(list)
    for ref in refs:
        path, _, line = ref.rpartition(':')
        out[path].append(int(line))
    return {path: sorted(lines) for path, lines in out.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--apply", action="store_true", help="rewrite envvars.json")
    ap.add_argument("--verbose", action="store_true", help="list every re-anchored read")
    args = ap.parse_args()

    table = json.loads(TABLE.read_text(encoding="utf-8"))
    live = inventory()
    changed = 0
    problems = []
    for row in table:
        var = row["var"]
        current = live.get(var, [])
        recorded = row["reads"]
        want = by_file(recorded)
        have = by_file(current)
        new_refs = []
        for path in sorted(set(want) | set(have)):
            before, after = want.get(path, []), have.get(path, [])
            if path not in want:
                # a file that gained reads for this variable is the missing-row
                # case, handled by envdocs --check; record it here too
                if path in have:
                    problems.append(f"{var}: {path} has {len(after)} live read(s), none recorded")
                    new_refs.extend(f"{path}:{line}" for line in after)
                continue
            if len(before) != len(after):
                problems.append(f"{var}: {path} recorded {len(before)} read(s), live {len(after)}")
                new_refs.extend(f"{path}:{line}" for line in before)
                continue
            for old_line, new_line in zip(before, after):
                if old_line != new_line:
                    changed += 1
                    if args.verbose:
                        print(f"  {var}: {path}:{old_line} -> {new_line}")
                new_refs.append(f"{path}:{new_line}")
        row["reads"] = sorted(new_refs)

    print(f"re-anchored {changed} read location(s)")
    for problem in problems:
        print("  needs a decision:", problem)
    if args.apply:
        TABLE.write_text(json.dumps(table, indent=1) + "\n", encoding="utf-8")
        print(f"applied: {TABLE.relative_to(ROOT)}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
