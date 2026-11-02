#!/usr/bin/env python3
"""Whole-token identifier rename for a sweep, with a prefix rule.

  python3 scripts/port/rename_identifiers.py --map /tmp/rename.json --apply engines/ds4

The map is JSON: {"exact": [["metal_ready","gpu_ready"], ...],
                 "prefix": [["metal_graph_","gpu_graph_"], ...]}
A prefix entry rewrites the head and keeps the tail, so one entry covers hundreds of names.
Matching is on word boundaries, and `_` is a word character, so `["METAL_DECODE_LAYER_", ...]`
cannot touch `DS4_METAL_FOO`. That is how chunk 2.2b renamed identifiers while deliberately
keeping the environment variable names.

Verify the result with scripts/port/preprocess_equiv.py --rename-file, passing this same map.
"""
import argparse
import json
import os
import re
import subprocess


def apply_map(text, mapping):
    for old, new in mapping.get("exact", []):
        text = re.sub(r"\b" + re.escape(old) + r"\b", new, text)
    for old, new in mapping.get("prefix", []):
        text = re.sub(r"\b" + re.escape(old) + r"[A-Za-z0-9_]*",
                      lambda m, n=new, o=old: n + m.group(0)[len(o):], text)
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map", required=True)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--ext", default=".c,.cpp,.h,.hpp,.cu,.cuh,.inc",
                    help="comma-separated extensions to rewrite")
    ap.add_argument("paths", nargs="+", help="files or directories")
    args = ap.parse_args()
    mapping = json.load(open(args.map))
    exts = tuple(args.ext.split(","))

    files = []
    for path in args.paths:
        if os.path.isdir(path):
            out = subprocess.run(["git", "ls-files", path], capture_output=True, text=True).stdout
            files += [f for f in out.split() if f.endswith(exts)]
        else:
            files.append(path)

    changed = {}
    for path in files:
        if "/external/" in path or "/vendor/" in path:
            continue                       # vendored third-party: leave it alone
        src = open(path, encoding="utf-8").read()
        new = apply_map(src, mapping)
        if new != src:
            if args.apply:
                open(path, "w", encoding="utf-8").write(new)
            changed[path] = sum(1 for a, b in zip(src.split("\n"), new.split("\n")) if a != b)
    print(f"{'rewrote' if args.apply else 'would rewrite'} {len(changed)} files")
    for path, n in sorted(changed.items(), key=lambda kv: -kv[1])[:20]:
        print(f"  {n:5d} lines  {path}")


if __name__ == "__main__":
    main()
