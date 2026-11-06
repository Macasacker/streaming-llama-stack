#!/usr/bin/env python3
"""Reword text inside string literals, leaving identifiers and comments alone.

    python3 scripts/port/string_text_replace.py --map map.json [--apply] FILE ...

The map is a JSON array of [old, new] pairs. Each pair is applied to string
literals only, longest pair first, so a phrase rule cannot be undone by a shorter
one. Character literals count as strings. A literal naming an environment
variable (anything matching `DS4_*METAL*`) is never rewritten, which is what makes
a message sweep safe next to the environment names a port keeps on purpose.

Without `--apply` the tool reports each pair's hit count and every literal that
still matches the pattern given by `--report` (default: any case-insensitive
`metal`). Verify the result with scripts/port/metal_word_census.py.
"""
import argparse
import json
import re
import sys

WORD = re.compile(r"[Mm][Ee][Tt][Aa][Ll]")
ENV_PREFIX = re.compile(r"^DS4_[A-Z_0-9]*METAL[A-Z_0-9]*$")
IDENT = re.compile(r"[A-Za-z_0-9]+")


def literals(text):
    """[(start, end)] of every string/character literal, contents included.
    `start` is the opening quote, `end` the closing quote's index + 1."""
    out = []
    state = "code"
    quote = '"'
    i, L = 0, len(text)
    begin = 0
    while i < L:
        c = text[i]
        if state == "code":
            if c == "/" and i + 1 < L and text[i + 1] == "/":
                i += 2
                while i < L and text[i] != "\n":
                    i += 1
                continue
            if c == "/" and i + 1 < L and text[i + 1] == "*":
                i += 2
                while i < L and not (text[i] == "*" and i + 1 < L and text[i + 1] == "/"):
                    i += 1
                i += 2
                continue
            if c == '"' or c == "'":
                state = "string"
                quote = c
                begin = i
                i += 1
                continue
            i += 1
            continue
        if c == "\\" and i + 1 < L:
            i += 2
            continue
        if c == quote:
            out.append((begin, i + 1))
            state = "code"
            i += 1
            continue
        i += 1
    return out


def named_env(text, seg):
    """True when a whole identifier inside the segment matches a kept env name."""
    for m in IDENT.finditer(seg):
        if ENV_PREFIX.match(m.group(0)):
            return True
    return False


def rewrite(text, pairs):
    segs = literals(text)
    hits = {pair: 0 for pair in pairs}
    pieces = []
    prev = 0
    for start, end in segs:
        pieces.append(text[prev:start])
        seg = text[start:end]
        if not named_env(text, seg):
            for old, new in pairs:
                if old in seg:
                    hits[(old, new)] += seg.count(old)
                    seg = seg.replace(old, new)
        pieces.append(seg)
        prev = end
    pieces.append(text[prev:])
    return "".join(pieces), hits


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--map", required=True, help="JSON array of [old, new] pairs")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--report", default="metal", help="leftover pattern to list")
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()

    pairs = [(a, b) for a, b in json.load(open(args.map))]
    pairs.sort(key=lambda p: -len(p[0]))
    leftover = re.compile(args.report, re.IGNORECASE)

    total = 0
    for path in args.files:
        text = open(path, encoding="utf-8", errors="replace").read()
        new_text, hits = rewrite(text, pairs)
        used = {k: v for k, v in hits.items() if v}
        for (old, new), n in sorted(used.items(), key=lambda kv: -kv[1]):
            print(f"  {n:3d}x  {old}  ->  {new}")
        total += sum(used.values())
        if args.apply and new_text != text:
            open(path, "w", encoding="utf-8").write(new_text)
            print(f"  wrote {path}")
        if not args.apply:
            for start, end in literals(new_text):
                seg = new_text[start:end]
                if leftover.search(seg) and not named_env(new_text, seg):
                    line = new_text.count("\n", 0, start) + 1
                    print(f"  LEFTOVER {path}:{line}  {seg[:120]}")
    print(f"{total} replacements across {len(args.files)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
