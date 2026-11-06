#!/usr/bin/env python3
"""Census of Metal-named words in a source tree, split by what the compiler sees.

    python3 scripts/port/metal_word_census.py [--strict] [PATH ...]

Default PATH is engines/ds4. Vendored files (external/, linenoise.c) are skipped:
they are inherited third-party sources the port does not rewrite.

Every case-insensitive `metal` mention is classified by its position in the file,
not by a word-boundary regex. `_` is a word character, so `grep -n '\bMETAL_'`
cannot see `DS4_GLM_METAL_FOO` or `DS4_METAL_PROFILE_X` - the two classes chunks
2.2b, 2.2d and 2.2e each missed in turn. Position is the only honest test:

  IDENTIFIER       in code position, so the compiler sees it. Must be 0.
  ENV_NAME         inside a name that envvars.json documents (kept on purpose:
                   renaming a knob breaks existing recipes).
  ENV_NAME_CITED   inside a DS4_*METAL* name that the table does not document.
                   Live reads through a wrapper such as glm_graph_env_present()
                   are invisible to envdocs' inventory, so these are usually
                   readable knobs; a name with no reader anywhere is stale.
  COMMENT          inside a // or /* */ comment.
  STRING           inside a string or character literal, i.e. printed output.

--strict exits 1 when any IDENTIFIER mention remains.
"""
import argparse
import json
import os
import re
import sys

VENDORED = (os.sep + "external" + os.sep, os.sep + "linenoise.c")
SOURCE_EXT = (".c", ".h", ".cpp", ".hpp", ".cu", ".cuh", ".inc", ".m", ".metal")
WORD = re.compile(r"[Mm][Ee][Tt][Aa][Ll]")
ENV_PREFIX = re.compile(r"^DS4_[A-Z_0-9]*METAL[A-Z_0-9]*$")


def scan(text):
    """Every `metal` mention as (line, offset, bucket) with bucket in
    code/line-comment/block-comment/string. Character state is tracked across the
    whole file, so a mention in a multi-line string is not called code."""
    rows = []
    n = 1
    state, quote = "code", '"'
    i, L = 0, len(text)
    while i < L:
        c = text[i]
        if c == "\n":
            n += 1
            i += 1
            continue
        if state == "code":
            if c == "/" and i + 1 < L and text[i + 1] == "/":
                state = "line"
                i += 2
                continue
            if c == "/" and i + 1 < L and text[i + 1] == "*":
                state = "block"
                i += 2
                continue
            if c == '"' or c == "'":
                state = "string"
                quote = c
                i += 1
                continue
            if WORD.match(text, i):
                rows.append((n, i, "code"))
                i += 5
                continue
            i += 1
            continue
        if state == "line":
            if WORD.match(text, i):
                rows.append((n, i, "line_comment"))
                i += 5
                continue
            i += 1
            continue
        if state == "block":
            if c == "*" and i + 1 < L and text[i + 1] == "/":
                state = "code"
                i += 2
                continue
            if WORD.match(text, i):
                rows.append((n, i, "block_comment"))
                i += 5
                continue
            i += 1
            continue
        if c == "\\":
            i += 2
            continue
        if c == quote:
            state = "code"
            i += 1
            continue
        if WORD.match(text, i):
            rows.append((n, i, "string"))
            i += 5
            continue
        i += 1
    return rows


def owner(text, off):
    """The whole identifier containing offset `off`."""
    s = off
    while s > 0 and (text[s - 1].isalnum() or text[s - 1] == "_"):
        s -= 1
    e = off
    while e < len(text) and (text[e].isalnum() or text[e] == "_"):
        e += 1
    return text[s:e]


def source_files(paths):
    files = []
    for root in paths:
        if os.path.isfile(root):
            files.append(root)
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            if "__pycache__" in dirpath or "build" in dirpath.split(os.sep):
                continue
            for fn in sorted(filenames):
                if fn.endswith(SOURCE_EXT):
                    files.append(os.path.join(dirpath, fn))
    return sorted(set(files))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("paths", nargs="*", default=["engines/ds4"])
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 when any identifier names Metal")
    ap.add_argument("--list", action="store_true",
                    help="print the identifier tokens and the file breakdown")
    ap.add_argument("--env-table", default="envvars.json",
                    help="documented environment names")
    args = ap.parse_args()

    documented = {r["var"] for r in json.load(open(args.env_table))}
    buckets = {}
    lines = {}
    idents = {}
    per_file = {}
    for path in source_files(args.paths):
        if any(marker in path for marker in VENDORED):
            continue
        text = open(path, encoding="utf-8", errors="replace").read()
        for n, off, kind in scan(text):
            tok = owner(text, off)
            if kind == "code":
                cls = "ENV_NAME" if tok in documented else "IDENTIFIER"
            elif ENV_PREFIX.match(tok.upper()):
                cls = "ENV_NAME" if tok in documented else "ENV_NAME_CITED"
            elif kind in ("line_comment", "block_comment"):
                cls = "COMMENT"
            else:
                cls = "STRING"
            buckets[cls] = buckets.get(cls, 0) + 1
            lines.setdefault(cls, set()).add((path, n))
            if cls == "IDENTIFIER":
                idents[tok] = idents.get(tok, 0) + 1
            if cls in ("COMMENT", "STRING"):
                per_file.setdefault((cls, path), set()).add(n)

    order = ("IDENTIFIER", "ENV_NAME", "ENV_NAME_CITED", "COMMENT", "STRING")
    for cls in order:
        print(f"{cls:15s} mentions={buckets.get(cls, 0):4d} "
              f"lines={len(lines.get(cls, set())):4d}")
    if args.list:
        for tok, count in sorted(idents.items(), key=lambda kv: -kv[1]):
            print(f"  identifier {tok} ({count})")
        for (cls, path), ns in sorted(per_file.items()):
            print(f"  {cls:8s} {len(ns):4d} lines  {path}")
    if args.strict and buckets.get("IDENTIFIER", 0):
        print("FAIL: a code identifier still names Metal")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
