#!/usr/bin/env python3
"""Prove that an edit to comments changed no code, and show the code it did change.

    python3 scripts/port/comment_code_diff.py BEFORE AFTER

Both files are stripped of comments - string and character literals are kept,
including the ones that span lines - and the remaining code lines are compared
with no context. A comment sweep therefore reports 0 changed lines even though
every comment it reworded shifted the line numbers below it, which is what makes
`preprocess_equiv.py` and `git diff` read as noise on this kind of chunk.
A sweep that also rewords printed strings reports exactly those string lines.

A file that uses `__LINE__` in a message is outside this proof: deleting comment
lines moves the reported number. Grep for it before trusting the result.

The comparison ignores leading whitespace, so reflowing a comment that shared a
line with code cannot hide a real change in that code.
"""
import argparse
import difflib
import sys


def strip_comments(text):
    """The text with every comment removed. String contents are preserved."""
    out = []
    state = "code"
    quote = '"'
    i, L = 0, len(text)
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
                    if text[i] == "\n":
                        out.append("\n")
                    i += 1
                i += 2
                continue
            if c == '"' or c == "'":
                state = "string"
                quote = c
                out.append(c)
                i += 1
                continue
            out.append(c)
            i += 1
            continue
        out.append(c)
        if c == "\\" and i + 1 < L:
            out.append(text[i + 1])
            i += 2
            continue
        if c == quote:
            state = "code"
        i += 1
    return "".join(out)


def code_lines(text):
    return [line.strip() for line in strip_comments(text).split("\n") if line.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("--show", type=int, default=60, help="changed lines to print")
    args = ap.parse_args()

    a = code_lines(open(args.before, encoding="utf-8", errors="replace").read())
    b = code_lines(open(args.after, encoding="utf-8", errors="replace").read())
    diff = [line for line in difflib.unified_diff(a, b, lineterm="", n=0)
            if line and (line[0] in "+-" and line[:3] not in ("---", "+++"))]
    changed = sum(1 for line in diff if line.startswith("+"))
    removed = sum(1 for line in diff if line.startswith("-"))
    print(f"{len(a)} vs {len(b)} code lines, {removed} removed, {changed} added")
    for line in diff[:args.show]:
        print(line[:200])
    if len(diff) > args.show:
        print(f"... {len(diff) - args.show} more")
    return 1 if diff else 0


if __name__ == "__main__":
    sys.exit(main())
