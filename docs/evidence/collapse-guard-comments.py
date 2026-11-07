#!/usr/bin/env python3
"""Rewrite trailing `#endif // <condition>` / `#else // <condition>` comments that name a macro
this sweep pinned to undefined, so they describe the guard that actually survives.

Comment-only: no code line is touched. The replacement is the opening directive's own condition as
it now reads; for an `#else` the comment becomes the negation of that condition when the condition
is a single defined() term, and is dropped otherwise.

Usage (from the repo root, so scripts/port resolves):
  python3 docs/evidence/collapse-guard-comments.py --tokens <names> [--apply] FILE...
"""
import argparse
import re
import sys

sys.path.insert(0, "scripts/port")
from collapse_conditionals import parse  # noqa: E402

NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def names(text, pinned):
    return set(NAME.findall(text)) & pinned


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokens", default="", help="comma-separated macros the sweep pinned")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()
    pinned = {t for t in args.tokens.split(",") if t}
    for path in args.files:
        lines, root = parse(path)
        rewrites = {}
        report = []

        def visit(node, cond):
            own = node["branches"][0]["expr"].strip() if node["branches"] else cond
            if node["end"] is not None:
                line = lines[node["end"] - 1]
                m = re.match(r"^(\s*#\s*endif\s*)//\s*(.*)$", line)
                if m and names(m.group(2), pinned):
                    rewrites[node["end"]] = m.group(1) + "// " + own
                    report.append((node["end"], m.group(2), own))
            for b in node["branches"][1:]:
                if b["kind"] != "else":
                    continue
                line = lines[b["dir"] - 1]
                m = re.match(r"^(\s*#\s*else\s*)//\s*(.*)$", line)
                if m and names(m.group(2), pinned):
                    single = re.fullmatch(r"defined\(([A-Za-z_][A-Za-z0-9_]*)\)", own)
                    new = ("// !defined(%s)" % single.group(1)) if single else None
                    rewrites[b["dir"]] = (m.group(1) + (new or "")).rstrip()
                    report.append((b["dir"], m.group(2), new or "(dropped)"))
            # An `#elif` whose own trailing comment names a pinned macro: rewrite to its own text
            # minus the stale comment terms.
            for b in node["branches"][1:]:
                if b["kind"] != "elif":
                    continue
                line = lines[b["dir"] - 1]
                m = re.match(r"^(\s*#\s*elif\s+)(.*?)\s*//\s*(.*)$", line)
                if m and names(m.group(3), pinned):
                    rewrites[b["dir"]] = m.group(1) + m.group(2)
                    report.append((b["dir"], m.group(3), "guard only"))
            for child in node["children"]:
                visit(child, own)

        for child in root["children"]:
            visit(child, "")
        if report:
            print(path)
            for line, old, new in report:
                print(f"   {line}: // {old[:70]}  ->  {new[:70]}")
            print(f"   {len(report)} comments rewritten")
        if args.apply and rewrites:
            out = [rewrites.get(i, line) for i, line in enumerate(lines, 1)]
            open(path, "w", encoding="utf-8").write("\n".join(out))


if __name__ == "__main__":
    main()
