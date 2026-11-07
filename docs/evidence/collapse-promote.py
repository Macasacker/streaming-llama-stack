#!/usr/bin/env python3
"""Resolve the `dead #if, later branch manual` nodes the collapse refuses to promote.

Same three-valued logic as scripts/port/collapse_conditionals.py, reused as a module: a node whose
first branch is provably DEAD and whose first later branch is not DEAD is folded by deleting the
dead branches up to that branch and rewriting it from `#elif` to `#if`. Branches after the surviving
one that are provably DEAD go with them; MANUAL and `#else` branches stay. Nothing else is touched,
so this only does what the tool declined to do because the surviving condition is undecided.

Usage (from the repo root, so scripts/port resolves):
  python3 docs/evidence/collapse-promote.py --tokens <names> --never-defined <names> [--apply] FILE...
"""
import argparse
import sys

sys.path.insert(0, "scripts/port")
from collapse_conditionals import Evaluator, parse  # noqa: E402


def split_comment(expr):
    """A trailing `// ...` is not part of a condition - the preprocessor ignores it - so a branch
    whose guard reads `#elif defined(X) // explanation` is still decidable. The comment is kept
    when that branch becomes the guard."""
    head, _, tail = expr.partition("//")
    return head.strip(), ("// " + tail.strip()) if tail.strip() else ""


def verdict(ev, branch):
    cond, _ = split_comment(branch["expr"])
    return ev.decide(cond)[0]


def plan(ev, scope, node, drops, rewrites, log):
    branches = node["branches"]
    if branches:
        first = branches[0]
        # Same gate as the collapse: a node is only work for this sweep if its first condition
        # names a macro in scope. `#if defined(_MSC_VER) / #else` is another chunk's business.
        cond0, _ = split_comment(first["expr"])
        if any(name in cond0 for name in scope):
            v0 = ev.decide(cond0)[0]
        else:
            v0 = None
        if v0 == "DEAD":
            live_at = None
            for idx, b in enumerate(branches[1:], 1):
                if b["kind"] == "else":
                    live_at = idx
                    break
                if verdict(ev, b) == "DEAD":
                    continue
                live_at = idx
                break
            if live_at is not None:
                target = branches[live_at]
                drops.update(range(node["line"], target["dir"]))
                cond, comment = split_comment(target["expr"])
                if target["kind"] == "else":
                    drops.add(target["dir"])
                    drops.add(node["end"])
                else:
                    rewrites[target["dir"]] = ("#if " + cond + ((" " + comment) if comment else "")).rstrip()
                log.append((node["line"], target["dir"], target["kind"],
                            cond + ((" " + comment) if comment else "")))
                for b in branches[live_at + 1:]:
                    if verdict(ev, b) == "DEAD":
                        drops.update(range(b["dir"], b["bend"] + 1))
                        log.append((node["line"], b["dir"], "drop-dead-later",
                                    b["expr"].strip()))
    for child in node["children"]:
        plan(ev, scope, child, drops, rewrites, log)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokens", default="")
    ap.add_argument("--never-defined", default="")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()
    scope = [t for t in args.tokens.split(",") if t]
    ev = Evaluator(scope, [t for t in args.never_defined.split(",") if t])
    for path in args.files:
        lines, root = parse(path)
        drops, rewrites, log = set(), {}, []
        for child in root["children"]:
            plan(ev, scope, child, drops, rewrites, log)
        out = [rewrites.get(i, line) for i, line in enumerate(lines, 1) if i not in drops]
        if log:
            print(path)
            for line, dirline, kind, expr in log:
                print(f"   node@{line}: {kind} @{dirline} :: {expr[:80]}")
            print(f"   {len(drops)} lines dropped, {len(rewrites)} guards rewritten")
        if args.apply and out != lines:
            open(path, "w", encoding="utf-8").write("\n".join(out))


if __name__ == "__main__":
    main()
