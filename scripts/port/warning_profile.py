#!/usr/bin/env python3
"""Compare the warning profile of a set of TUs between two trees.

The other regression check for a mechanical sweep: compile the same translation
units with the same commands in a worktree at the parent commit and in the
working tree, and compare the warning texts with file, line and column stripped.
A deletion sweep should change nothing; a rename should differ only in the
renamed symbol's own warnings.

    git worktree add build/wt-base HEAD
    python3 scripts/port/warning_profile.py build/wt-base .

Both arguments are trees that contain the same relative source paths. The
compile commands are read from a configured build, and their absolute paths are
rewritten to each tree, so the include directories resolve inside the tree being
compiled.
"""
import argparse, collections, json, os, re, subprocess, sys

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--compile-commands", default=os.path.join(ROOT, "build/llama/compile_commands.json"))
    ap.add_argument("--scratch", default=os.path.join(ROOT, "build"),
                    help="where the throwaway objects go")
    ap.add_argument("--list", default="llama-weight-fetcher.cpp,llama-dsv41-weights.cpp,"
                                      "llama-dsv41-sweep.cpp,llama-weight-fetcher-adapter.cpp,"
                                      "llama-prefill-stream.cpp,main.cpp",
                    help="comma-separated TU basenames to compare")
    ap.add_argument("base", help="tree at the parent commit")
    ap.add_argument("new", help="tree under review")
    args = ap.parse_args()

    cc = json.load(open(args.compile_commands, encoding="utf-8"))

    def entry(name):
        for e in cc:
            if e["file"].endswith("/" + name):
                return e
        raise SystemExit(f"no compile command for {name}")

    def compile_in(tree, name, out):
        e = entry(name)
        parts, skip = [], 0
        for token in e["command"].split():
            if skip:
                skip -= 1
                continue
            if token == "-o":
                parts += ["-o", out]
                skip = 1
                continue
            parts.append(token.replace(ROOT, tree))
        return subprocess.run(" ".join(parts), shell=True, cwd=e["directory"],
                              capture_output=True, text=True)

    def warnings(text):
        found = []
        for line in text.splitlines():
            m = re.match(r"^(.*?):\d+:\d+: (warning|error): (.*$)", line)
            if m:
                path, kind, msg = m.groups()
                found.append(f"{os.path.basename(path)}: {kind}: {msg}")
        return found

    os.makedirs(args.scratch, exist_ok=True)

    def profile(tree, tag):
        acc = []
        for name in args.list.split(","):
            out = os.path.join(args.scratch, f"warn-{tag}-{name}")
            acc += warnings(compile_in(os.path.abspath(tree), name, out).stderr)
            try:
                os.unlink(out)
            except OSError:
                pass
        return acc

    b, n = profile(args.base, "base"), profile(args.new, "new")
    cb, cn = collections.Counter(b), collections.Counter(n)
    print(f"base occurrences {len(b)}, new occurrences {len(n)}, "
          f"distinct base {len(cb)}, new {len(cn)}")
    for t in sorted(set(cn) - set(cb)):
        print("  + NEW", t)
    for t in sorted(set(cb) - set(cn)):
        print("  - REMOVED", t)
    for t in sorted(set(cb) & set(cn)):
        if cb[t] != cn[t]:
            print(f"  ~ count {cb[t]} -> {cn[t]}: {t}")


if __name__ == "__main__":
    main()
