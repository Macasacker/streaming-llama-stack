#!/usr/bin/env python3
"""Show what a mechanical sweep actually changed in the compiled code.

Preprocesses a "before" file and an "after" file with conditionals resolved but macros left alone
(`gcc -E -fdirectives-only -P`), so dead branches vanish from both sides and the diff reports only
what the compiler saw. With `--rename-file` the same rename is applied to the "before" text first,
so a rename sweep cancels itself out and any surviving difference is a real change.

  git show HEAD:engines/ds4/ds4.c > /tmp/before.c
  python3 scripts/port/preprocess_equiv.py /tmp/before.c engines/ds4/ds4.c
  python3 scripts/port/preprocess_equiv.py --cc /opt/cuda/bin/nvcc --include engines/ds4 \
      --rename-file /tmp/rename.json /tmp/before.cu engines/ds4/ds4_cuda.cu

nvcc has no `-fdirectives-only`, so it fully expands macros: `__LINE__` values inside trace
messages differ when lines were deleted above them, which is expected.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile


def preprocess(cc, path, include, extra, lang):
    cmd = [cc, "-E", "-P"]
    if cc.endswith("gcc") or cc.endswith("clang") or cc.endswith("g++") or cc.endswith("clang++"):
        cmd.append("-fdirectives-only")
    if lang:
        cmd += ["-x", lang]
    if include:
        cmd += ["-I" + include]
    cmd += extra + [path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"preprocess failed for {path}: {r.stderr[:400]}", file=sys.stderr)
        sys.exit(1)
    # Drop the line markers nvcc emits ("# 42 \"file\""), which shift with deletions.
    return "\n".join(l for l in r.stdout.split("\n") if not re.match(r"^# \d+ ", l))


def rename(text, mapping):
    for old, new in mapping.get("exact", []):
        text = re.sub(r"\b" + re.escape(old) + r"\b", new, text)
    for old, new in mapping.get("prefix", []):
        text = re.sub(r"\b" + re.escape(old) + r"[A-Za-z0-9_]*",
                      lambda m, n=new, o=old: n + m.group(0)[len(o):], text)
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cc", default="gcc")
    ap.add_argument("--include", default="", help="include directory, e.g. engines/ds4")
    ap.add_argument("--extra", default="", help="extra compiler flags, space separated")
    ap.add_argument("--rename-file", help="JSON {\"exact\": [[old,new]...], \"prefix\": [[old,new]...]}")
    ap.add_argument("before")
    ap.add_argument("after")
    args = ap.parse_args()
    mapping = json.load(open(args.rename_file)) if args.rename_file else None
    extra = args.extra.split() if args.extra else []
    lang = "c" if args.after.endswith((".c", ".h")) else "c++"

    with tempfile.NamedTemporaryFile("w", suffix=".i", delete=False) as f:
        before_text = open(args.before).read()
        f.write(before_text if not mapping else rename(before_text, mapping))
        before_path = f.name
    try:
        a = preprocess(args.cc, before_path, args.include, extra, lang).split("\n")
        b = preprocess(args.cc, args.after, args.include, extra, lang).split("\n")
    finally:
        os.unlink(before_path)
    diffs = [(i + 1, x, y) for i, (x, y) in enumerate(zip(a, b)) if x != y]
    tail = abs(len(a) - len(b))
    print(f"{len(a)} vs {len(b)} live lines, {len(diffs)} differing position(s), {tail} length delta")
    for i, x, y in diffs[:25]:
        print(f"  {i}: -{x.strip()[:90]!r}")
        print(f"  {i}: +{y.strip()[:90]!r}")
    return 1 if (diffs or tail) else 0


if __name__ == "__main__":
    sys.exit(main())
