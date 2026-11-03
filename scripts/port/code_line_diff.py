#!/usr/bin/env python3
"""Compare the *code* lines the compiler sees, before and after a sweep.

`preprocess_equiv.py` shows the whole diff. This is the variant for a chunk that
also edits comments: a changed comment near the top shifts every later line, so
the positional diff becomes noise. Preprocess both files with
`-fdirectives-only -P` (conditionals resolved, macros left alone), drop comment
and blank lines, and diff with no context. A deletion sweep then reports 0
changed code lines apart from the edits it intended; a port reports only the
regions it rewrote.

    git show HEAD:engines/streaming-llama/prefill_llama.cpp/src/llama-weight-fetcher.cpp \
        > build/before.cpp
    python3 scripts/port/code_line_diff.py \
        build/before.cpp engines/streaming-llama/prefill_llama.cpp/src/llama-weight-fetcher.cpp \
        llama-weight-fetcher.cpp

Triples repeat: BEFORE AFTER NAME. NAME is the basename the build's
compile_commands.json entry ends with, which is where the include directories
and defines come from.
"""
import argparse, difflib, json, os, re, subprocess, sys, tempfile

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--compile-commands", default=os.path.join(ROOT, "build/llama/compile_commands.json"),
                    help="compile_commands.json of the configured build")
    ap.add_argument("--include", action="append", default=[],
                    help="extra include directory, e.g. a TU's own tool directory")
    ap.add_argument("--cuda-include", default=None,
                    help="CUDA include directory, for __has_include(<cuda_runtime.h>)")
    ap.add_argument("--show", type=int, default=40, help="changed lines to print")
    ap.add_argument("triples", nargs="+")
    args = ap.parse_args()

    if len(args.triples) % 3:
        raise SystemExit("usage: code_line_diff.py BEFORE AFTER NAME [...]")

    cc = json.load(open(args.compile_commands, encoding="utf-8"))

    def flags(name):
        for e in cc:
            if e["file"].endswith("/" + name):
                # The build's own -I and -D flags, so the conditionals resolve the
                # way the compiler resolves them. Without the defines the CUDA and
                # GDS regions vanish from both sides and the comparison proves less.
                return " ".join(re.findall(r"-[ID]\S+", e["command"]))
        raise SystemExit(f"no compile command for {name}")

    def code_lines(path, name):
        tmp = tempfile.mktemp(suffix=".i")
        extra = " ".join(f"-I{d}" for d in args.include)
        cuda = f"-I{args.cuda_include}" if args.cuda_include else ""
        cmd = (f"g++ -E -fdirectives-only -P -std=gnu++17 -x c++ {cuda} {extra} "
               f"{flags(name)} {path} -o {tmp}")
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if r.returncode:
            sys.stdout.write(r.stderr[:2000])
            raise SystemExit(f"preprocess failed for {path}")
        try:
            return [ln for ln in open(tmp, encoding="utf-8").read().split("\n")
                    if ln.strip() and not ln.lstrip().startswith("//")]
        finally:
            os.unlink(tmp)

    for i in range(0, len(args.triples), 3):
        before, after, name = args.triples[i:i + 3]
        b, a = code_lines(before, name), code_lines(after, name)
        diff = [d for d in difflib.unified_diff(b, a, lineterm="", n=0)
                if d[:1] in "+-" and d[:3] not in ("+++", "---")]
        print(f"{name}: before {len(b)} code lines, after {len(a)}, changed {len(diff)}")
        for d in diff[:args.show]:
            print("   ", d[:160])
        if len(diff) > args.show:
            print(f"    ... {len(diff) - args.show} more")


if __name__ == "__main__":
    main()
