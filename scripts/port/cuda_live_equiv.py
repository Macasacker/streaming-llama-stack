#!/usr/bin/env python3
"""Compare the live (preprocessed) text of llama ggml sources between two trees.

`preprocess_equiv.py` preprocesses both sides with one flag set, which is wrong for a sweep that
touches a **header**: the "before" file then includes the after tree's headers and the comparison
proves nothing. This tool preprocesses each side with its own tree's include directories, so a
change in `common.cuh` is compared against the `common.cuh` the before-side TU actually includes.

    git worktree add build/wt-base HEAD
    python3 scripts/port/cuda_live_equiv.py build/wt-base . \
        engines/streaming-llama/prefill_llama.cpp/ggml/src/ggml-cuda/common.cuh \
        engines/streaming-llama/prefill_llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu

`gcc -E -fdirectives-only` resolves conditionals without expanding macros, so `__LINE__` inside a
`CUDA_CHECK` does not move the diff. `.cu` is preprocessed as C++ for the same reason - nothing is
compiled here, only directives evaluated - and the CUDA and CCCL include directories are passed
explicitly because the build passes them as `-isystem`, which this tool does not read.

A `.cuh` fragment preprocesses on its own as long as its includes resolve; a file whose conditionals
the compiler cannot settle (no `-DGGML_USE_CUDA` on a CPU-only build) is compared with those flags
changed, not with a different tool.
"""
import argparse, difflib, os, subprocess, sys

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))) + "/..")

DEFS = ["-DGGML_USE_CUDA", "-DGGML_USE_CPU", "-DGGML_CUDA_USE_GRAPHS",
        "-DGGML_CUDA_PEER_MAX_BATCH_SIZE=128", "-DGGML_SCHED_MAX_COPIES=4",
        "-D_GNU_SOURCE", "-D_XOPEN_SOURCE=600", "-DNDEBUG"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("base", help="tree at the parent commit")
    ap.add_argument("new", help="tree under review")
    ap.add_argument("--cuda-include", default="/opt/cuda/include")
    ap.add_argument("--cc", default="g++")
    ap.add_argument("--no-cuda-defines", action="store_true",
                    help="preprocess without -DGGML_USE_CUDA, for a CPU-only build's text")
    ap.add_argument("files", nargs="+", help="paths relative to each tree's repo root")
    args = ap.parse_args()

    defs = [d for d in DEFS if not args.no_cuda_defines or not d.startswith("-DGGML_USE_CUDA")]
    tmp = os.path.join(ROOT, "build", "cuda-live-equiv")
    os.makedirs(tmp, exist_ok=True)
    bad = 0
    for rel in args.files:
        name = os.path.basename(rel)
        if not os.path.exists(os.path.join(args.new, rel)):
            print(f"{rel}: gone in the new tree - a deletion is not a sweep, nothing to compare")
            continue
        if not os.path.exists(os.path.join(args.base, rel)):
            print(f"{rel}: not in the base tree - added here, nothing to compare")
            continue
        outs = []
        for label, tree in (("before", args.base), ("after", args.new)):
            # The include set is the CUDA backend's own: its directory, ggml/src, ggml/include and
            # the llama tree's include/, plus the CUDA and CCCL headers the build passes as
            # -isystem. Each side resolves inside its own tree.
            llama = os.path.join(tree, rel[:rel.index("prefill_llama.cpp") + len("prefill_llama.cpp")])
            ggml = os.path.join(llama, "ggml")
            inc = [f"-I{os.path.join(ggml, 'src', 'ggml-cuda')}", f"-I{os.path.join(ggml, 'src')}",
                   f"-I{os.path.join(ggml, 'include')}", f"-I{os.path.join(llama, 'include')}",
                   f"-I{args.cuda_include}", f"-I{args.cuda_include}/cccl"]
            out = os.path.join(tmp, f"{name}.{label}")
            cmd = [args.cc, "-E", "-fdirectives-only", "-P", "-x", "c++", *inc, *defs,
                   os.path.join(tree, rel), "-o", out]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode:
                print(f"{rel}: preprocess FAILED on the {label} side: {r.stderr.split(chr(10))[0][:160]}")
                bad += 1
                outs = None
                break
            outs.append(out)
        if not outs:
            continue
        bl = [l for l in open(outs[0], encoding="utf-8").read().split("\n") if l.strip()]
        al = [l for l in open(outs[1], encoding="utf-8").read().split("\n") if l.strip()]
        diff = [d for d in difflib.unified_diff(bl, al, lineterm="", n=0)
                if d[:1] in "+-" and d[:3] not in ("+++", "---")]
        print(f"{rel}: live lines {len(bl)} -> {len(al)}, differing {len(diff)}")
        for d in diff[:10]:
            print("   ", d[:160])
        bad += 1 if diff else 0
    print("files with differences or failures:", bad)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
