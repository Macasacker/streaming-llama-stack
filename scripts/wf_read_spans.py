#!/usr/bin/env python3
"""Emit the span list a DSV4.1 prefill stage reads, for scripts/wf_read_bench.cpp.

A stage is one transformer block: the three routed expert tensors plus that block's
dense tensors. Routing activates `--active` of the 384 experts, and the engine reads
only the activated slices, so the spans are the active slices scattered through each
tensor plus the dense tensors whole. Expert ids are taken from a fixed seed so two
runs of this script name the same bytes (a comparison, not a sample).

  scripts/wf_read_spans.py --block 22 --active 262 -o /tmp/blk22.spans
"""
import argparse
import importlib.util
import pathlib
import random
import struct
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location('rp', REPO / 'scripts' / 'repack_dsv41_dspark_sidecar.py')


def load_reader():
    mod = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(mod)
    return mod


def routed_expert_slice(tensor, index, expert_count, rp):
    """Byte range of one expert inside a [ne0, ne1, experts] projection.

    Block-quantized types round to whole blocks, so the slice size is the packed
    size of one expert's ne0*ne1 elements, not elements x a per-element size.
    """
    ne0, ne1, experts = tensor['dims']
    if experts != expert_count:
        raise ValueError('%s carries %d experts, expected %d' % (tensor['name'], experts, expert_count))
    one = rp.nbytes(tensor['type'], ne0 * ne1)
    return tensor['abs'] + index * one, one


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--file', default='/home/mac/workspace/local-llm-playground/ds4-models/'
                                     'DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf')
    ap.add_argument('--block', type=int, default=22)
    ap.add_argument('--active', type=int, default=262,
                    help='experts the router activates in this stage (the measured prefill figure)')
    ap.add_argument('--experts', type=int, default=384)
    ap.add_argument('--seed', type=int, default=22)
    ap.add_argument('-o', '--out', default='-')
    args = ap.parse_args(argv)

    rp = load_reader()
    gguf = rp.read_gguf(pathlib.Path(args.file))
    block = 'blk.%d.' % args.block
    routed, dense = [], []
    for t in gguf['tensors']:
        if not t['name'].startswith(block):
            continue
        if len(t['dims']) == 3:
            routed.append(t)
        else:
            dense.append(t)
    if len(routed) != 3:
        raise SystemExit('%s: expected 3 routed projections, found %d' % (block, len(routed)))

    rng = random.Random(args.seed)
    active = sorted(rng.sample(range(args.experts), min(args.active, args.experts)))
    spans = []
    for t in routed:
        for eid in active:
            spans.append(routed_expert_slice(t, eid, args.experts, rp))
    for t in dense:
        spans.append((t['abs'], t['bytes']))

    out = sys.stdout if args.out == '-' else open(args.out, 'w')
    for off, ln in spans:
        out.write('%d %d\n' % (off, ln))
    if out is not sys.stdout:
        out.close()
    total = sum(ln for _, ln in spans)
    sys.stderr.write('%s: %d spans (%d experts x %d tensors + %d dense), %.2f GiB\n'
                     % (block, len(spans), len(active), len(routed), len(dense), total / 2**30))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
