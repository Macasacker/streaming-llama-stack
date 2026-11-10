#!/usr/bin/env python3
"""Run scripts/wf_read_bench.cpp over one block's real spans, cold each time.

The card is busy, so the read path has to be measured on disk alone. Each case
drops the block's bytes out of the page cache first (`POSIX_FADV_DONTNEED`), which
is the state a 340 GiB file is in on a 93 GiB box: nothing of that stage is
resident when the stage is requested. Pass 0 is therefore cold NVMe and the later
passes are what retention across turns would buy.

  scripts/wf_read_matrix.py --block 22 --active 262
"""
import argparse
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = '/home/mac/workspace/local-llm-playground/ds4-models/' \
        'DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf'
BENCH = '/tmp/wf_read_bench'
MIB = 1024 * 1024

CASES = [
    ('engine shape: 8 rings x QD4, 4 MiB', 8, 4, 4 * MIB, 0),
    ('1 ring x QD4, 4 MiB', 1, 4, 4 * MIB, 0),
    ('2 rings x QD4, 4 MiB', 2, 4, 4 * MIB, 0),
    ('4 rings x QD4, 4 MiB', 4, 4, 4 * MIB, 0),
    ('8 rings x QD1, 4 MiB', 8, 1, 4 * MIB, 0),
    ('8 rings x QD8, 4 MiB', 8, 8, 4 * MIB, 0),
    ('8 rings x QD32, 4 MiB', 8, 32, 4 * MIB, 0),
    ('8 rings x QD4, 2 MiB', 8, 4, 2 * MIB, 0),
    ('8 rings x QD4, 8 MiB', 8, 4, 8 * MIB, 0),
    ('8 rings x QD4, 16 MiB', 8, 4, 16 * MIB, 0),
    ('8 rings x QD4, 32 MiB', 8, 4, 32 * MIB, 0),
    ('1 ring x QD8, 8 MiB', 1, 8, 8 * MIB, 0),
    ('O_DIRECT 8 x QD4, 4 MiB', 8, 4, 4 * MIB, 1),
    ('O_DIRECT 1 x QD8, 4 MiB', 1, 8, 4 * MIB, 1),
]


def cached_kib():
    for line in open('/proc/meminfo'):
        if line.startswith('Cached:'):
            return int(line.split()[1])
    return 0


def drop(path, lo, hi):
    fd = os.open(path, os.O_RDONLY)
    os.posix_fadvise(fd, lo, hi - lo, os.POSIX_FADV_DONTNEED)
    os.close(fd)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--block', type=int, default=22)
    ap.add_argument('--active', type=int, default=262)
    ap.add_argument('--file', default=MODEL)
    ap.add_argument('--passes', type=int, default=2)
    ap.add_argument('--only', default='', help='substring filter on the case label')
    args = ap.parse_args(argv)

    spans = '/tmp/blk%d.spans' % args.block
    subprocess.run([sys.executable, os.path.join(REPO, 'scripts', 'wf_read_spans.py'),
                    '--block', str(args.block), '--active', str(args.active),
                    '--file', args.file, '-o', spans], check=True)
    pairs = [(int(a), int(b)) for a, b in
             (line.split() for line in open(spans))]
    lo = min(o for o, _ in pairs)
    hi = max(o + n for o, n in pairs)

    if not os.path.exists(BENCH):
        subprocess.run(['g++', '-O2', '-std=c++17', os.path.join(REPO, 'scripts', 'wf_read_bench.cpp'),
                        '-luring', '-o', BENCH], check=True)

    print('stage %.2f GiB, %d spans, file window %.2f-%.2f GiB'
          % (sum(n for _, n in pairs) / 2**30, len(pairs), lo / 2**30, hi / 2**30))
    for label, threads, qd, block, direct in CASES:
        if args.only and args.only not in label:
            continue
        drop(args.file, lo, hi)
        before = cached_kib()
        r = subprocess.run([BENCH, '--spans', spans, '--file', args.file,
                            '--threads', str(threads), '--qd', str(qd),
                            '--block', str(block), '--direct', str(direct),
                            '--passes', str(args.passes)],
                           capture_output=True, text=True)
        print('\n%s   rc=%d  cached %d -> %d KiB' % (label, r.returncode, before, cached_kib()))
        for line in r.stderr.strip().split('\n'):
            print('  ' + line)
        if r.returncode:
            sys.stdout.flush()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
