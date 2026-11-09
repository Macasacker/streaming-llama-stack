#!/usr/bin/env python3
"""Step 6 measurement: does the second turn of a session reuse the first turn's prefix.

docs/DECODE_GAP_PLAN.md Step 6 accepts at >= 70 % of the second turn's prompt tokens reported as
reused. The engine already says this - `timings.cache_n` and `usage.prompt_tokens_details.cached_tokens`
come from `reused_prefix_tokens` in `slstack/combo_single.py:540` - and the only run so far was a
cold first turn, which correctly reported `cache_n=0`. This sends turn 1, then the same transcript
plus an assistant reply and one more user line, and reports the ratio it got.

Run it against a started stack:

    scripts/start-dsv41-single-262k.sh --config rtx.local.json \
        --model ~/workspace/local-llm-playground/ds4-models/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
    python3 scripts/dsv41_reuse_probe.py

It needs the card. Turn 2 is byte-identical to turn 1 up to the appended line, so a zero here is
the append path not being taken, not a prompt that drifted.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / 'runtime' / 'bench-50k.txt'
ACCEPT_REUSE = 0.70


def post(base, body, timeout=1800):
    request = urllib.request.Request(base.rstrip('/') + '/v1/chat/completions',
                                     data=json.dumps(body).encode(),
                                     headers={'content-type': 'application/json'})
    start = time.time()
    raw = urllib.request.urlopen(request, timeout=timeout).read()
    return json.loads(raw), time.time() - start


def report(label, data, wall):
    usage = data.get('usage') or {}
    timings = data.get('timings') or {}
    details = usage.get('prompt_tokens_details') or {}
    prompt = usage.get('prompt_tokens') or 0
    cached = timings.get('cache_n', details.get('cached_tokens', 0)) or 0
    completion = usage.get('completion_tokens') or 0
    decode_ms = timings.get('predicted_ms') or 0
    out = {'label': label, 'prompt_tokens': prompt, 'cache_n': cached,
           'prompt_n': timings.get('prompt_n'), 'prompt_ms': timings.get('prompt_ms'),
           'completion_tokens': completion, 'wall_s': round(wall, 1),
           'reuse': (cached / prompt) if prompt else 0.0,
           'decode_tok_s': (completion * 1000.0 / decode_ms) if decode_ms and completion else None}
    print('%-7s prompt %6d  cache_n %6d (%.1f %%)  prompt_n %6s  prefill %8.1f s  '
          'completion %4d  wall %6.1f s%s'
          % (label, prompt, cached, 100 * out['reuse'], out['prompt_n'],
             (out['prompt_ms'] or 0) / 1000.0, completion, wall,
             ('  decode %.2f tok/s' % out['decode_tok_s']) if out['decode_tok_s'] else ''))
    return out


def prompt_text(chars):
    if not BENCH.is_file():
        raise SystemExit('missing %s - the probe needs a stable long prompt' % BENCH)
    return BENCH.read_text(encoding='utf-8', errors='replace')[:chars]


def gpu_used():
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=memory.used',
                              '--format=csv,noheader,nounits'], capture_output=True, text=True,
                             timeout=20).stdout.strip()
        return int(out.splitlines()[0])
    except Exception:
        return -1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base', default='http://127.0.0.1:8088')
    parser.add_argument('--prompt-chars', type=int, default=8000,
                        help='turn 1 prompt size; the same text must prefix turn 2')
    parser.add_argument('--max-tokens', type=int, default=64)
    parser.add_argument('--ignore-gpu', action='store_true',
                        help='run anyway (the lease gate will refuse admission if the card is busy)')
    args = parser.parse_args(argv)

    used = gpu_used()
    if used > 4096 and not args.ignore_gpu:
        raise SystemExit('the card holds %d MiB - stop whatever serves a model first, this probe '
                         'will otherwise only see an admission refusal' % used)
    try:
        health = urllib.request.urlopen(args.base + '/health', timeout=10).read()
    except Exception as exc:
        raise SystemExit('%s is not answering /health (%r). Start it with '
                         'scripts/start-dsv41-single-262k.sh first.' % (args.base, exc))
    print('health', health[:160].decode('utf-8', 'replace'))

    text = prompt_text(args.prompt_chars)
    first = [{'role': 'user', 'content': text +
              '\nIn one line: what did the harbor master record first?'}]
    turn1, wall1 = post(args.base, {'messages': first, 'max_tokens': args.max_tokens})
    one = report('turn 1', turn1, wall1)
    answer = ((turn1.get('choices') or [{}])[0].get('message') or {}).get('content', '')
    if not answer:
        print('turn 1 returned no content:', json.dumps(turn1)[:300])
        return 1

    second = first + [{'role': 'assistant', 'content': answer},
                      {'role': 'user', 'content': 'Now list the three numbers you mentioned, one per line.'}]
    turn2, wall2 = post(args.base, {'messages': second, 'max_tokens': args.max_tokens})
    two = report('turn 2', turn2, wall2)

    verdict = 'PASS' if two['reuse'] >= ACCEPT_REUSE else 'FAIL'
    print('%s: turn 2 reused %.1f %% of %d prompt tokens (bar %.0f %%)'
          % (verdict, 100 * two['reuse'], two['prompt_tokens'], 100 * ACCEPT_REUSE))
    print('results json: %s' % json.dumps([one, two]))
    if verdict == 'FAIL':
        print('Read the producer log for why: grep -aE "restart_prompt|append|PROMPT_HEAD_COMPLETE|'
              'retire|prefix divergence" runtime/single-engine/ultra-*/stderr.log')
    return 0 if verdict == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
