#!/usr/bin/env python3
"""Step 6 measurement: does the second turn of a session reuse the first turn's prefix, and what
the append costs.

docs/DECODE_GAP_PLAN.md Step 6 accepts at >= 70 % of the second turn's prompt tokens reported as
reused. The engine already says this - `timings.cache_n` and `usage.prompt_tokens_details.cached_tokens`
come from `reused_prefix_tokens` in `slstack/combo_single.py:540` - and the only run so far was a
cold first turn, which correctly reported `cache_n=0`. This sends turn 1, then the same transcript
plus an assistant reply and one more user line, and reports the ratio it got.

With `--turns N` it keeps appending short turns, which is the agentic cadence: reuse is the thing
that saves the prompt head, and `prompt_ms` on those turns is what a turn actually costs. That is
the measurement for the resident producer's routing (docs/SESSION_FAILURE_PLAN.md, item 7), which
is set once at pipeline build and therefore decides every later append:

    scripts/start-dsv41-single-262k.sh --config rtx.tail.json ...     # env_overrides: ROUTED=tail
    python3 scripts/dsv41_reuse_probe.py --turns 5 --expect-routed tail
    scripts/start-dsv41-single-262k.sh --config rtx.local.json ...    # default, now '1'
    python3 scripts/dsv41_reuse_probe.py --turns 5 --expect-routed 1

Run it against a started stack:

    scripts/start-dsv41-single-262k.sh --config rtx.local.json \
        --model ~/workspace/local-llm-playground/ds4-models/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
    python3 scripts/dsv41_reuse_probe.py

The launcher stays attached so `Ctrl+C` is the shutdown; run it in its own terminal, or pass
`--detach` and stop later with `scripts/start-dsv41-single.sh --preset 262k --stop`.

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


def recorded_routed(root=REPO):
    """The routing the newest resident producer recorded (launch.json), or None.

    The producer reads LLAMADECODE_PREFILL_ROUTED once, when its pipeline is built, so an A/B of
    append cost has to name the mode each side actually ran - not the mode the shell exported.
    """
    files = sorted(root.glob('runtime/single-engine/ultra-*/launch.json'), key=lambda p: p.stat().st_mtime)
    if not files:return None
    try:return json.loads(files[-1].read_text(encoding='utf-8')).get('routed_prefill')
    except (OSError, ValueError):return None


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
    parser.add_argument('--turns', type=int, default=2,
                        help='turns to send; turns after the first append a short line, which is the '
                             'agentic cadence whose cost the resident producer\'s routing decides')
    parser.add_argument('--expect-routed', choices=('0', '1', 'tail'),
                        help='fail if the resident producer recorded a different LLAMADECODE_PREFILL_ROUTED')
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
    if args.expect_routed:
        recorded = recorded_routed()
        if recorded != args.expect_routed:
            raise SystemExit('the resident producer recorded routed_prefill=%s, not %s - restart it '
                             'with env_overrides.LLAMADECODE_PREFILL_ROUTED=%s before measuring'
                             % (recorded, args.expect_routed, args.expect_routed))
        print('recorded routed_prefill=%s' % recorded)
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

    # The turns an agent actually pays for: a few dozen tokens appended to the resident producer.
    transcript = second
    appends = []
    for index in range(3, args.turns + 1):
        reply = ((turn2.get('choices') or [{}])[0].get('message') or {}).get('content', '')
        transcript = transcript + [{'role': 'assistant', 'content': reply},
                                   {'role': 'user', 'content': 'Next: one short line, then stop.'}]
        turn, wall = post(args.base, {'messages': transcript, 'max_tokens': args.max_tokens})
        row = report('turn %d' % index, turn, wall)
        row['appended'] = (row['prompt_n'] or 0)
        appends.append(row)
        turn2 = turn
    if appends:
        seconds = sorted(row['prompt_ms'] or 0 for row in appends)
        print('append prefill over %d turns: median %.1f s, best %.1f s, worst %.1f s'
              % (len(appends), seconds[len(seconds) // 2] / 1000.0, seconds[0] / 1000.0,
                 seconds[-1] / 1000.0))

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
