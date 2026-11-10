#!/usr/bin/env python3
"""Decode step budget from a llama-dsv41-prefill producer log.

Step 1 of docs/DECODE_GAP_PLAN.md: before changing any cache placement, a run has to
answer "how many of these 2.4 GB came from disk". This reads a producer stderr log
(runtime/single-engine/ultra-*/stderr.log, or the rotated file) and reports

    rate        median/mean/p10/p90 of LLAMADECODE_STEP step_ms, as tok/s
    bytes       per-step footprint and what each tier served (file, RAM, VRAM L2, VRAM L1)
    sizing      the device/host admission, the L2 and RAM expert tiers, the ds4x open
    reuse       prefill expert reuse (new vs active) and CARRY_RAM_LOCK_PARTIAL

Byte lines come from LLAMADECODE_STEP_BYTES, which the producer emits with
LLAMADECODE_BYTE_LEDGER=1. Logs without it fall back to LLAMADECODE_DS4X_TOKEN, whose
ds_bytes are file bytes only; the tiers are then inferred from the hit counts, and the
report says so rather than presenting the inference as measurement.

    scripts/analyze_dsv41_decode.py runtime/single-engine/ultra-*/stderr.log
"""

import argparse
import re
import statistics
import sys

KV = re.compile(r"(\w+)=(\"[^\"]*\"|[^\s]+)")
STEP = re.compile(r"LLAMADECODE_STEP\b")
LEDGER = "LLAMADECODE_STEP_BYTES"
TOKEN = "LLAMADECODE_DS4X_TOKEN"
PREFILL_EXPERTS = "[dsv41-prefill-experts]"

# The lines the plan wants captured in the same run, and the ones that show a refusal.
WATCH = ("DS4PERF_GPU_JOB_ADMISSION", "LLAMADECODE_HOST_PLAN", "LLAMADECODE_DS4X_OPEN",
         "LLAMADECODE_DS4X_OPENED", "LLAMADECODE_DS4X_READY", "LLAMADECODE_L2 ",
         "LLAMADECODE_HOST_CACHE slots=", "CARRY_RAM_LOCK_PARTIAL", "LLAMADECODE_LIVE_BIAS",
         "DS4X_AFFINITY", "affinity calibration written", "incomplete affinity calibration",
         "wrong-model affinity calibration", "cannot write affinity calibration",
         # The extent relay's host slots are cudaHostAlloc'd and the fallback is silent; this says
         # which transport a run actually got, which is what its fetch rate turns on. The plan line
         # says the geometry the run asked for, so a log can be read as 4 or 8 slots without
         # guessing from the environment.
         "DS4X_EXTENT_WINDOW", "LLAMADECODE_EXTENT_WINDOW_PLAN",
         # The decode VRAM guard is what ended three runs on 2026-11-09; the headroom line says
         # how far above the reserve the steady state actually sat.
         "LLAMADECODE_VRAM_HEADROOM", "VRAM_RESERVE_BREACH",
         "PRODUCER_ABORT", "LLAMADECODE_DECODE_COMPLETE", "budget_breaches")


def kv(line):
    return {k: v.strip('"') for k, v in KV.findall(line)}


def numbers(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def pct(values, q):
    if not values:
        return float("nan")
    return sorted(values)[min(len(values) - 1, int(len(values) * q))]


def gib(value):
    return value / 2 ** 30


def median(values):
    return statistics.median(values) if values else float("nan")


def report(path, min_steps):
    text = open(path, errors="replace").read()
    lines = text.splitlines()
    steps, ledger, tokens = [], [], []
    for line in lines:
        if STEP.search(line):
            step = numbers(kv(line).get("step_ms"))
            if step is not None:
                steps.append(step)
        elif line.startswith(LEDGER):
            ledger.append(kv(line))
        elif line.startswith(TOKEN):
            tokens.append(kv(line))

    print("== %s" % path)
    if not steps:
        print("  decode steps: none - the run did not reach a token")
    else:
        mean = sum(steps) / len(steps)
        print("  decode steps=%d median_ms=%.1f mean_ms=%.1f p10_ms=%.1f p90_ms=%.1f"
              % (len(steps), median(steps), mean, pct(steps, .10), pct(steps, .90)))
        print("  rate from median %.2f tok/s, from mean %.2f tok/s" % (1000.0 / median(steps), 1000.0 / mean))
        if len(steps) < min_steps:
            print("  NOTE: only %d steps; the plan's bar is a run of at least %d tokens" % (len(steps), min_steps))

    # Bytes per step. Measured where the ledger ran, inferred where it did not.
    if ledger:
        need = sum(int(row.get("footprint", 0)) for row in ledger)
        served = sum(int(row.get("served", 0)) for row in ledger)
        by = {tier: sum(int(row.get(tier, 0)) for row in ledger)
              for tier in ("from_file", "from_ram", "from_l2", "from_l1")}
        copied = sum(int(row.get("copied", 0)) for row in ledger)
        stored = sum(int(row.get("stored", 0)) for row in ledger)
        unattributed = sum(1 for row in ledger if int(row.get("unattributed", 0)))
        over = sum(1 for row in ledger if int(row.get("over_served", 0)))
        print("  bytes measured from %s: footprint %.2f GiB/step, served %.2f GiB/step"
              % (LEDGER, gib(need / len(ledger)), gib(served / len(ledger))))
        for tier in ("from_file", "from_ram", "from_l2", "from_l1"):
            per = by[tier] / len(ledger)
            print("    %-10s %7.3f GiB/step  %5.1f %% of footprint" % (tier, gib(per), 100.0 * per / (need / len(ledger)) if need else 0.0))
        print("    copied     %7.3f GiB/step (device-to-device), stored %7.3f GiB/step (RAM tier writes)"
              % (gib(copied / len(ledger)), gib(stored / len(ledger))))
        hits = {k: sum(int(row.get(k, 0)) for row in ledger) for k in ("l1_hits", "l2_hits", "ram_hits", "misses")}
        total = sum(hits.values())
        if total:
            print("    expert tiers: l1 %.1f %%  l2 %.1f %%  ram %.1f %%  miss %.1f %%  (hit rate %.1f %%)"
                  % tuple([100.0 * hits[k] / total for k in ("l1_hits", "l2_hits", "ram_hits", "misses")]
                          + [100.0 * (total - hits["misses"]) / total]))
        print("    steps with bytes no tier claims: %d, steps serving more than their footprint: %d" % (unattributed, over))
    elif tokens:
        # ds_bytes are file bytes only; a triple's bytes are what one miss cost.
        triple = [int(row["ds_bytes"]) / int(row["misses"]) for row in tokens
                  if int(row.get("ds_bytes", 0)) and int(row.get("misses", 0))]
        if not triple:
            print("  bytes: no misses in any step, so no file-byte ratio to infer")
        else:
            per_triple = median(triple)
            experts = sum(int(row.get(k, 0)) for row in tokens for k in ("l1_hits", "l2_hits", "misses")) / len(tokens)
            disk = sum(int(row.get("ds_bytes", 0)) for row in tokens) / len(tokens)
            hits = sum(int(row.get(k, 0)) for row in tokens for k in ("l1_hits", "l2_hits")) / len(tokens)
            print("  bytes INFERRED from %s (no %s line; run with LLAMADECODE_BYTE_LEDGER=1 to measure):"
                  % (TOKEN, LEDGER))
            print("    expert triple ~%.2f MiB, %.1f triples/step -> footprint ~%.2f GiB/step"
                  % (per_triple / 2 ** 20, experts, gib(experts * per_triple)))
            print("    from file %.2f GiB/step (%.1f %%), from VRAM L1+L2 %.2f GiB/step (%.1f %%)"
                  % (gib(disk), 100.0 * disk / (experts * per_triple), gib(hits * per_triple),
                     100.0 * hits * per_triple / (experts * per_triple)))
            print("    RAM tier served bytes are not distinguishable from L2 hits without the ledger")
            fetch_ms = sum(numbers(row.get("ds_ms", 0)) or 0.0 for row in tokens)
            if fetch_ms:
                print("    implied file read rate %.2f GiB/s (ds_bytes / ds_ms over %d steps)"
                      % (gib(sum(int(row.get("ds_bytes", 0)) for row in tokens) * 1000.0 / fetch_ms), len(tokens)))

    print("  run setup and refusals:")
    last_token = tokens[-1] if tokens else None
    sized = [kv(line) for line in lines if "LLAMADECODE_HOST_CACHE slots=" in line]
    if last_token:
        print("    expert tiers in the cache: l1_slots=%s l2_slots=%s host_slots=%s host_bytes=%s"
              % (last_token.get("l1_slots"), last_token.get("l2_slots"),
                 last_token.get("host_slots"), last_token.get("host_bytes")))
        if sized and sized[-1].get("slots") not in (None, "0"):
            # The step lines report what the last step hit; the HOST_CACHE line is the only
            # one that says the tier was sized at all. Conflating them read a run that died at
            # its first token as "the RAM tier was never admitted".
            print("    RAM tier sized at open: slots=%s bytes=%s offer=%s; the last step served "
                  "host_slots=%s from it"
                  % (sized[-1].get("slots"), sized[-1].get("bytes"), sized[-1].get("offer"),
                     last_token.get("host_slots")))
        elif last_token.get("host_slots") == "0":
            print("    no RAM tier: LLAMADECODE_HOST_CACHE_MB unset (unset is off), so every "
                  "victim went back to the file")
    for tag in WATCH:
        seen = []
        for line in lines:
            if tag in line and line not in seen:
                seen.append(line)
        for line in seen[-2:]:
            print("    %s" % line[:200])
        if len(seen) > 2:
            print("    ... %d more %s lines" % (len(seen) - 2, tag))

    prefill = [kv(line) for line in lines if line.startswith(PREFILL_EXPERTS)]
    if prefill:
        active = sum(int(row.get("active", 0)) for row in prefill)
        new = sum(int(row.get("new", 0)) for row in prefill)
        print("  prefill expert reuse: %d layers, %d active, %d newly read (%.1f %% reused)"
              % (len(prefill), active, new, 100.0 * (active - new) / active if active else 0.0))
    return 0 if steps else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--min-steps", type=int, default=200,
                        help="flag runs shorter than this (the plan measures runs of >=200 tokens)")
    args = parser.parse_args()
    code = 0
    for path in args.logs:
        code |= report(path, args.min_steps)
    return code


if __name__ == "__main__":
    sys.exit(main())
