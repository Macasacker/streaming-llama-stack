"""Turn a ds4-first-run stamp into the numbers the turn-cost plan is measured on.

Two rules this encodes, because both cost a reading before they were written down:

* **Step 0 is not a decode step.** The decode-layout rebuild lands inside the first decoded
  step, which is what made `tok_s=10.56` read as a decode rate against a steady 20 tok/s.
  Every step figure here excludes it.
* **The decode step must be divided by the bytes it fetched.** With the affinity bias on,
  two runs of the same request place different experts on the miss path, so step medians
  are not comparable at the resolution a fetch-window change moves. The per-byte fit
  (`assign_ms` regressed on `ds_bytes`) is the attributable number; the step median is the
  consequence of that rate times whatever the run happened to miss.

  python3 scripts/ds4_run_metrics.py --stamp 20261110-184342 --stamp 20261110-185850
  python3 scripts/ds4_run_metrics.py --dir runtime/single-engine/ultra-e2aa376cfa7c9a3a --json

With more than one run it prints the first as the control and every later one as a
difference against it.
"""
import argparse
import json
import os
import re
import statistics
import sys
import time

# realpath so a caller reaching this through a symlink still reads the repo's runtime tree.
ROOT = os.path.dirname(os.path.dirname(os.path.realpath(os.path.abspath(__file__))))
SINGLE_ENGINE = os.path.join(ROOT, 'runtime', 'single-engine')
MIB = 1024 * 1024
GIB = 1024 ** 3

# The layout flip's split (docs/TURN_COST_PLAN.md Step 1 item 1). Summed as milliseconds, and the
# handoff's own layout_ms is deliberately not in this list: it overlaps switch_ms and would
# double-count the same seconds.
FLIP_KEYS = ('switch_ms', 'tiers_ms', 'destroy_ms', 'detach_ms', 'build_ms', 'quiesce_ms',
             'build_fetcher_ms', 'build_banks_ms', 'build_register_ms', 'build_other_ms')


def fields(line):
    return {k: float(v) if '.' in v else int(v)
            for k, v in re.findall(r'(\w+)=([0-9]+(?:\.[0-9]+)?)', line)}


def percentile(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def fit(xs, ys):
    """Least squares `y = slope * x + intercept`, or None for fewer than two distinct x."""
    if len(xs) < 2 or len(set(xs)) < 2:
        return None
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    if not den:
        return None
    slope = num / den
    return {'slope_ms_per_unit': slope, 'intercept_ms': my - slope * mx}


def stamp_epoch(stamp):
    """The probe stamps a run at its start, so the stamp is also the lower bound of its dirs."""
    return time.mktime(time.strptime(stamp, '%Y%m%d-%H%M%S'))


def dir_birth(path):
    """When the oldest file in a producer directory was created - the run that started it.

    ctime is creation on Linux and appending moves mtime, so this survives a producer that keeps
    writing after the run that made it.
    """
    times = []
    for name in os.listdir(path):
        try:
            times.append(os.path.getctime(os.path.join(path, name)))
        except OSError:
            continue
    return min(times) if times else 0.0


def producers_for_stamp(stamp, next_stamp=None):
    """The producer directories the probe printed for that stamp, in the order it ran them.

    A directory belongs to the stamp that created it. A producer that outlives its run appears in
    the next run's log as well - measured 2026-11-10 19:33, where the run after an aborted
    producer printed that producer's directory among its own - and reading it into both stamps
    would credit one stamp with the other run's environment.
    """
    log = os.path.join(ROOT, 'runtime', 'debug', 'run-%s.log' % stamp)
    if not os.path.isfile(log):
        raise SystemExit('no probe log for stamp %s (%s)' % (stamp, log))
    after, before = stamp_epoch(stamp), stamp_epoch(next_stamp) if next_stamp else None
    seen, dirs = set(), []
    for line in open(log, errors='ignore'):
        for match in re.finditer(r'(ultra-[0-9a-f]{16})', line):
            name = match.group(1)
            if name in seen:
                continue
            path = os.path.join(SINGLE_ENGINE, name)
            if not os.path.isdir(path):
                continue
            seen.add(name)
            born = dir_birth(path)
            if born < after - 5 or (before is not None and born >= before):
                continue
            dirs.append(path)
    if not dirs:
        raise SystemExit('stamp %s names no producer directory created in its own window under %s'
                         % (stamp, SINGLE_ENGINE))
    return dirs


def read_lines(path):
    """Rotated first, then the live file: a run's lines are split across the two."""
    lines = []
    for name in ('stderr-rotated.log', 'stderr.log'):
        candidate = os.path.join(path, name)
        if os.path.isfile(candidate):
            lines += open(candidate, errors='ignore').read().splitlines()
    return lines


def metrics(path):
    lines = read_lines(path)
    out = {'dir': os.path.basename(path), 'window': '', 'window_plan': '', 'host_plan': '',
           'request': None, 'position': None, 'tok_s': None}

    routed = []
    for line in lines:
        if '[dsv41-prefill-experts]' in line:
            routed.append(fields(line))
    # A new prompt head restarts the stage counter, so bursts are separated by that reset
    # rather than by a marker line the probe does not print.
    bursts, current, previous = [], None, None
    for record in routed:
        if previous is None or record['stage'] < previous:
            current = {'bytes': 0, 'fetch_ms': 0.0, 'layers': 0}
            bursts.append(current)
        current['bytes'] += record['bytes']
        current['fetch_ms'] += record['fetch_ms']
        current['layers'] += 1
        previous = record['stage']
    total_bytes = sum(b['bytes'] for b in bursts)
    total_fetch = sum(b['fetch_ms'] for b in bursts)
    out['routed'] = {
        'bytes': total_bytes, 'fetch_s': total_fetch / 1000.0,
        'layers': sum(b['layers'] for b in bursts),
        'gib_s': (total_bytes / GIB) / (total_fetch / 1000.0) if total_fetch else None,
    }

    tokens = [fields(l) for l in lines if 'LLAMADECODE_DS4X_TOKEN' in l]
    steady = tokens[1:]
    step = [t['decode_ms'] for t in steady]
    assign = [t['assign_ms'] for t in steady]
    miss_gib = [t['ds_bytes'] / GIB for t in steady]
    per_byte = fit(miss_gib, assign)
    ds_bytes = sum(t['ds_bytes'] for t in tokens)
    ds_ms = sum(t['ds_ms'] for t in tokens)
    out['decode'] = {
        'steps': len(tokens), 'steady_steps': len(steady),
        'step0_ms': tokens[0]['decode_ms'] if tokens else None,
        'step_median_ms': statistics.median(step) if step else None,
        'step_p90_ms': percentile(step, 0.9) if step else None,
        'assign_median_ms': statistics.median(assign) if assign else None,
        'assign_p90_ms': percentile(assign, 0.9) if assign else None,
        'miss_gib_median': statistics.median(miss_gib) if miss_gib else None,
        'misses_per_step': (sum(t['misses'] for t in steady) / len(steady)) if steady else None,
        'relay_bytes': ds_bytes, 'relay_s': ds_ms / 1000.0,
        'relay_gib_s': (ds_bytes / GIB) / (ds_ms / 1000.0) if ds_ms else None,
        'assign_ms_per_gib': per_byte['slope_ms_per_unit'] if per_byte else None,
        'per_byte_gib_s': 1000.0 / per_byte['slope_ms_per_unit'] if per_byte and per_byte['slope_ms_per_unit'] > 0 else None,
    }

    # The decode-layout flip: one per producer. LLAMADECODE_LAYOUT_SWITCH carries the split
    # written by Step 1 item 1; the handoff line carries what the producer itself timed around
    # the first token.
    switch = [fields(line) for line in lines if line.startswith('LLAMADECODE_LAYOUT_SWITCH')]
    handoff = [fields(line) for line in lines if line.startswith('LLAMADECODE_DS4X_HANDOFF')]
    out['layout'] = dict(switch[-1]) if switch else {}
    if handoff:
        out['layout']['layout_ms'] = handoff[-1].get('layout_ms')
        out['layout']['handoff_total_ms'] = handoff[-1].get('total_ms')
    # The fetcher's own create/destroy split. A producer builds the fetcher twice - the cold
    # create and the flip's - so every line is kept and the last two are printed.
    out['wf_phases'] = [line.strip() for line in lines
                        if line.startswith('WF_CREATE_PHASE') or line.startswith('WF_DESTROY_PHASE')]

    # Step 3's acceptance number is host_hits over misses, not the from_ram share: a tier that
    # stores an expert nobody asks for moves bytes without ending a wait. It is kept out of the
    # decode dict so the request-by-request diff loop does not print it - a tier that is off in
    # the control has zeros there, and the interesting number is exactly that jump from zero.
    tier_hits = sum(t.get('host_hits') or 0 for t in tokens)
    tier_misses = sum(t.get('misses') or 0 for t in tokens)
    tier_lookups = tier_hits + tier_misses + sum(t.get('l1_hits') or 0 for t in tokens) \
        + sum(t.get('l2_hits') or 0 for t in tokens)
    out['tier'] = {
        'slots': tokens[-1].get('host_slots') or 0 if tokens else 0,
        'hits': tier_hits,
        'misses': tier_misses,
        'hit_over_miss': tier_hits / tier_misses if tier_misses else None,
        'stored': sum(t.get('host_stored') or 0 for t in tokens),
        'share': tier_hits / tier_lookups if tier_lookups else None,
        'cache': next((line.strip() for line in lines if line.startswith('LLAMADECODE_HOST_CACHE ')), ''),
        'working_set': next((line.strip() for line in lines
                             if line.startswith('LLAMADECODE_HOST_WORKING_SET')), ''),
    }

    for line in lines:
        if line.startswith('LLAMADECODE_LIVE_BIAS'):
            out['request'] = fields(line).get('position')   # the prompt length: the request's identity
        elif line.startswith('DS4X_EXTENT_WINDOW'):
            out['window'] = line.strip()
        elif line.startswith('LLAMADECODE_EXTENT_WINDOW_PLAN'):
            out['window_plan'] = line.strip()
        elif line.startswith('LLAMADECODE_HOST_PLAN'):
            out['host_plan'] = line.strip()
            # components=name:bytes, not key=value, so fields() cannot read it.
            match = re.search(r'provider:(\d+)', line)
            if match:
                out['provider_host_bytes'] = int(match.group(1))
        elif line.startswith('LLAMADECODE_DECODE_COMPLETE'):
            record = fields(line)
            out['position'] = record.get('position')
            out['tok_s'] = record.get('tok_s')
        elif line.startswith('LLAMADECODE_VRAM_HEADROOM'):
            record = fields(line)
            low = record.get('min_free_since_l2')
            if low is not None and (out.setdefault('vram_min_free', None) is None or low < out['vram_min_free']):
                out['vram_min_free'] = low
    out['refusals'] = [l.strip() for l in lines
                       if 'ADMISSION_REFUSED' in l or 'PAGING_STOP' in l or 'VRAM_RESERVE_BREACH' in l
                       or 'PRODUCER_ABORT' in l]
    reuse = [l.strip() for l in lines if 'prefill expert reuse' in l]
    out['reuse'] = reuse[-1] if reuse else ''
    return out


def cell(value, pattern='%.2f'):
    return pattern % value if isinstance(value, (int, float)) else '-'


def table(rows):
    header = ('producer', 'request', 'window', 'routed GiB/s', 'routed GiB', 'fetch s', 'layers',
              'step med/p90 ms', 'assign med/p90 ms', 'miss GiB/step', 'per-byte GiB/s')
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    for row in rows:
        routed, decode = row['routed'], row['decode']
        window = fields(row['window']).get('slots') if row['window'] else None
        lines.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
            row['dir'], row['request'] if row['request'] is not None else '-',
            window if window is not None else '-',
            cell(routed['gib_s']), cell(routed['bytes'] / GIB), cell(routed['fetch_s'], '%.1f'),
            routed['layers'],
            '%s / %s' % (cell(decode['step_median_ms'], '%.1f'), cell(decode['step_p90_ms'], '%.1f')),
            '%s / %s' % (cell(decode['assign_median_ms'], '%.1f'), cell(decode['assign_p90_ms'], '%.1f')),
            cell(decode['miss_gib_median'], '%.3f'), cell(decode['per_byte_gib_s'])))
    return '\n'.join(lines)


def number(value):
    return '%.3f' % value if abs(value) < 1 else '%.2f' % value


def grouped(paths, stamps):
    """Each producer log carries the run it came from, so matched pairs can be found later.

    `request` is the prompt length the producer logged for its live bias, which is the request's
    identity: the probe prints its producers in rotation order, not in the order the requests were
    sent, so pairing by order would compare a 588-token head against a 64 854-token one.
    """
    rows = []
    for path in paths:
        row = metrics(path)
        row['label'] = stamps[0] if stamps else os.path.basename(path)
        rows.append(row)
    return rows


def table(rows):
    header = ('run', 'request', 'producer', 'window', 'routed GiB/s', 'routed GiB', 'fetch s',
              'layers', 'step med/p90 ms', 'assign med/p90 ms', 'miss GiB/step', 'per-byte GiB/s',
              'ram hits/miss', 'layout ms')
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    for row in sorted(rows, key=lambda r: (r['label'], r['request'] or 0)):
        routed, decode = row['routed'], row['decode']
        slots = fields(row['window']).get('slots') if row['window'] else None
        lines.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
            row['label'], row['request'] if row['request'] is not None else '-', row['dir'],
            slots if slots is not None else '-',
            cell(routed['gib_s']), cell(routed['bytes'] / GIB), cell(routed['fetch_s'], '%.1f'),
            routed['layers'],
            '%s / %s' % (cell(decode['step_median_ms'], '%.1f'), cell(decode['step_p90_ms'], '%.1f')),
            '%s / %s' % (cell(decode['assign_median_ms'], '%.1f'), cell(decode['assign_p90_ms'], '%.1f')),
            cell(decode['miss_gib_median'], '%.3f'), cell(decode['per_byte_gib_s']),
            'off' if not row['tier']['slots'] else
            '%s of %d (%d slots)' % (cell(row['tier']['hit_over_miss'], '%.3f'),
                                     row['tier']['misses'], row['tier']['slots']),
            cell(row['layout'].get('layout_ms') if row['layout'].get('layout_ms') is not None
                 else row['layout'].get('switch_ms'), '%.0f')))
    return '\n'.join(lines)


def aggregate(rows):
    out = {}
    for row in rows:
        total = out.setdefault(row['label'], {'bytes': 0, 'fetch_s': 0.0, 'producers': 0, 'window': '',
                                              'tier_hits': 0, 'tier_misses': 0, 'tier_slots': 0,
                                              'flip': {name: 0 for name in FLIP_KEYS}})
        total['bytes'] += row['routed']['bytes']
        total['fetch_s'] += row['routed']['fetch_s']
        total['producers'] += 1
        total['window'] = total['window'] or row['window']
        total['tier_hits'] += row['tier']['hits']
        total['tier_misses'] += row['tier']['misses']
        total['tier_slots'] = max(total['tier_slots'], row['tier']['slots'])
        for name in FLIP_KEYS:
            total['flip'][name] += row['layout'].get(name) or 0
    for total in out.values():
        total['gib_s'] = (total['bytes'] / GIB) / total['fetch_s'] if total['fetch_s'] else None
        total['tier_hit_over_miss'] = total['tier_hits'] / total['tier_misses'] if total['tier_misses'] else None
    return out


def differences(rows):
    """Every non-first run against the first, request by request."""
    labels = sorted({row['label'] for row in rows}, key=lambda label: [r['label'] for r in rows].index(label))
    if len(labels) < 2:
        return []
    base = labels[0]
    out = []
    for row in rows:
        if row['label'] == base:
            continue
        control = next((c for c in rows if c['label'] == base and c['request'] == row['request']), None)
        if control is None:
            out.append('%s: the control run has no request at prompt %s, so this producer is '
                       'unmatched and contributes nothing to the comparison' % (row['label'], row['request']))
            continue
        for key in ('routed', 'decode'):
            for name, value in row[key].items():
                before = control[key].get(name)
                if isinstance(value, (int, float)) and isinstance(before, (int, float)) and before:
                    out.append('%s vs %s on request %s: %s %s -> %s (%+.1f%%)' % (
                        row['label'], base, row['request'], name, number(before), number(value),
                        100 * (value / before - 1)))
        for name in FLIP_KEYS[1:]:
            value, before = row['layout'].get(name), control['layout'].get(name)
            if isinstance(value, (int, float)) and isinstance(before, (int, float)):
                out.append('%s on request %s: layout %s %.0f -> %.0f ms (%+.0f ms)' % (
                    row['label'], row['request'], name, before, value, value - before))
        if row['window'] and control['window'] and row['window'] != control['window']:
            out.append('%s on request %s: extent window %s -> %s' % (
                row['label'], row['request'], fields(control['window']).get('slots'),
                fields(row['window']).get('slots')))
        if row.get('provider_host_bytes') and control.get('provider_host_bytes') and \
                row['provider_host_bytes'] != control['provider_host_bytes']:
            out.append('%s on request %s: host plan provider component %d -> %d bytes (%+d), '
                       'sum_matches %s -> %s' % (
                           row['label'], row['request'], control['provider_host_bytes'],
                           row['provider_host_bytes'],
                           row['provider_host_bytes'] - control['provider_host_bytes'],
                           fields(control['host_plan']).get('sum_matches'),
                           fields(row['host_plan']).get('sum_matches')))
        # The tier is either funded or not, so it is reported as a state change rather than as a
        # percentage of a zero control.
        if (row['tier']['slots'], row['tier']['hits']) != (control['tier']['slots'], control['tier']['hits']):
            out.append('%s on request %s: RAM tier %s slots -> %s slots, host_hits %d -> %d of %d '
                       'misses (%s -> %s), stored %d -> %d entries'
                       % (row['label'], row['request'], control['tier']['slots'], row['tier']['slots'],
                          control['tier']['hits'], row['tier']['hits'], row['tier']['misses'],
                          cell(control['tier']['hit_over_miss'], '%.3f'),
                          cell(row['tier']['hit_over_miss'], '%.3f'),
                          control['tier']['stored'], row['tier']['stored']))
    totals = aggregate(rows)
    first, last = totals[base], totals[labels[-1]]
    if len(labels) == 2 and first['fetch_s'] and last['fetch_s'] and first['gib_s']:
        out.append('AGGREGATE %s -> %s: %.2f -> %.2f GiB/s (x%.3f), %.1f s -> %.1f s of prompt-head '
                   'fetch (%+.1f s) over %.2f GiB read in both' % (
                       base, labels[-1], first['gib_s'], last['gib_s'], last['gib_s'] / first['gib_s'],
                       first['fetch_s'], last['fetch_s'], last['fetch_s'] - first['fetch_s'],
                       last['bytes'] / GIB))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stamp', action='append', default=[], help='ds4-first-run stamp, repeatable')
    parser.add_argument('--dir', action='append', default=[], help='producer directory, repeatable')
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--output', default='', help='write the report here as well')
    args = parser.parse_args(argv)
    if not args.stamp and not args.dir:
        parser.error('give at least one --stamp or --dir')

    def resolve(path):
        if os.path.isabs(path):
            return path
        joined = os.path.join(ROOT, path)
        if os.path.isdir(joined):
            return joined
        return os.path.join(SINGLE_ENGINE, path if path.startswith('ultra-') else 'ultra-%s' % path)

    rows, headings = [], []
    for path in args.dir:
        rows.append(grouped([resolve(path)], [])[0])
    for index, stamp in enumerate(args.stamp):
        later = args.stamp[index + 1] if index + 1 < len(args.stamp) else None
        paths = producers_for_stamp(stamp, later)
        headings.append('%s: %s' % (stamp, ' '.join(os.path.basename(p) for p in paths)))
        rows += grouped(paths, [stamp])

    parts = [head for head in headings if head]
    parts.append(table(rows))
    totals = aggregate(rows)
    parts.append('Routed column summed over each run\'s producers:\n' + '\n'.join(
        '  %s: %d producers, %.2f GiB in %.1f s = %.2f GiB/s%s' % (
            label, total['producers'], total['bytes'] / GIB, total['fetch_s'], total['gib_s'] or 0,
            ' (%s)' % fields(total['window']).get('slots') if total['window'] else '')
        for label, total in totals.items()))
    parts.append('RAM tier summed over each run\'s producers (Step 3 acceptance is hits >= 0.5 of misses):\n'
                 + '\n'.join('  %s: %s slots, %d hits of %d misses (%s), %d decode steps'
                             % (label, total['tier_slots'] or 'off', total['tier_hits'],
                                total['tier_misses'], cell(total['tier_hit_over_miss'], '%.3f'),
                                sum(r['decode']['steady_steps'] for r in rows if r['label'] == label))
                             for label, total in totals.items()))
    parts.append('Layout flip summed over each run\'s producers, ms (Step 1: it is allocation, not fetch):\n'
                 + '\n'.join(
                       '  %s: switch %.0f = tiers %.0f + destroy %.0f + detach %.0f + build %.0f + quiesce %.0f;'
                       ' build = fetcher %.0f + banks %.0f + register %.0f + other %.0f'
                       % (label, total['flip']['switch_ms'], total['flip']['tiers_ms'],
                          total['flip']['destroy_ms'], total['flip']['detach_ms'],
                          total['flip']['build_ms'], total['flip']['quiesce_ms'],
                          total['flip']['build_fetcher_ms'], total['flip']['build_banks_ms'],
                          total['flip']['build_register_ms'], total['flip']['build_other_ms'])
                       for label, total in totals.items()))
    diff = differences(rows)
    if diff:
        parts.append('Differences:\n' + '\n'.join('  ' + line for line in diff))
    for row in rows:
        if row['refusals']:
            parts.append('%s %s: refusals %s' % (row['label'], row['dir'], '; '.join(row['refusals'][:3])))
        if row['tier']['cache']:
            parts.append('%s %s %s' % (row['label'], row['dir'], row['tier']['cache']))
        if row['tier']['working_set']:
            parts.append('%s %s %s' % (row['label'], row['dir'], row['tier']['working_set']))
        for phase in row['wf_phases'][-2:]:
            parts.append('%s %s %s' % (row['label'], row['dir'], phase))
        if row['reuse']:
            parts.append('%s %s %s' % (row['label'], row['dir'], row['reuse']))
    text = json.dumps(rows, indent=1) + '\n' if args.json else '\n\n'.join(parts) + '\n'
    sys.stdout.write(text)
    if args.output:
        with open(args.output, 'w') as handle:
            handle.write(text)
    return 0


if __name__ == '__main__':
    sys.exit(main())
