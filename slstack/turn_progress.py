"""Turn progress: what a running turn is doing, one line per second, on the engine's stdout.

Strata's server prints the same shape (`[strata] thinking: 47 of budget …`), and it is what makes
a 400K-token prompt and a long tool call legible in the terminal instead of a silent wait. The
single engine writes these lines to its own log (runtime/single-engine/server-<run>.log);
`scripts/start-dsv41-single.sh` follows that log while attached, so the lines appear in the
launcher's window next to its own output.

    [ds4single] reading the prompt: 399,161 of 399,166 tokens, 1 s so far
    [ds4single] thinking: 47 of max 32768 tokens, 90.7 tok/s, 1 s
    [ds4single] writing a tool call: edit: 149 of max 32768 tokens, 93.5 tok/s, 2 s
    [ds4single] answering: 274 of max 32768 tokens, 102.6 tok/s, 3 s
    [ds4single] done: 1071 tokens in 10 s (115.1 tok/s) (stop, cancel=False), expert cache 97.0% hit (+3.0% of the routed experts over PCIe)

Elapsed is measured from the start of the turn (queue wait excluded), so the reading line says
what the prefill has cost so far. The token lines are the decode position against the request's
`max_tokens`; their rate is all generated tokens over the time since the first one, which is what
Strata's live line reports. The done line's rate is the engine's own decode clock (first
evaluated step to last), the same figure `decode_tok_s` records in the request metrics.

Strata's thinking line has a `budget N (max M)` form when a thinking budget applies. This engine
has none: `reasoning_effort` only changes the rendered prompt (slstack/ds4_render.py), so the cap
shown while thinking is the request's max_tokens.

`"turn_progress": false` in the configuration turns the lines off; they cost one write per second
and nothing else.
"""
from __future__ import annotations
import re
import time

from .ds4_tool_parse import find_any_tool_start

PREFIX = '[ds4single]'
EVERY_SECONDS = 1.0
# One token has no first-to-last span to divide by; Strata's answer_speed applies the same bound.
RATE_MIN_SPAN_S = 0.5
INVOKE_NAME = re.compile(r'invoke\s+name="([^"]+)"')
# An invoke tag before any `tool_calls` wrapper has closed: the model is writing a call.
INVOKE_TAG = re.compile(r'<\s*(?:｜?DSML｜\s*)?invoke\b')
DS4X_TOKEN = re.compile(rb'LLAMADECODE_DS4X_TOKEN')
_COUNTERS = {}


def _count(line, key):
    """An `key=N` field of a producer line, or 0 when the line does not carry it."""
    pattern = _COUNTERS.setdefault(key, re.compile((r'\b%s=(\d+)' % key).encode()))
    hit = pattern.search(line)
    return int(hit.group(1)) if hit else 0


class ExpertCache:
    """The routed-expert cache counters of one turn's decode, read from the producer's per-token lines.

    `LLAMADECODE_DS4X_TOKEN` says, for each decoded token, how the expert slots its routing asked
    for were served: `l1_hits` already resident in VRAM, `l2_hits` copied from the VRAM L2 tier,
    `host_hits` uploaded from the RAM tier and `misses` read from the model file. The first two
    cross no PCIe link, the last two do, so one pair of totals gives both the hit rate and the
    share that came over the bus. Only the lines inside a decode call are fed here, so the figures
    are the turn's decode, not its prefill (`LLAMADECODE_DS4X_PREFILL` says the latter). A decode
    that writes no such line (the llama.cpp sweep fallback, or a producer without ds4 decode)
    leaves the totals at zero and the done line says nothing about the cache.
    """

    def __init__(self):
        self.vram = 0
        self.pcie = 0

    def feed(self, data):
        """Consume producer diagnostic bytes. A trailing line without its newline is a partial
        read (the caller hands complete lines); it is left for the next feed rather than counted."""
        if not data:return
        lines = data.split(b'\n')
        if not data.endswith(b'\n'):lines.pop()
        for line in lines:
            if not DS4X_TOKEN.search(line):
                continue
            self.vram += _count(line, 'l1_hits') + _count(line, 'l2_hits')
            self.pcie += _count(line, 'host_hits') + _count(line, 'misses')

    @property
    def routed(self):
        return self.vram + self.pcie

    def counters(self):
        return dict(routed=self.routed, vram_hits=self.vram, over_pcie=self.pcie,
                    hit_percent=None if not self.routed else round(100.0 * self.vram / self.routed, 1))

    def summary(self):
        """`, expert cache 97.0% hit (+3.0% of the routed experts over PCIe)`; '' with no counters."""
        if not self.routed:
            return ''
        hit = 100.0 * self.vram / self.routed
        pcie = 100.0 * self.pcie / self.routed
        over = f' (+{pcie:.1f}% of the routed experts over PCIe)' if pcie else ''
        return f', expert cache {hit:.1f}% hit{over}'


def tool_phase(text):
    """The tool-call phase of the answer text, named once the invoke tag has been written."""
    start = find_any_tool_start(text)
    if start is None:
        hit = INVOKE_TAG.search(text)
        if hit is None:return None
        start = hit.start()
    name = INVOKE_NAME.search(text, start)
    return 'writing a tool call' + (': ' + name.group(1) if name else '')


def phase_of(text, think):
    """Thinking, answering, or writing a tool call, from the text written so far.

    Same three phases Strata reports, read here from the generated text rather than from parser
    events: the reasoning region is what precedes `</think>`, and a call ends it early when the
    model opens one without closing the thinking.
    """
    body = text;thinking = bool(think)
    if thinking:
        end = body.find('</think>')
        if end >= 0:body = body[end + 8:];thinking = False
    tool = tool_phase(body)
    if tool is not None:return tool
    return 'thinking' if thinking else 'answering'


class TurnProgress:
    """One request's progress lines, written at most once per `every` seconds."""

    def __init__(self, cfg, *, say=None, clock=time.monotonic, every=EVERY_SECONDS):
        value = cfg.get('turn_progress', True)
        if type(value) is not bool:
            raise ValueError('turn_progress must be true or false')
        self.enabled = value
        self.say = say if say is not None else (lambda text: print(text, flush=True))
        self.clock = clock
        self.every = float(every)
        self.started = clock()
        self.last = None
        self.first_token = None

    def _line(self, text):
        if not self.enabled:
            return
        now = self.clock()
        if self.last is not None and now - self.last < self.every:
            return
        self.last = now
        self.say(PREFIX + ' ' + text)

    def reading(self, done, total):
        """The prefill line, from the 1 Hz heartbeat's progress snapshot.

        `done` None says the producer gave no sweep position (a cold launch still opening), so the
        line names only what has to be read. Ignored once the first token arrived: the prompt is
        read by then, and the heartbeat keeps running through the decode.
        """
        if self.first_token is not None or not self.enabled:
            return
        el = self.clock() - self.started
        total = int(total or 0)
        if done is None:
            self._line(f'reading the prompt: {total:,} tokens, {el:.0f} s so far')
        else:
            self._line(f'reading the prompt: {int(done):,} of {total:,} tokens, {el:.0f} s so far')

    def token(self, count, phase_of_text, limit):
        """The decode line: how far the turn has got, against the request's cap.

        `phase_of_text` is a callable so the phase scan of the generated text runs once per
        printed line, not once per token.
        """
        if not self.enabled:
            return
        now = self.clock()
        if self.first_token is None:
            self.first_token = now
        if self.last is not None and now - self.last < self.every:
            return
        span = now - self.first_token
        if span < RATE_MIN_SPAN_S:
            return
        self._line(f'{phase_of_text()}: {count} of max {int(limit)} tokens, {count / span:.1f} tok/s, '
                   f'{now - self.started:.0f} s')

    def done(self, count, finish, cancelled, cache, decode_ms=0.0):
        """The turn's summary, always written when the request ends normally."""
        if not self.enabled:
            return
        el = self.clock() - self.started
        seconds = decode_ms / 1000.0
        rate = count / seconds if count >= 2 and seconds > 0 else None
        speed = f'{rate:.1f} tok/s' if rate is not None else 'no speed'
        self.last = None
        self._line(f'done: {count} tokens in {el:.0f} s ({speed}) ({finish}, cancel={bool(cancelled)})'
                   + cache.summary())

    def failed(self, reason):
        """The turn ended in an error rather than a done line."""
        self.last = None
        if self.enabled:
            self._line(f'failed: {reason} after {self.clock() - self.started:.0f} s')
