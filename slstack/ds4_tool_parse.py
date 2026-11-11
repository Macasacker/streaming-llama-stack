"""DeepSeek generated-message parser: a Python port of the engine's exact logic.

The parser is C, so this is a line-for-line port, not the process — except in tag matching, where
this port accepts whitespace inside a DSML tag and the bare `</｜DSML｜>` close. This model writes
both (docs/SESSION_FAILURE_PLAN.md), and the single-engine path reads the generated text here
rather than in C, so the tolerance lives on this side and the C port stays as it was.
Provenance: the Windows ds4 engine's ds4_server.c at stack revision b5907f2 (fork base
058ae09159a1, merge e1e45374ddc3): find_any_tool_start/end, trim_tool_separator_ws,
find_tool_structural_text, dsml_unescape_text, dsml_attr, tool_call_json_args_add,
dsml_parse_leaf_param_json, dsml_parse_nested_params_object, split_reasoning_content,
ds4_local_unterminated_reasoning, ds4_unterminated_reasoning_before_tool,
parse_deepseek_generated_message_ex, try_repair_dsml,
parse_generated_message_for_response_for_syntax (DeepSeek-V4.1 syntax),
append_json_object_or_empty / append_tool_call_deltas_json, random_tool_id;
ds4_tool_text.h ds4_tool_text_unescape.
"""
from __future__ import annotations
import secrets
import json
from .ds4_render import (DS41_TOOL_CALLS_START, DS41_TOOL_CALLS_END, DS41_INVOKE_START, DS41_INVOKE_END,
                         DS41_PARAM_START, DS41_PARAM_END, DSML_BARE_CLOSE, json_escape, json_minify_raw_value, json_args_parse, skip_value)

DSML = '｜DSML｜';DSML_SHORT = 'DSML｜'
TOOL_CALLS_START = '<' + DSML + 'tool_calls>';TOOL_CALLS_END = '</' + DSML + 'tool_calls>'
INVOKE_START = '<' + DSML + 'invoke';INVOKE_END = '</' + DSML + 'invoke>'
PARAM_START = '<' + DSML + 'parameter';PARAM_END = '</' + DSML + 'parameter>'
TOOL_CALLS_START_SHORT = '<' + DSML_SHORT + 'tool_calls>';TOOL_CALLS_END_SHORT = '</' + DSML_SHORT + 'tool_calls>'
INVOKE_START_SHORT = '<' + DSML_SHORT + 'invoke';INVOKE_END_SHORT = '</' + DSML_SHORT + 'invoke>'
PARAM_START_SHORT = '<' + DSML_SHORT + 'parameter';PARAM_END_SHORT = '</' + DSML_SHORT + 'parameter>'
SPACE = ' \t\n\r\v\f'
TOOL_MARKUP = ('<｜DSML｜', '</｜DSML｜', '<DSML｜', '</DSML｜', '<tool_calls', '<tool_call>', '<invoke ')
STYLES = {0: (TOOL_CALLS_START, TOOL_CALLS_END, INVOKE_START, INVOKE_END, PARAM_START, PARAM_END),
          1: ('<tool_calls>', '</tool_calls>', '<invoke', '</invoke>', '<parameter', '</parameter>'),
          2: (TOOL_CALLS_START_SHORT, TOOL_CALLS_END_SHORT, INVOKE_START_SHORT, INVOKE_END_SHORT, PARAM_START_SHORT, PARAM_END_SHORT),
          3: (DS41_TOOL_CALLS_START, DS41_TOOL_CALLS_END, DS41_INVOKE_START, DS41_INVOKE_END, DS41_PARAM_START, DS41_PARAM_END)}
# A block whose elements close with the bare marker: `</｜DSML｜>` stands for whichever element is
# open. Style 1 has no marker to write bare, so it keeps only its own close tags.
BARE_CLOSE = {0: '</' + DSML + '>', 2: '</' + DSML_SHORT + '>', 3: DSML_BARE_CLOSE}


def _find(s, needle, start=0):
    i = s.find(needle, start)
    return None if i < 0 else i


_SPLIT = {}


def tag_split(tag):
    """(marker prefix, element text) of a DSML tag; (None, tag) when the tag carries no marker."""
    out = _SPLIT.get(tag)
    if out is not None:return out
    for marker in (DSML, DSML_SHORT):
        i = tag.find(marker)
        if i >= 0:
            out = (tag[:i + len(marker)], tag[i + len(marker):].lstrip(SPACE))
            _SPLIT[tag] = out;return out
    _SPLIT[tag] = (None, tag);return _SPLIT[tag]


def match_tag(s, p, tag):
    """Length of `tag` at p, tolerating a whitespace run between a DSML marker and the element.

    The model writes `<｜DSML｜\n calls>` and `<｜DSML｜ invoke` in the same block: the marker is a
    vocabulary token of its own and the element follows it, so whitespace there is generation
    drift rather than a different format. A tag with no marker (`<tool_calls>`) is exact.
    """
    prefix, rest = tag_split(tag)
    if prefix is None:return len(tag) if s.startswith(tag, p) else None
    if not s.startswith(prefix, p):return None
    q = p + len(prefix)
    while q < len(s) and s[q] in SPACE:q += 1
    return q - p + len(rest) if s.startswith(rest, q) else None


def find_tag(s, tag, start=0):
    """First match of a tolerantly matched tag at or after start, as (index, length)."""
    prefix, rest = tag_split(tag)
    if prefix is None:
        i = s.find(tag, start)
        return None if i < 0 else (i, len(tag))
    i = start
    while True:
        i = s.find(prefix, i)
        if i < 0:return None
        length = match_tag(s, i, tag)
        if length is not None:return i, length
        i += len(prefix)


def close_at(s, p, tag, bare):
    """Length of the element's own close at p, or of the bare marker close."""
    length = match_tag(s, p, tag)
    if length is not None:return length
    return len(bare) if bare is not None and s.startswith(bare, p) else None


def find_close(s, tag, start, bare):
    """Earliest of the two close forms at or after start, as (index, length, the form)."""
    own = find_tag(s, tag, start)
    other = -1 if bare is None else s.find(bare, start)
    if own is not None and (other < 0 or own[0] < other):return own[0], own[1], tag
    if other >= 0:return other, len(bare), bare
    return None


def find_any_tool_start(s, start=0):
    hits = [r[0] for r in (find_tag(s, n, start) for n in (DS41_TOOL_CALLS_START, TOOL_CALLS_START, TOOL_CALLS_START_SHORT, '<tool_calls>', '<tool_call>')) if r]
    return min(hits) if hits else None


def find_tool_markup_start(s):
    hits=[s.find(marker) for marker in TOOL_MARKUP if marker in s]
    return min(hits) if hits else None


def find_any_tool_end(s, start=0):
    hits = [r[0] for r in (find_tag(s, n, start) for n in (DS41_TOOL_CALLS_END, TOOL_CALLS_END, TOOL_CALLS_END_SHORT, '</tool_calls>', '</tool_call>')) if r]
    return min(hits) if hits else None


def skip_ws(s, i):
    while i < len(s) and s[i] in SPACE:i += 1
    return i


WRAPPERS = ((DS41_PARAM_START, DS41_PARAM_END, DSML_BARE_CLOSE), (PARAM_START, PARAM_END, '</' + DSML + '>'),
            (PARAM_START_SHORT, PARAM_END_SHORT, '</' + DSML_SHORT + '>'),
            ('<parameter', '</parameter>', None), ('<arg_key>', '</arg_key>', None), ('<arg_value>', '</arg_value>', None))


def find_tool_structural_text(s, needle, last):
    found = None;i = 0
    while i < len(s):
        if match_tag(s, i, needle) is not None:
            found = i
            if not last:return found
        skipped = False
        for start, end, bare in WRAPPERS:
            if match_tag(s, i, start) is None:continue
            tag_end = _find(s, '>', i)
            close = find_close(s, end, tag_end + 1, bare) if tag_end is not None else None
            if close is None:return found
            i = close[0] + close[1];skipped = True
            break
        if not skipped:i += 1
    return found


def dsml_unescape_text(s):
    out = [];i = 0
    while i < len(s):
        if s[i] != '&':out.append(s[i]);i += 1;continue
        for entity, ch in (('&amp;', '&'), ('&lt;', '<'), ('&gt;', '>'), ('&quot;', '"'), ('&apos;', "'")):
            if s.startswith(entity, i):out.append(ch);i += len(entity);break
        else:out.append('&');i += 1
    return ''.join(out)


def dsml_attr(tag, name):
    pattern = name + '="';p = tag.find(pattern)
    if p < 0:return None
    p += len(pattern);q = tag.find('"', p)
    if q < 0:return None
    return dsml_unescape_text(tag[p:q])


def tool_text_unescape(s, end, bare=None):
    """ds4_tool_text_unescape, for either close form of the element.

    The writer escapes a `<` only where it would open a terminator, so the reader unescapes
    exactly those: the element's own close, and the bare marker close the model also writes.
    """
    tails = [end[1:]] + ([bare[1:]] if bare is not None else [])
    out = [];i = 0
    while i < len(s):
        if s[i] == '&':
            j = i + 1
            while s.startswith('amp;', j):j += 4
            if s.startswith('lt;', j) and any(s.startswith(tail, j + 3) for tail in tails):
                if s.startswith('&amp;', i):out.append('&');i += 5
                else:out.append('<');i += 4
                continue
        out.append(s[i]);i += 1
    return ''.join(out)


def _args_add(args, name, value, is_string):
    piece = (', ' if args else '') + json_escape(name or '') + ': '
    if is_string == 'true':piece += json_escape(value or '')
    else:
        text = value if value is not None else 'null'
        stripped = text.lstrip(SPACE);end = skip_value(stripped, 0)
        minified = json_minify_raw_value(stripped[:end]) if end is not None else text
        piece += minified if minified else 'null'
    args.append(piece)


def _leaf(s, p, param_start, param_end, members, bare=None):
    if match_tag(s, p, param_start) is None:return None
    tag_end = _find(s, '>', p)
    if tag_end is None:return None
    tag = s[p:tag_end + 1];name = dsml_attr(tag, 'name');is_string = dsml_attr(tag, 'string')
    if name is None:return None
    hit = find_close(s, param_end, tag_end + 1, bare)
    if hit is None:return None
    value = s[tag_end + 1:hit[0]];kind = is_string if is_string is not None else 'true'
    if kind == 'true':value = tool_text_unescape(value, hit[2], bare)
    _args_add(members, name, value, kind)
    return hit[0] + hit[1]


def _nested(s, p, param_start, param_end, bare=None):
    members = [];any_member = False
    while True:
        p = skip_ws(s, p)
        if match_tag(s, p, param_start) is None:break
        p = _leaf(s, p, param_start, param_end, members, bare)
        if p is None:return None, None
        any_member = True
    if not any_member:return None, None
    return '{' + ''.join(members) + '}', p


def split_reasoning_content(text):
    body = text[7:] if text.startswith('<think>') else text
    end = body.find('</think>')
    if end >= 0:return body[end + 8:], body[:end]
    return text, None


def _cut_c(text):
    """C buffers end at NUL."""
    return text.split('\0', 1)[0]


def parse_deepseek_generated(text, require_thinking_closed, note=None):
    """parse_deepseek_generated_message_ex -> (ok, content, reasoning, calls, raw_tool_text).

    Tags are matched tolerantly (`match_tag`, `close_at`): whitespace inside a DSML tag and the
    bare `</｜DSML｜>` close are both read, because that is what this model writes. Arguments are
    still never guessed — anything the walk cannot follow returns ok=False. When `note` is a
    dict, it is filled with the tag the walk stopped at, for the refusal message.
    """
    def stop(why):
        if note is not None:note.clear();note['why'] = why
        return False, None, None, [], None
    text = _cut_c(text or '');tool_search = 0;recovered_unclosed_tool = False
    if require_thinking_closed:
        think_end = find_tool_structural_text(text, '</think>', True)
        if think_end is None:
            candidate = find_any_tool_start(text)
            if candidate is None or find_any_tool_end(text, candidate) is None:
                body = text[7:] if text.startswith('<think>') else text
                return True, '', body, [], None
            tool_search = candidate;recovered_unclosed_tool = True
        else:tool_search = think_end + 8
    start = None;style = 0
    for tag, st in ((TOOL_CALLS_START, 0), (TOOL_CALLS_START_SHORT, 2), ('<tool_calls>', 1), (DS41_TOOL_CALLS_START, 3)):
        hit = find_tag(text, tag, tool_search)
        if hit is not None and (start is None or hit[0] < start):start = hit[0];style = st
    if start is None:
        content, reasoning = split_reasoning_content(text)
        return True, content, reasoning, [], None
    # The raw block keeps the blank line the writer puts before it: `tool_memory` replays it
    # verbatim into the next prompt, so the recorded text must match what was rendered.
    tag_at = start
    if start >= 2 and text[start - 2:start] == '\n\n':start -= 2
    content_len = start
    while content_len > 0 and text[content_len - 1] in SPACE:content_len -= 1
    calls_start, calls_end, invoke_start, invoke_end, param_start, param_end = STYLES[style]
    bare = BARE_CLOSE.get(style)
    length = match_tag(text, tag_at, calls_start)
    if length is None:return stop('the tool block opens with ' + calls_start)
    p = tag_at + length;calls = []
    while True:
        p = skip_ws(text, p)
        closing = close_at(text, p, calls_end, bare)
        if closing is not None:
            raw = text[start:p + closing]
            if recovered_unclosed_tool:
                body = text[:content_len]
                if content_len >= 7 and body.startswith('<think>'):body = body[7:]
                return True, '', body, calls, raw
            content, reasoning = split_reasoning_content(text[:content_len])
            return True, content, reasoning, calls, raw
        if match_tag(text, p, invoke_start) is None:return stop('an invoke with an attribute list, ' + invoke_start + ' name="…"')
        tag_end = _find(text, '>', p)
        if tag_end is None:return stop('the closing > of the invoke tag')
        name = dsml_attr(text[p:tag_end + 1], 'name')
        if name is None:return stop('invoke name="…"')
        p = tag_end + 1;args = []
        while True:
            p = skip_ws(text, p)
            closing = close_at(text, p, invoke_end, bare)
            if closing is not None:p += closing;break
            if match_tag(text, p, param_start) is None:return stop('a parameter tag, ' + param_start + ' name="…"')
            tag_end = _find(text, '>', p)
            if tag_end is None:return stop('the closing > of the parameter tag')
            tag = text[p:tag_end + 1];param_name = dsml_attr(tag, 'name');param_is_string = dsml_attr(tag, 'string')
            if param_name is None:return stop('parameter name="…"')
            value_start = tag_end + 1
            if param_is_string is None and match_tag(text, skip_ws(text, value_start), param_start) is not None:
                nested, nested_p = _nested(text, value_start, param_start, param_end, bare)
                if nested is None:return stop('a complete nested object of parameter tags')
                _args_add(args, param_name, nested or '{}', 'false')
                p = skip_ws(text, nested_p)
                closing = close_at(text, p, param_end, bare)
                if closing is not None:p += closing
                continue
            hit = find_close(text, param_end, value_start, bare)
            if hit is None:return stop('the end of the parameter value, ' + param_end)
            kind = param_is_string if param_is_string is not None else 'true'
            value = text[value_start:hit[0]]
            if kind == 'true':value = tool_text_unescape(value, hit[2], bare)
            _args_add(args, param_name, value, kind)
            p = hit[0] + hit[1]
        calls.append(dict(id=None, name=name, arguments='{' + ''.join(args) + '}'))


def try_repair_dsml(s):
    """Only a missing outer wrapper around complete invocations is repaired."""
    if not s:return None
    think_end = find_tool_structural_text(s, '</think>', True)
    scan = think_end + 8 if think_end is not None else 0
    first = find_any_tool_start(s, scan)
    if first is None:return None
    for style in (3, 0, 2, 1):
        if match_tag(s, first, STYLES[style][0]) is not None:break
    else:return None
    tags = STYLES[style];bare = BARE_CLOSE.get(style)
    tos, toe, ios, ioe, pos, poe, depth = 0, 0, 0, 0, 0, 0, 0
    p = scan
    while p < len(s):
        length = match_tag(s, p, tags[0])
        if length is not None:tos += 1;depth += 1;p += length;continue
        length = match_tag(s, p, tags[2])
        if length is not None:ios += 1;depth += 1;p += length;continue
        length = match_tag(s, p, tags[4])
        if length is not None:pos += 1;depth += 1;p += length;continue
        length = match_tag(s, p, tags[5])
        if length is not None:poe += 1;depth -= 1;p += length;continue
        length = match_tag(s, p, tags[3])
        if length is not None:ioe += 1;depth -= 1;p += length;continue
        length = match_tag(s, p, tags[1])
        if length is not None:toe += 1;depth -= 1;p += length;continue
        if bare is not None and s.startswith(bare, p):
            # The bare marker closes whichever element is open, so it counts at that level.
            if depth == 3:poe += 1;depth = 2
            elif depth == 2:ioe += 1;depth = 1
            elif depth == 1:toe += 1;depth = 0
            p += len(bare);continue
        p += 1
    if ios != ioe or pos != poe or tos == toe or toe > tos or depth != tos - toe:return None
    return s + tags[1] * (tos - toe)


def parse_for_response(text, *, has_tools, saw_tool_start, require_thinking_closed, finish, note=None):
    """parse_generated_message_for_response_for_syntax (DeepSeek-V4.1).
    Returns dict(ok, content, reasoning, calls, raw_tool_text, finish, recovered)."""
    ok, content, reasoning, calls, raw = parse_deepseek_generated(text, require_thinking_closed, note)
    if ok:return dict(ok=True, content=content, reasoning=reasoning, calls=calls, raw_tool_text=raw, finish=finish, recovered=False)
    finish = finish or 'stop';recovered = False
    if has_tools and saw_tool_start and finish != 'error':
        finish = 'length' if finish == 'length' else 'stop';recovered = True
    return dict(ok=False, content=_cut_c(text or ''), reasoning=None, calls=[], raw_tool_text=None, finish=finish, recovered=recovered)


def finish_message(text, *, has_tools, think, finish):
    """The finish-time path of the native chat worker: wrapper repair, then the
    response parse. saw_tool_start/end follow the stream observer's marker scan
    after the closing </think> (or the whole text without thinking)."""
    scan = text
    if think:
        end = text.rfind('</think>')
        scan = text[end + 8:] if end >= 0 else ''
    saw_start = has_tools and find_any_tool_start(scan) is not None
    saw_end = has_tools and saw_start and find_any_tool_end(scan, find_any_tool_start(scan)) is not None
    if has_tools and saw_start and not saw_end and finish not in ('error', 'length'):
        repaired = try_repair_dsml(text)
        if repaired is not None:
            ok, _, _, calls, _ = parse_deepseek_generated(repaired, False)
            if ok and calls:text = repaired
    note = {}
    result = parse_for_response(text, has_tools=has_tools, saw_tool_start=saw_start, require_thinking_closed=think,
                                finish=finish, note=note)
    if result['calls']:
        try:
            if not all(isinstance(json.loads(call['arguments']),dict) for call in result['calls']):
                raise ValueError('Tool arguments must be JSON objects')
        except (ValueError,TypeError):
            result['calls']=[]
    if result['calls']:
        for call in result['calls']:call['id'] = random_tool_id()
        result['finish'] = 'tool_calls'
    elif has_tools:
        content, reasoning = split_reasoning_content(text)
        start = find_tool_markup_start(content)
        if start is not None:
            # Invalid arguments are never guessed or executed. Keep diagnostics
            # separate from answer text, including malformed/repeated DSML tags.
            result.update(content=content[:start].rstrip(),reasoning=reasoning,
                          tool_error=dict(error='Malformed tool call was not executed',raw=content[start:],
                                          reason=note.get('why') or 'the block carries no tool_calls wrapper this parser reads'))
    return result


def random_tool_id():
    return 'call_' + secrets.token_hex(16)


def json_object_or_empty(arguments):
    """append_json_object_or_empty: a compact JSON object text."""
    args = json_args_parse(arguments)
    if args is None:return '{}'
    return '{' + ','.join(json_escape(k) + ':' + (json_escape(v) if s else v) for k, v, s in args) + '}'


def tool_call_deltas(calls):
    """append_tool_call_deltas_json as Python rows (arguments as the JSON-object string)."""
    return [dict(index=i, id=c['id'], type='function', function=dict(name=c['name'] or '', arguments=json_object_or_empty(c['arguments'])))
            for i, c in enumerate(calls)]
