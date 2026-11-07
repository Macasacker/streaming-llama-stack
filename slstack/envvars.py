"""Environment inventory, CLI surface and child environment share one checked table."""
import bisect
import json
import math
import os
from pathlib import Path
import re
import sys
from functools import lru_cache

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / 'envvars.json'
_WARNED = set()
READ = re.compile(r'''(?:\b\w*env\w*|environ\.get|environ\.setdefault)\s*\(\s*[Lru]*["']([A-Z][A-Z0-9_]+)["']|environ\s*\[\s*["']([A-Z][A-Z0-9_]+)["']''')
DIRECT_READ = re.compile(r'''(?:process|import\.meta)\.env(?:\.([A-Z][A-Z0-9_]+)|\[\s*["']([A-Z][A-Z0-9_]+)["']\s*\])|getEnvVar(?:Owned|Sentinel)\s*\([^,]+,\s*["']([A-Z][A-Z0-9_]+)["']''')
WINDOWS_READ = re.compile(r'''\b(?:GetEnvironmentVariable[AW]|getenv_s|_dupenv_s)\s*\([^;\n]*?["']([A-Z][A-Z0-9_]+)["']''')
SWIFT_READ = re.compile(r'''\bProcessInfo\.processInfo\.environment\s*\[\s*["']([A-Z][A-Z0-9_]+)["']''')
# A direct env call whose argument is an expression rather than a plain literal, e.g.
# `getenv(tp == 2 ? "DS4_METAL_DISABLE_V41_TP_DECODE_QUEUE" : "DS4_METAL_DISABLE_V41_DECODE_QUEUE")`.
# Every name literal inside that argument is read; which one depends on a runtime value.
DIRECT_CALL = re.compile(r'''\b(?:std::)?(?:getenv|setenv|getenv_s|_dupenv_s|GetEnvironmentVariable[AW])\s*\(''')
NAME_LITERAL = re.compile(r'''["']([A-Z][A-Z0-9_]+)["']''')
# The name-carrying parameter of a wrapper: an identifier that stands alone as an
# argument of a direct env call (a `g->field` or a ternary is not a parameter).
PARAM_ARG = re.compile(r'''\b(?:std::)?(?:getenv|setenv|getenv_s|_dupenv_s|GetEnvironmentVariable[AW])\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*[,)]''')
FUNC_HEAD = re.compile(r'''([A-Za-z_][A-Za-z0-9_]*)\s*\(''')
IDENT = re.compile(r'''[A-Za-z_][A-Za-z0-9_]*''')
SOURCE_SUFFIXES = {'.c', '.h', '.cpp', '.hpp', '.cu', '.cuh', '.inc', '.m', '.mm', '.py', '.zig', '.js', '.ts', '.svelte', '.swift'}
SKIP_DIRECTORIES = {'.git', 'node_modules', 'build', '.venv', '__pycache__', '.svelte-kit', 'dist', '_app'}
HOST_VARIABLES = {'PATH', 'HOME', 'USERPROFILE', 'LOCALAPPDATA', 'APPDATA', 'SYSTEMROOT',
                  'WINDIR', 'LANG', 'LC_ALL', 'LC_CTYPE', 'TERM', 'COMSPEC', 'PATHEXT',
                  'USERNAME', 'LOGNAME', 'USER', 'SHELL', 'PWD', 'OLDPWD', 'CUDA_PATH',
                  'NUMBER_OF_PROCESSORS', 'PROCESSOR_ARCHITECTURE', 'NO_COLOR', 'FORCE_COLOR'}
SCOPE_ALIASES = {'rtx-producer': 'rtx-llama'}


@lru_cache(maxsize=16)
def _header_fingerprint(filename, device, inode, size, modified):
    value = 14695981039346656037
    with Path(filename).open('rb') as stream:
        for byte in stream.read(65536):
            value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return format(value, '016x')


def matches_model_signature(cfg, signature):
    model = Path(cfg.get('model', ''))
    if not model.is_file():
        return False
    info = model.stat()
    if info.st_size != signature['file_size']:
        return False
    return _header_fingerprint(str(model.resolve()), info.st_dev, info.st_ino,
                               info.st_size, info.st_mtime_ns) == signature['header_fnv64']


def _strip_comments(text):
    """Comments replaced by spaces so an offset stays an offset. String literals and
    line breaks are kept, because a wrapper call site is found by its literals."""
    out = []
    state, quote = 'code', '"'
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if state == 'code':
            if c == '/' and i + 1 < n and text[i + 1] == '/':
                state = 'line'
                out.append('  ')
                i += 2
                continue
            if c == '/' and i + 1 < n and text[i + 1] == '*':
                state = 'block'
                out.append('  ')
                i += 2
                continue
            if c == '"' or c == "'":
                state = 'string'
                quote = c
            out.append(c)
            i += 1
            continue
        if state == 'line':
            if c == '\n':
                state = 'code'
                out.append('\n')
            else:
                out.append(' ')
            i += 1
            continue
        if state == 'block':
            if c == '*' and i + 1 < n and text[i + 1] == '/':
                state = 'code'
                out.append('  ')
                i += 2
                continue
            out.append('\n' if c == '\n' else ' ')
            i += 1
            continue
        if c == '\\' and i + 1 < n:
            out.append(text[i:i + 2])
            i += 2
            continue
        if c == quote:
            state = 'code'
        out.append(c)
        i += 1
    return ''.join(out)


def _group_end(src, open_idx, opens, closes):
    """Offset of the closing `closes` matching the opening at `open_idx`, or -1."""
    depth = 0
    state, quote = 'code', '"'
    i = open_idx
    while i < len(src):
        c = src[i]
        if state == 'code':
            if c == '"' or c == "'":
                state = 'string'
                quote = c
            elif c == opens:
                depth += 1
            elif c == closes:
                depth -= 1
                if depth == 0:
                    return i
        else:
            if c == '\\' and i + 1 < len(src):
                i += 1
            elif c == quote:
                state = 'code'
        i += 1
    return -1


def _call_args(src, open_idx, close_idx):
    """The argument slices of a call whose parentheses span [open_idx, close_idx]."""
    _, parts = _split_top_level(src, open_idx + 1, close_idx, ',')
    return [p.strip() for p in parts]


def _split_top_level(src, start, stop, sep):
    depth = 0
    state, quote = 'code', '"'
    parts, cur, i = [], start, start
    while i < stop:
        c = src[i]
        if state == 'code':
            if c == '"' or c == "'":
                state = 'string'
                quote = c
            elif c in '([{':
                depth += 1
            elif c in ')]}':
                depth -= 1
            elif c == sep and depth == 0:
                parts.append(src[cur:i])
                cur = i + 1
        else:
            if c == '\\' and i + 1 < stop:
                i += 1
            elif c == quote:
                state = 'code'
        i += 1
    parts.append(src[cur:stop])
    return depth, parts


def _line_of(newlines, offset):
    """1-based line of `offset`, given the offsets of every newline in the text.
    Counting newlines per hit is O(offset), and ds4.c has thousands of hits."""
    return bisect.bisect_right(newlines, offset) + 1


def _newline_offsets(text):
    """Offsets of every newline in `text`, for `_line_of`."""
    offsets, at = [], 0
    while True:
        at = text.find('\n', at)
        if at < 0:
            return offsets
        offsets.append(at)
        at += 1


def _source_texts():
    """The scanned trees as {repo-relative path: (text, text without comments)}.

    The comment-stripped copy is what the call-shape passes scan, so it is built once
    per file here rather than once per pass.
    """
    texts = {}
    for root in (ROOT / 'engines', ROOT / 'vendor'):
        for directory, children, files in os.walk(root):
            children[:] = [s for s in children if s not in SKIP_DIRECTORIES]
            for name in sorted(files):
                p = Path(directory) / name
                if p.suffix not in SOURCE_SUFFIXES:
                    continue
                text = p.read_text(encoding='utf-8')
                texts[p.relative_to(ROOT).as_posix()] = (text, _strip_comments(text))
    return texts


def _function_bodies(src):
    """(name, parameter names, head paren offset, body offset, body slice) for
    definitions with a braced body."""
    for m in FUNC_HEAD.finditer(src):
        open_idx = src.index('(', m.end() - 1)
        close = _group_end(src, open_idx, '(', ')')
        if close < 0:
            continue
        after = close + 1
        while after < len(src) and src[after] in ' \t\n\r':
            after += 1
        if after >= len(src) or src[after] != '{':
            continue
        end = _group_end(src, after, '{', '}')
        if end < 0:
            continue
        params = []
        for arg in _call_args(src, open_idx, close):
            idents = IDENT.findall(arg.replace('(', ' '))
            params.append(idents[-1] if idents else None)
        yield m.group(1), params, open_idx, after, src[after:end + 1]


def _env_wrappers(texts):
    """{function: parameter indices that carry an environment name to a read}.

    A function qualifies when its body hands a parameter to a direct env call, or to
    a function already known to be a wrapper; the scan runs to a fixed point so a
    forwarding helper qualifies too. Only files that contain such a call are parsed,
    which is why this is not a whole-tree AST walk.
    """
    parsed = []
    for relative, (text, src) in texts.items():
        if not PARAM_ARG.search(src):
            continue
        for name, params, open_idx, body_at, body in _function_bodies(src):
            calls = {}
            for m in FUNC_HEAD.finditer(body):
                calls.setdefault(m.group(1), []).append(body_at + m.end() - 1)
            direct = [params.index(m.group(1)) for m in PARAM_ARG.finditer(body) if m.group(1) in params]
            parsed.append((relative, name, params, open_idx, src, body, sorted(set(direct)), calls))
    known = {name: tuple(indices) for _, name, _, _, _, _, indices, _ in parsed if indices}
    while True:
        added = False
        for _, name, params, _, src, body, _, calls in parsed:
            found = set(known.get(name, ()))
            for wrapper, indices in list(known.items()):
                if wrapper == name:
                    continue
                for open_idx in calls.get(wrapper, ()):  # offsets are into the file, not the body
                    close = _group_end(src, open_idx, '(', ')')
                    if close < 0:
                        continue
                    args = _call_args(src, open_idx, close)
                    for index in indices:
                        if index < len(args) and args[index].strip() in params:
                            target = params.index(args[index].strip())
                            if target not in found:
                                found.add(target)
                                added = True
            if found != set(known.get(name, ())):
                known[name] = tuple(sorted(found))
        if not added:
            break
    definitions = {}
    for relative, name, _, open_idx, _, _, _, _ in parsed:
        definitions.setdefault(relative, set()).add(open_idx)
    return known, definitions


def _wrapper_pattern(wrappers):
    names = '|'.join(sorted(wrappers))
    return re.compile(r'\b(' + names + r')\s*\(')


def _is_declaration(args):
    """True when every argument reads as `type name`, i.e. a parameter list rather
    than a call. A call statement also ends at a `;`, so this is what separates a
    prototype from a real call."""
    if not args or args == ['']:
        return False
    declarator = re.compile(r'^(?:const\s+|static\s+|unsigned\s+|struct\s+)*'
                            r'[A-Za-z_][A-Za-z0-9_]*(?:<[^<>]*>)?(?:\s*[*&])?\s*[A-Za-z_][A-Za-z0-9_]*(?:\s*\[[^\]]*\])?$')
    return all(declarator.match(arg) for arg in args)


def _wrapper_reads(text, src, wrappers, definitions, relative, pattern, newlines):
    """(reads, discarded, unresolved) for one file.

    `reads` are name literals at an argument the wrapper hands to a real env call.
    `discarded` are literals at an argument the wrapper ignores - `glm_graph_env_present(
    "DS4_ROCM_X", "DS4_METAL_X")` reads only the second one, so the first is an alias
    nothing reads and the table must not claim it. `unresolved` is what this cannot
    decide: a name passed through a variable or a table cell, listed by
    `unresolved_wrapper_reads()` for a human to read.

    A definition site is skipped: its argument is the parameter declaration, not a
    name. So is a prototype, which has no call at all.
    """
    found = []
    discarded = []
    unresolved = []
    own = definitions.get(relative, set())
    for m in pattern.finditer(src):
        name = m.group(1)
        open_idx = src.index('(', m.end() - 1)
        if open_idx in own:
            continue
        close = _group_end(src, open_idx, '(', ')')
        if close < 0:
            continue
        args = _call_args(src, open_idx, close)
        after = close + 1
        while after < len(src) and src[after] in ' \t\n\r':
            after += 1
        if after < len(src) and src[after] == ';' and _is_declaration(args):
            continue                                  # a prototype, not a call
        position = open_idx + 1
        for index, arg in enumerate(args):
            at = src.index(arg, position) if arg else position
            literal = re.fullmatch(r'[Lru]*["\']([A-Z][A-Z0-9_]+)["\']', arg)
            if not literal:
                if index in wrappers.get(name, ()):
                    unresolved.append((name, _line_of(newlines, at), arg[:70]))
            elif index in wrappers.get(name, ()):
                found.append((literal.group(1), _line_of(newlines, at)))
            else:
                discarded.append((literal.group(1), _line_of(newlines, at)))
            position = at + len(arg)
    return found, discarded, unresolved


def _wrapper_scan(texts, wrappers, definitions):
    """(reads, discarded, unresolved) for every scanned file, one pass over each text."""
    pattern = _wrapper_pattern(wrappers)
    reads, discarded, unresolved = {}, {}, []
    for relative, (text, src) in texts.items():
        hit, dropped, miss = _wrapper_reads(text, src, wrappers, definitions, relative,
                                            pattern, _newline_offsets(text))
        for name, line in hit:
            reads.setdefault(name, []).append(f'{relative}:{line}')
        for name, line in dropped:
            discarded.setdefault(name, []).append(f'{relative}:{line}')
        for name, line, arg in miss:
            unresolved.append((relative, line, name, arg))
    return reads, discarded, unresolved


def unresolved_wrapper_reads():
    """Wrapper call sites whose name argument is not a literal - a variable or a
    table cell. The inventory cannot resolve these; read them by hand."""
    texts = _source_texts()
    wrappers, definitions = _env_wrappers(texts)
    return _wrapper_scan(texts, wrappers, definitions)[2]


def discarded_alias_reads():
    """Name literals at an argument a wrapper discards. They are cited in the source
    and set by old recipes, but nothing reads them, so they are not inventory reads."""
    texts = _source_texts()
    wrappers, definitions = _env_wrappers(texts)
    return _wrapper_scan(texts, wrappers, definitions)[1]


def _literal_reads(text, src, newlines):
    """(name, line, confirmed) for one file.

    `confirmed` marks a literal standing in the argument of a direct `getenv` or
    `setenv` - the read itself. The others are recognised by the *shape* of the call
    (a function whose name contains `env`), which is a heuristic: it also matches an
    argument a wrapper discards, and those are removed by `inventory()`. Scanning over
    the whole file rather than one line at a time is what makes `foo(\n  "NAME")` a
    read at all.
    """
    found = []
    for regex in (READ, DIRECT_READ, SWIFT_READ, WINDOWS_READ):
        for m in regex.finditer(text):
            group = next(i for i, v in enumerate(m.groups(), 1) if v)
            found.append((m.group(group), _line_of(newlines, m.start(group)), False))
    for m in DIRECT_CALL.finditer(src):
        open_idx = src.index('(', m.end() - 1)
        close = _group_end(src, open_idx, '(', ')')
        if close < 0:
            continue
        for literal in NAME_LITERAL.finditer(src, open_idx + 1, close):
            found.append((literal.group(1), _line_of(newlines, literal.start()), True))
    return found


def inventory():
    """Every environment name the scanned sources read, with its `path:line` reads.

    Three passes, because one regex cannot see all of them:
      1. a name literal in the call position the table has always matched, over the
         whole file rather than one line at a time, so `foo(\n  "NAME")` counts;
      2. every name literal inside a direct `getenv`/`setenv` argument, which covers
         a conditional such as `getenv(a ? "NAME_A" : "NAME_B")`;
      3. name literals handed to a wrapper helper - `glm_graph_env_present(
         "DS4_ROCM_X", "DS4_METAL_X")` reads only its second parameter, and no
         line-bound regex can see either of them.
    A read through a variable, such as a name table cell, is not resolvable here;
    `unresolved_wrapper_reads()` lists those call sites. A literal at an argument the
    wrapper discards is the reverse - visible but never read - and is dropped; see
    `discarded_alias_reads()`.
    """
    found = {}
    texts = _source_texts()
    wrappers, definitions = _env_wrappers(texts)
    wrapped, dropped, _ = _wrapper_scan(texts, wrappers, definitions)
    confirmed = set()
    for relative, (text, src) in texts.items():      # walk order, so recorded reads keep their order
        newlines = _newline_offsets(text)
        for name, line_number, direct in _literal_reads(text, src, newlines):
            reference = f'{relative}:{line_number}'
            if direct:
                confirmed.add(reference)
            reads = found.setdefault(name, [])
            if reference not in reads:      # two names can share a line, so dedupe per name
                reads.append(reference)
    for name, references in wrapped.items():
        for reference in references:
            confirmed.add(reference)
            if reference not in found.setdefault(name, []):
                found[name].append(reference)
    for name, references in dropped.items():
        # An argument the wrapper discards is not a read, unless the same position is
        # confirmed as one by a direct env call or by the wrapper's own read argument.
        found[name] = [reference for reference in found.get(name, [])
                       if reference not in references or reference in confirmed]
    found = {name: refs for name, refs in found.items() if refs}
    return found


def rows():
    return json.loads(TABLE.read_text(encoding='utf-8'))


def add_flags(parser, *, suppress=False):
    import argparse
    for row in rows():
        if row['flag']:
            parser.add_argument(row['flag'], dest='env_' + row['var'], default=argparse.SUPPRESS if suppress else None,
                                help=row['doc'])


def apply(env, cfg, scope):
    """Explicit values win; inherited engine knobs never silently override a preset."""
    overrides = cfg.get('env_overrides', {})
    scope = SCOPE_ALIASES.get(scope, scope)
    context = {**cfg, 'host_budget_bytes': int(cfg.get('host_budget_gib', 8) * (1 << 30)),
               'host_budget_mb': int(cfg.get('host_budget_gib', 8) * 1024),
               'physical_floor_bytes': int(cfg.get('min_available_gib', .5) * (1 << 30)),
               'runtime_floor_bytes': int(cfg.get('runtime_min_free_gib', 0.5) * (1 << 30)),
               'host_budget_decimal_gb': cfg.get('host_budget_gib', 8) * (1 << 30) / 1e9,
               'physical_floor_decimal_gb': cfg.get('min_available_gib', 4) * (1 << 30) / 1e9,
               'job_commit_mb': int(cfg.get('job_commit_gib', 48) * 1024),
               'native_commit_mb': min(45056, int(cfg.get('job_commit_gib', 48) * 1024))}
    if cfg.get('runtime_root'):
        from .config import path
        context.update(runtime_root=str(Path(cfg['runtime_root']).resolve()),
                       studio_root=str(path(cfg.get('studio_root', 'vendor/studio'))),
                       streaming_root=str(path(cfg.get('streaming_root', 'vendor/engines'))),
                       weight_fetcher=str(path(cfg.get('weight_fetcher', 'build/llama/bin/libslstack-weight-fetcher.so'))),
                       affinity_spread_file=str(path(cfg.get('cache_affinity_spread_file') or 'presets/dsv41-affinity-spreads.txt')),
                       library_pythonpath=os.pathsep.join((str(ROOT), str(path(cfg.get('studio_root', 'vendor/studio'))),
                                                          str(path(cfg.get('streaming_root', 'vendor/engines'))))))
    for row in rows():
        var = row['var']
        if var.upper() in HOST_VARIABLES and not row.get('defaults_by_scope'):
            continue  # Host variables (PATH, HOME, locale) keep their OS semantics.
        inherited = env.pop(var, None)
        applicable = scope in row['scopes']
        value = overrides.get(var) if applicable else None
        if value is None and applicable:
            value = cfg.get('ds4_env', {}).get(var) if scope == 'ds4' else None
            if value is None:
                value = row.get('defaults_by_scope', {}).get(scope)
                if row.get('model_signature') and not matches_model_signature(cfg, row['model_signature']):
                    value = None
                if value is not None:
                    # Only table-owned placeholders expand. User values, including
                    # JSON and braces in file names, remain literal.
                    def substitute(match):
                        if match[1] not in context:
                            raise ValueError(f'Missing launcher context {match[1]} for {var}')
                        return str(context[match[1]])
                    value = re.sub(r'\{([A-Za-z_][A-Za-z0-9_]*)\}', substitute, str(value))
        if value is not None:
            value = str(value)
            if row.get('allowed_values') and value not in row['allowed_values']:
                raise ValueError(f'{var}: value violates required memory-residency policy')
            if 'minimum' in row or 'maximum' in row:
                try:
                    numeric = float(value)
                    valid = math.isfinite(numeric) and row.get('minimum', -math.inf) <= numeric <= row.get('maximum', math.inf)
                except ValueError:
                    valid = False
                if not valid:
                    raise ValueError(f'{var}: value outside its documented range')
            if row.get('equals_context') and value != str(context[row['equals_context']]):
                raise ValueError(f'{var}: must match the launcher {row["equals_context"]} setting')
            for rule, predicate in (('maximum_context', lambda a, b: a <= b),
                                    ('minimum_context', lambda a, b: a >= b)):
                if row.get(rule):
                    try:
                        convert = float if row.get('numeric_type') == 'float' else int
                        numeric, boundary = convert(value), convert(context[row[rule]])
                        valid = math.isfinite(numeric) and numeric > 0 and predicate(numeric, boundary)
                    except ValueError:
                        valid = False
                    if not valid:
                        raise ValueError(f'{var}: conflicts with the admitted host-memory budget/floor')
            env[var] = value
        if inherited is not None:
            identity = (var, inherited, value)
            if identity not in _WARNED:
                _WARNED.add(identity)
                displayed = '<redacted>' if is_secret(var) else value
                print(f'Environment override: inherited {var} ignored; selected {displayed if value is not None else "engine default"}', file=sys.stderr)
    return env


SECRET_WORDS = ('TOKEN', 'KEY', 'SECRET', 'PASSWORD')


def is_secret(name):
    return any(word in name.upper() for word in SECRET_WORDS)


def split_secrets(env):
    """(env with secret values redacted, {name: value} secrets) for anything written to disk."""
    secrets = {k: v for k, v in env.items() if is_secret(k)}
    return {k: ('<redacted>' if k in secrets else v) for k, v in env.items()}, secrets


def redact_cfg(cfg):
    """A config copy that is safe to write into a job spec or a worker request."""
    def clean(value):
        if isinstance(value, dict):
            # The model's tokenizer is a local asset path, not an access token.
            return {k: '<redacted>' if str(k) != 'tokenizer' and is_secret(str(k)) else clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value
    return clean(cfg)


def restore_secrets(spec):
    """Fill a job spec's redacted values from this process's own environment."""
    for name in spec.get('secret_env', []):
        if name not in os.environ:
            raise RuntimeError('Secret ' + name + ' was not passed to the guarded job')
        spec['env'][name] = os.environ[name]
    return spec


def argv_overrides(command, cfg, scope):
    """A table flag beats a conflicting engine argument in a user's config."""
    scope = SCOPE_ALIASES.get(scope, scope)
    selected = {**(cfg.get('ds4_env', {}) if scope == 'ds4' else {}), **cfg.get('env_overrides', {})}
    conflicts = {row['argv_flag'] for row in rows() if scope in row['scopes'] and row.get('argv_flag') and row['var'] in selected}
    result = []
    index = 0
    while index < len(command):
        item = command[index]
        if item in conflicts:
            if index + 1 >= len(command):
                raise ValueError('Missing configured engine argument for ' + item)
            index += 2
        elif item.split('=', 1)[0] in conflicts:
            index += 1
        else:
            result.append(item)
            index += 1
    return result


def validate_table():
    table = rows()
    present = {r['var'] for r in table}
    current = inventory()
    missing = set(current) - present
    indexed = {row['var']: row for row in table}
    stale = [var for var in current if var in indexed and set(indexed[var]['reads']) != set(current[var])]
    if missing or stale or len(present) != len(table) or any(not r['doc'] for r in table):
        raise ValueError(f'Environment table out of date: missing={sorted(missing)}, stale={sorted(stale)}')
    return len(table)
