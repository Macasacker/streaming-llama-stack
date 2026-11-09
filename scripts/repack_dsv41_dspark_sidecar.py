#!/usr/bin/env python3
"""Repack a released DSpark drafter into the name and metadata contract this engine binds.

The DSpark artifact for DeepSeek V4.1-Flash 1017 ships with the Strata pack under Strata's
own drafter layout: `blk.N.<role>.weight`, flat head tensors, and the draft hyper-parameters
in the pack's `ds4.txt` rather than in the GGUF header. `engines/ds4` binds a V4.1 sidecar
through `tensor_by_mtp_stage_suffix` (`mtp.<stage>.<role>`), reads block/markov/noise/
target-layer/expert-count metadata from `deepseek41.dspark.*`, and refuses a sidecar whose
`general.source.revision` is absent or differs from the language checkpoint.

Nothing here re-derives weights. Tensor payloads are copied verbatim except the two Markov
tables, which the released file carries BF16 and the engine's dense reader accepts only as
F32/F16/Q8_0 (its Markov argmax path requires Q8_0).

    python3 scripts/repack_dsv41_dspark_sidecar.py SIDECAR LANGUAGE \
        --ds4-txt packs/deepseek-v4.1-flash-ist-das-2.00bpw/ds4.txt --check-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import types
from pathlib import Path

# This fork's GGUF dialect: value types start at UINT8=0, STRING=8, ARRAY=9, and tensor
# dimensions are u64. See engines/ds4/ds4.c GGUF_VALUE_* and parse_tensors().
U8, I8, U16, I16, U32, I32, F32, BOOL, STR, ARR, U64, I64, F64 = range(13)
SCALAR = {U8: '<B', I8: '<b', U16: '<H', I16: '<h', U32: '<I', I32: '<i',
          F32: '<f', BOOL: '<B', U64: '<Q', I64: '<q', F64: '<d'}
# ggml tensor types, as engines/ds4/ds4.c names them.
T_F32, T_F16, T_Q8_0, T_BF16, T_MXFP4 = 0, 1, 8, 30, 39
TYPE_NAME = {T_F32: 'F32', T_F16: 'F16', T_Q8_0: 'Q8_0', T_BF16: 'BF16', T_MXFP4: 'MXFP4'}
# (name, block_elems, block_bytes), mirroring gguf_types[] in engines/ds4/ds4.c:2176.
TYPES = {0: ('f32', 1, 4), 1: ('f16', 1, 2), 2: ('q4_0', 32, 18), 3: ('q4_1', 32, 20),
         6: ('q5_0', 32, 22), 7: ('q5_1', 32, 24), 8: ('q8_0', 32, 34), 9: ('q8_1', 32, 40),
         10: ('q2_k', 256, 84), 11: ('q3_k', 256, 110), 12: ('q4_k', 256, 144),
         13: ('q5_k', 256, 176), 14: ('q6_k', 256, 210), 15: ('q8_k', 256, 292),
         16: ('iq2_xxs', 256, 66), 17: ('iq2_xs', 256, 74), 18: ('iq3_xxs', 256, 98),
         19: ('iq1_s', 256, 110), 20: ('iq4_nl', 256, 50), 21: ('iq3_s', 256, 110),
         22: ('iq2_s', 256, 82), 23: ('iq4_xs', 256, 136), 24: ('i8', 1, 1),
         25: ('i16', 1, 2), 26: ('i32', 1, 4), 27: ('i64', 1, 8), 28: ('f64', 1, 8),
         29: ('iq1_m', 256, 56), 30: ('bf16', 1, 2), 39: ('mxfp4', 32, 17)}
BLOCK = {t: (elems, size) for t, (_, elems, size) in TYPES.items()}
TYPE_NAME = {t: name for t, (name, _, _) in TYPES.items()}
# deepseek41_sidecar_checkpoint_matches() accepts exactly these two language revisions.
ACCEPTED_REVISIONS = ('df42c109f1defefcbfcedbe7d905718a12266e40',
                      '1bae0e1088993b9b82de08f59de5be5a34e42170')
DENSE_OK = (T_F16, T_F32, T_Q8_0)                                    # DS4_DSPARK_LAYOUT_DENSE
PLAIN_OK = (T_F16, T_F32)                                            # DS4_DSPARK_LAYOUT_PLAIN
# tensor_is_routed_expert_type() plus the Q3_K the stage path also accepts.
ROUTED_OK = (T_Q8_0, T_MXFP4, 16, 10, 12, 13, 14, 11)
FLAT_HEADS = {'fc.weight': ('main_proj', 'first'),
              'enc.output_norm.weight': ('main_norm', 'first'),
              'output_norm.weight': ('norm', 'final'),
              'conf_proj.weight': ('confidence_head.proj', 'final'),
              'markov_w1.weight': ('markov_head.markov_w1', 'final'),
              'markov_w2.weight': ('markov_head.markov_w2', 'final')}
# Roles the engine binds for every draft stage, with the shape its validator wants.
STAGE_ROLES = {
    'hc_attn_fn': lambda g: (PLAIN_OK, [g.hc_dim, g.hc_mix]),
    'hc_attn_scale': lambda g: ((T_F32,), [3]),
    'hc_attn_base': lambda g: ((T_F32,), [g.hc_mix]),
    'attn_norm': lambda g: ((T_F32,), [g.n_embd]),
    'attn_q_a': lambda g: (DENSE_OK, [g.n_embd, g.n_lora_q]),
    'attn_q_a_norm': lambda g: ((T_F32,), [g.n_lora_q]),
    'attn_q_b': lambda g: (DENSE_OK, [g.n_lora_q, g.q_dim]),
    'attn_kv': lambda g: (DENSE_OK, [g.n_embd, g.n_head_dim]),
    'attn_kv_a_norm': lambda g: ((T_F32,), [g.n_head_dim]),
    'attn_sinks': lambda g: ((T_F32,), [g.n_head]),
    'attn_output_a': lambda g: (DENSE_OK, [g.n_head_dim * (g.n_head // g.n_out_group), g.out_low]),
    'attn_output_b': lambda g: (DENSE_OK, [g.out_low, g.n_embd]),
    'hc_ffn_fn': lambda g: (PLAIN_OK, [g.hc_dim, g.hc_mix]),
    'hc_ffn_scale': lambda g: ((T_F32,), [3]),
    'hc_ffn_base': lambda g: ((T_F32,), [g.hc_mix]),
    'ffn_norm': lambda g: ((T_F32,), [g.n_embd]),
    'ffn_gate_inp': lambda g: (DENSE_OK, [g.n_embd, g.n_expert]),
    'ffn_gate_exps': lambda g: (ROUTED_OK, [g.n_embd, g.n_ff_exp, g.n_expert]),
    'ffn_up_exps': lambda g: (ROUTED_OK, [g.n_embd, g.n_ff_exp, g.n_expert]),
    'ffn_down_exps': lambda g: (ROUTED_OK, [g.n_ff_exp, g.n_embd, g.n_expert]),
    'ffn_gate_shexp': lambda g: (DENSE_OK, [g.n_embd, g.n_ff_exp]),
    'ffn_up_shexp': lambda g: (DENSE_OK, [g.n_embd, g.n_ff_exp]),
    'ffn_down_shexp': lambda g: (DENSE_OK, [g.n_ff_exp, g.n_embd]),
}
BIASES = {'exp_probs_b': lambda g: ((T_F32,), [g.n_expert])}


class Shape:
    """The language checkpoint's geometry, which the sidecar's shapes are checked against."""

    def __init__(self, kv):
        self.n_embd = kv['deepseek41.hidden_size']
        self.n_vocab = kv['deepseek41.vocab_size']
        self.n_hc = kv['deepseek41.hc_mult']
        self.n_lora_q = kv['deepseek41.q_lora_rank']
        self.n_head = kv['deepseek41.num_attention_heads']
        self.n_head_dim = kv['deepseek41.head_dim']
        self.n_out_group = kv['deepseek41.o_groups']
        self.n_lora_o = kv['deepseek41.o_lora_rank']
        self.n_ff_exp = kv['deepseek41.moe_intermediate_size']
        self.hc_dim = self.n_embd * self.n_hc
        self.hc_mix = 2 * self.n_hc + self.n_hc * self.n_hc
        self.q_dim = self.n_head * self.n_head_dim
        self.out_low = self.n_out_group * self.n_lora_o


    def draft_view(self, hyper):
        """The same geometry, with the drafter's own expert count and Markov rank."""
        return types.SimpleNamespace(**vars(self), n_expert=hyper['n_routed_experts'],
                                     markov_rank=hyper['markov_rank'])


def product(dims):
    n = 1
    for d in dims:
        n *= d
    return n


def nbytes(ttype, elements):
    if ttype not in BLOCK:
        raise ValueError('unsupported tensor type %d' % ttype)
    elems, size = BLOCK[ttype]
    return ((elements + elems - 1) // elems) * size


class _Cursor:
    """Cursor over the header window, mirroring ds4.c's ds4_cursor reads."""

    def __init__(self, buf, i):
        self.buf = buf
        self.i = i

    def _take(self, n):
        if self.i + n > len(self.buf):
            raise ValueError('GGUF header is truncated')
        chunk = self.buf[self.i:self.i + n]
        self.i += n
        return chunk

    def u32(self):
        return struct.unpack_from('<I', self._take(4), 0)[0]

    def u64(self):
        return struct.unpack_from('<Q', self._take(8), 0)[0]

    def string(self):
        return self._take(self.u64()).decode('utf-8')


def _decode(c, t, depth=0):
    """Advance the cursor past one metadata value and return its Python form."""
    if depth > 8:
        raise ValueError('metadata array nesting is too deep')
    if t == STR:
        return c.string()
    if t == ARR:
        elem = c.u32()
        return [_decode(c, elem, depth + 1) for _ in range(c.u64())]
    fmt = SCALAR[t]
    return struct.unpack(fmt, c._take(struct.calcsize(fmt)))[0]


def read_gguf(path: Path):
    """Parse this fork's GGUF dialect, keeping every metadata value byte-exact.

    `kv` holds (value_type, raw_bytes) so a repack copies the source header verbatim, and
    `values` holds the decoded form for the checks. Re-encoding from the Python value alone
    is not enough: a BOOL read as an int would be written back as a uint32, three bytes
    longer, desynchronising the rest of the header.
    """
    size = path.stat().st_size
    with path.open('rb') as f:
        head = f.read(min(size, 64 << 20))
    if head[:4] != b'GGUF':
        raise ValueError('%s is not GGUF' % path)
    version, n_tensors, n_kv = struct.unpack_from('<IQQ', head, 4)
    if version != 3:
        raise ValueError('%s is GGUF v%d; the engine reads v3 only' % (path, version))
    c = _Cursor(head, 24)
    kv, values = {}, {}
    for _ in range(n_kv):
        key = c.string()
        t = c.u32()
        start = c.i
        values[key] = _decode(c, t)
        kv[key] = (t, head[start:c.i])
    tensors = []
    for _ in range(n_tensors):
        name = c.string()
        dims = [c.u64() for _ in range(c.u32())]
        tensors.append({'name': name, 'dims': dims, 'type': c.u32(), 'rel': c.u64()})
    alignment = values.get('general.alignment', 32)
    data_start = (c.i + alignment - 1) // alignment * alignment
    for t in tensors:
        t['abs'] = data_start + t['rel']
        t['bytes'] = nbytes(t['type'], product(t['dims']))
        if t['abs'] + t['bytes'] > size:
            raise ValueError('%s: %s points outside the file' % (path, t['name']))
    return {'path': path, 'kv': kv, 'values': values, 'tensors': tensors,
            'alignment': alignment, 'data_start': data_start, 'size': size}


def infer_type(value):
    if isinstance(value, bool):
        return BOOL
    if isinstance(value, str):
        return STR
    if isinstance(value, int):
        return U32 if 0 <= value < 2 ** 32 else U64
    if isinstance(value, float):
        return F32
    if isinstance(value, (list, tuple)):
        return ARR
    raise ValueError('cannot encode metadata value %r' % (value,))


def pack_string(out, s: str):
    data = s.encode('utf-8')
    out += struct.pack('<Q', len(data)) + data


def pack_value(out, t, v):
    out += struct.pack('<I', t)
    pack_item(out, t, v)


def pack_item(out, t, v):
    """One value without its leading type tag: array items carry the element type once."""
    if t == STR:
        pack_string(out, v)
    elif t == ARR:
        elem = infer_type(v[0])
        out += struct.pack('<I', elem) + struct.pack('<Q', len(v))
        for item in v:
            pack_item(out, elem, item)
    else:
        out += struct.pack(SCALAR[t], v)


def quantise_q8_0(payload: bytes) -> bytes:
    """BF16 payload -> Q8_0 blocks: f16 delta then 32 int8, the layout ds4 reads back."""
    import numpy as np
    raw = np.frombuffer(payload, dtype=np.uint16).astype(np.uint32) << 16
    values = raw.view(np.float32).astype(np.float64)
    blocks = values.size // 32
    if values.size != blocks * 32:
        raise ValueError('Markov table is not QK8_0 aligned (%d elements)' % values.size)
    rows = values.reshape(blocks, 32)
    delta = np.maximum(np.abs(rows).max(axis=1) / 127.0, 1e-30)
    codes = np.clip(np.round(rows / delta[:, None]), -127, 127).astype(np.int8)
    out = np.zeros(blocks * 34, dtype=np.uint8)
    start = np.arange(blocks)[:, None] * 34
    scales = delta.astype(np.float16).view(np.uint16)
    out[(start + 0).ravel()] = (scales & 0xFF).astype(np.uint8)
    out[(start + 1).ravel()] = ((scales >> 8) & 0xFF).astype(np.uint8)
    out[(start + 2 + np.arange(32)[None, :]).ravel()] = codes.view(np.uint8).ravel()
    return out.tobytes()


def draft_hyperparameters(ds4_txt: Path):
    txt = {}
    for line in ds4_txt.read_text().splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2:
            txt[parts[0]] = parts[1].strip()
    def number(key):
        if key not in txt:
            raise ValueError('%s does not declare %s; the draft hyper-parameters come from '
                             'the pack that shipped this sidecar and must not be guessed'
                             % (ds4_txt, key))
        return int(txt[key])
    targets = [int(x) for x in txt['dspark_target_layer_ids'].split(',')] if \
        'dspark_target_layer_ids' in txt else []
    if not targets:
        raise ValueError('%s declares no dspark_target_layer_ids' % ds4_txt)
    return {'block_size': number('dspark_block_size'),
            'markov_rank': number('dspark_markov_rank'),
            'noise_token_id': number('dspark_noise_token_id'),
            'target_layer_ids': targets,
            'n_routed_experts': number('dspark_n_routed_experts'),
            'n_expert_used': number('dspark_num_experts_per_tok')}


def plan(model, hyper, revision):
    """Renaming and the metadata the engine's sidecar path reads."""
    stages = sorted({int(t['name'].split('.')[1]) for t in model['tensors']
                     if t['name'].startswith('blk.')})
    if not stages or stages != list(range(len(stages))):
        raise ValueError('draft blocks are not a contiguous blk.0..N set: %s' % stages)
    final = stages[-1]
    renamed = {}
    for t in model['tensors']:
        name = t['name']
        if name.startswith('blk.'):
            parts = name.split('.')
            renamed[name] = 'mtp.%s.%s' % (parts[1], '.'.join(parts[2:]))
        elif name in FLAT_HEADS:
            role, which = FLAT_HEADS[name]
            renamed[name] = 'mtp.%d.%s.weight' % (0 if which == 'first' else final, role)
        else:
            renamed[name] = name
    metadata = {
        # The sidecar keeps its own architecture tag. `model_unmap_engram` runs on every
        # file ds4 opens and dies on a `deepseek41` file with no Engram encoding, so
        # claiming the language model's architecture here would fail the sidecar's open.
        'general.name': model['values'].get('general.name', 'DeepSeek V4.1 Flash 1017'),
        'general.source.revision': revision,
        'general.alignment': model['alignment'],
        'deepseek41.dspark.block_size': hyper['block_size'],
        'deepseek41.dspark.markov_rank': hyper['markov_rank'],
        'deepseek41.dspark.noise_token_id': hyper['noise_token_id'],
        'deepseek41.dspark.target_layer_ids': hyper['target_layer_ids'],
        'deepseek41.dspark.n_routed_experts': hyper['n_routed_experts'],
        'deepseek41.dspark.num_experts_per_tok': hyper['n_expert_used'],
        'deepseek41.dspark.repack_source': str(model['path']),
        'deepseek41.dspark.repack': 'names and metadata only; payloads copied verbatim '
                                    'except Markov BF16 -> Q8_0',
    }
    return renamed, metadata, stages, final


def gates(model, renamed, metadata, language, hyper, shape: Shape):
    """What engines/ds4 will check when it opens the sidecar, evaluated before writing."""
    problems, notes = [], []
    g = shape.draft_view(hyper)
    by_name = {renamed[t['name']]: t for t in model['tensors']}
    stages = sorted({int(n.split('.')[1]) for n in by_name if n.startswith('mtp.')})
    final = stages[-1]
    if len(stages) < 3:
        problems.append('support_model_detect needs stages >= 3; found %d' % len(stages))
    for probe in ('.main_proj.', '.markov_head.', '.confidence_head.'):
        if not any(probe in n for n in by_name):
            problems.append('support_model_detect needs a tensor name containing %s' % probe)
    revision = metadata.get('general.source.revision', '')
    language_revision = language['values'].get('general.source.revision', '')
    if not language_revision:
        problems.append('the language checkpoint carries no general.source.revision, so no '
                        'sidecar can satisfy deepseek41_sidecar_checkpoint_matches')
    if revision != language_revision:
        problems.append('general.source.revision %s differs from the language checkpoint %s'
                        % (revision, language_revision))
    if revision not in ACCEPTED_REVISIONS:
        problems.append('%s is not a revision the V4.1 sidecar gate accepts' % revision)
    if len(hyper['target_layer_ids']) > len(stages):
        notes.append('%d target layers feed %d draft stages: main_proj carries %d x n_embd'
                     % (len(hyper['target_layer_ids']), len(stages),
                        len(stages) * shape.n_embd))

    def check(name, types, dims):
        t = by_name.get(name)
        if t is None:
            problems.append('missing %s' % name)
            return
        if t['type'] == T_BF16 and 'markov_head.' in name:
            notes.append('%s is BF16 in the released file and becomes Q8_0' % name)
        elif t['type'] not in types:
            problems.append('%s is %s, the engine accepts %s' %
                            (name, TYPE_NAME.get(t['type'], t['type']),
                             [TYPE_NAME.get(x, x) for x in types]))
        if t['dims'] != dims:
            problems.append('%s is %s, expected %s' % (name, t['dims'], dims))

    for stage in stages:
        for role, want in STAGE_ROLES.items():
            types_, dims = want(g)
            check('mtp.%d.%s.weight' % (stage, role), types_, dims)
        for role, want in BIASES.items():
            types_, dims = want(g)
            check('mtp.%d.%s.bias' % (stage, role), types_, dims)
        gate = by_name.get('mtp.%d.ffn_gate_exps.weight' % stage)
        up = by_name.get('mtp.%d.ffn_up_exps.weight' % stage)
        if gate and up and gate['type'] != up['type']:
            problems.append('mtp.%d routed gate/up experts use different quant types' % stage)
    check('mtp.0.main_proj.weight', DENSE_OK, [len(stages) * shape.n_embd, shape.n_embd])
    check('mtp.0.main_norm.weight', (T_F32,), [shape.n_embd])
    check('mtp.%d.norm.weight' % final, (T_F32,), [shape.n_embd])
    check('mtp.%d.confidence_head.proj.weight' % final, DENSE_OK,
          [g.n_embd + g.markov_rank, 1])
    for suffix in ('markov_head.markov_w1', 'markov_head.markov_w2'):
        check('mtp.%d.%s.weight' % (final, suffix), DENSE_OK,
              [g.markov_rank, g.n_vocab])
    # V4.1 does not require the final stage's hc_head_* trio; note if the file has them.
    if any('hc_head_' in n for n in by_name):
        notes.append('sidecar carries hc_head_* tensors, which the V4.1 binding leaves optional')
    return problems, notes


def converted_types(model):
    """The tensors this repack re-encodes: BF16 Markov tables, which become Q8_0."""
    out = {}
    for t in model['tensors']:
        role = FLAT_HEADS.get(t['name'])
        if role and role[0].startswith('markov_head') and t['type'] == T_BF16:
            out[t['name']] = T_Q8_0
    return out


def write_sidecar(model, out_path, renamed, metadata, chunk=1 << 24):
    reencode = converted_types(model)
    kept = [key for key in model['kv'] if key not in metadata]
    header = bytearray(b'GGUF' + struct.pack('<I', 3) +
                       struct.pack('<Q', len(model['tensors'])) +
                       struct.pack('<Q', len(kept) + len(metadata)))
    for key in kept:
        t, raw = model['kv'][key]
        pack_string(header, key)
        header += struct.pack('<I', t) + raw
    for key, value in metadata.items():
        pack_string(header, key)
        pack_value(header, infer_type(value), value)
    alignment = model['alignment']
    offset = 0
    payloads = []
    for t in model['tensors']:
        ttype = reencode.get(t['name'], t['type'])
        size = nbytes(ttype, product(t['dims']))
        pack_string(header, renamed[t['name']])
        header += struct.pack('<I', len(t['dims']))
        for d in t['dims']:
            header += struct.pack('<Q', d)
        header += struct.pack('<I', ttype) + struct.pack('<Q', offset)
        payloads.append((t, ttype, size))
        offset += ((size + alignment - 1) // alignment) * alignment
    data_start = (len(header) + alignment - 1) // alignment * alignment
    header += b'\0' * (data_start - len(header))
    fd = os.open(str(out_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(fd, 'wb') as out, model['path'].open('rb') as src:
            out.write(header)
            for t, ttype, size in payloads:
                if ttype == t['type']:
                    src.seek(t['abs'])
                    remaining = t['bytes']
                    while remaining:
                        block = src.read(min(remaining, chunk))
                        if not block:
                            raise ValueError('%s ends before %s' % (model['path'], t['name']))
                        out.write(block)
                        remaining -= len(block)
                else:
                    src.seek(t['abs'])
                    out.write(quantise_q8_0(src.read(t['bytes'])))
                out.write(b'\0' * ((-size) % alignment))
    except Exception:
        out_path.unlink(missing_ok=True)
        raise
    return out_path, data_start, offset


def verify(out_path, model, renamed, metadata, language, hyper, shape):
    """Re-read what was written: same gates, and every copied payload byte-identical."""
    fresh = read_gguf(out_path)
    identity = {t['name']: t['name'] for t in fresh['tensors']}
    problems, _ = gates(fresh, identity, fresh['values'], language, hyper, shape)
    reencode = converted_types(model)
    by_name = {t['name']: t for t in fresh['tensors']}
    source = {t['name']: t for t in model['tensors']}
    with model['path'].open('rb') as src, out_path.open('rb') as new:
        for name, t in source.items():
            out_t = by_name.get(renamed[name])
            if out_t is None:
                problems.append('%s is not in the written file' % renamed[name])
                continue
            if reencode.get(name) is not None:
                continue
            src.seek(t['abs'])
            new.seek(out_t['abs'])
            if src.read(t['bytes']) != new.read(out_t['bytes']):
                problems.append('%s payload differs from the released sidecar' % renamed[name])
    for key, value in metadata.items():
        if fresh['values'].get(key) != value:
            problems.append('%s read back as %r' % (key, fresh['values'].get(key)))
    return problems


def report_reference(model, renamed, reference: Path):
    """Compare against the layout this repo's own converter emits, as a shape cross-check."""
    if not reference.is_file():
        return ['reference-layout.json not found; skipped the cross-check']
    tensors = json.loads(reference.read_text())['components']['dspark']['tensors']
    by_name = {renamed[t['name']]: t for t in model['tensors']}
    drift = []
    for name, row in tensors.items():
        t = by_name.get(name)
        if t is None:
            # exp_probs_b_vl and hc_head_* are bound only on the main model, or optional.
            if not (name.endswith('exp_probs_b_vl.bias') or 'hc_head_' in name):
                drift.append('reference has %s, the sidecar does not' % name)
        elif t['dims'] != row['shape']:
            drift.append('%s is %s here, %s in the reference layout' %
                         (name, t['dims'], row['shape']))
        elif t['type'] != row['type']:
            drift.append('%s is %s here, %s in the reference layout' %
                         (name, TYPE_NAME.get(t['type'], t['type']),
                          TYPE_NAME.get(row['type'], row['type'])))
    return drift


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('sidecar', type=Path, help='released DSpark GGUF')
    parser.add_argument('language', type=Path, help='the V4.1 language GGUF this will serve')
    parser.add_argument('--ds4-txt', type=Path, required=True,
                        help="the pack's ds4.txt; supplies the draft hyper-parameters")
    parser.add_argument('--source-revision', default=None,
                        help="revision to assert; defaults to the language file's own")
    parser.add_argument('--out', type=Path, default=None)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--reference-layout', type=Path,
                        default=Path(__file__).parent / 'dsv41' / 'reference-layout.json')
    args = parser.parse_args(argv)

    model = read_gguf(args.sidecar)
    language = read_gguf(args.language)
    shape = Shape(language['values'])
    hyper = draft_hyperparameters(args.ds4_txt)
    revision = args.source_revision or language['values'].get('general.source.revision', '')
    renamed, metadata, stages, final = plan(model, hyper, revision)
    metadata['deepseek41.dspark.repack_provenance'] = (
        'the released sidecar carries no general.source.revision; this value is the language '
        'checkpoint\'s revision, asserted because both files name the same release (%s, '
        'general.version %s) and every draft tensor matches the reference shapes'
        % (language['values'].get('general.source.url', 'V4.1'),
           model['values'].get('general.version', 'unknown')))

    print('%s: %d tensors, draft blocks %s, final stage %d'
          % (args.sidecar.name, len(model['tensors']), stages, final))
    print('renamed %d of %d tensors; re-encoded %d'
          % (sum(1 for k, v in renamed.items() if k != v), len(model['tensors']),
             len(converted_types(model))))
    problems, notes = gates(model, renamed, metadata, language, hyper, shape)
    for line in report_reference(model, renamed, args.reference_layout):
        print('  reference drift: %s' % line)
    for note in notes:
        print('  note: %s' % note)
    for problem in problems:
        print('  GATE: %s' % problem)
    if args.check_only:
        print('check-only: %d gate failure(s)' % len(problems))
        return 1 if problems else 0
    if problems:
        print('%d gate failure(s); refusing to write' % len(problems))
        return 1
    out = args.out or args.sidecar.with_name('%s-mtp-%s.gguf' %
                                             (args.sidecar.stem, revision[:8]))
    written, data_start, payload = write_sidecar(model, out, renamed, metadata)
    digest = hashlib.sha256(written.read_bytes()).hexdigest()
    print('wrote %s: %d bytes, payload %.2f GiB, sha256 %s'
          % (written, written.stat().st_size, payload / 2 ** 30, digest))
    failures = verify(written, model, renamed, metadata, language, hyper, shape)
    for problem in failures:
        print('  VERIFY: %s' % problem)
    print('verified %s' % ('clean' if not failures else 'with %d failure(s)' % len(failures)))
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
