"""The DSpark sidecar repack must survive a byte-exact round trip, or it is not a sidecar.

`engines/ds4` reads a V4.1 draft file with its own GGUF dialect (value types UINT8=0 through
FLOAT64=12, u64 tensor dimensions, array element type written once). Two encodings were
measured wrong while writing `scripts/repack_dsv41_dspark_sidecar.py`: a BOOL metadata value
decoded as an int re-encodes as a uint32 - three bytes longer, desynchronising every KV
after it - and an array whose items each repeat the element type pushes the tensor directory
past the header. Both produced a file that parses as garbage rather than failing loudly, so
the repack's own reader is the thing under test here, on a geometry small enough to build.
"""
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

import repack_dsv41_dspark_sidecar as repack

U8, I8, U16, I16, U32, I32, F32, BOOL, STR, ARR, U64, I64, F64 = range(13)
REVISION = '1bae0e1088993b9b82de08f59de5be5a34e42170'

# A miniature V4.1: the same relations the engine checks, in bytes that fit a test.
G = {'hidden_size': 64, 'vocab_size': 96, 'hc_mult': 4, 'q_lora_q': 32, 'heads': 4,
     'head_dim': 16, 'o_groups': 2, 'o_lora': 16, 'ff_exp': 32, 'experts': 8,
     'markov_rank': 32, 'stages': 3}


def emit(out, key, t, value):
    repack.pack_string(out, key)
    out += struct.pack('<I', t)
    repack.pack_item(out, t, value)   # the type tag is written once, above


def write_gguf(path, kv, tensors):
    """kv is a list of (key, value_type, python_value); tensors of (name, dims, type, payload)."""
    out = bytearray(b'GGUF' + struct.pack('<IQQ', 3, len(tensors), len(kv)))
    for key, t, value in kv:
        emit(out, key, t, value)
    offset = 0
    payloads = []
    for name, dims, ttype, payload in tensors:
        repack.pack_string(out, name)
        out += struct.pack('<I', len(dims))
        for d in dims:
            out += struct.pack('<Q', d)
        out += struct.pack('<I', ttype) + struct.pack('<Q', offset)
        size = repack.nbytes(ttype, repack.product(dims))
        assert len(payload) == size, (name, len(payload), size)
        payloads.append((payload, size))
        offset += ((size + 32 - 1) // 32) * 32
    out += b'\0' * ((-len(out)) % 32)
    for payload, size in payloads:
        out += payload + b'\0' * ((-size) % 32)
    path.write_bytes(bytes(out))
    return path


def bf16(values):
    import numpy as np
    return (np.asarray(values, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16).tobytes()


def draft_tensors(stage, expert_payload, markov=False):
    g = G
    rows = {
        'hc_attn_fn': (repack.T_F32, [g['hidden_size'] * g['hc_mult'],
                                      2 * g['hc_mult'] + g['hc_mult'] ** 2]),
        'hc_attn_scale': (repack.T_F32, [3]),
        'hc_attn_base': (repack.T_F32, [2 * g['hc_mult'] + g['hc_mult'] ** 2]),
        'attn_norm': (repack.T_F32, [g['hidden_size']]),
        'attn_q_a': (repack.T_Q8_0, [g['hidden_size'], g['q_lora_q']]),
        'attn_q_a_norm': (repack.T_F32, [g['q_lora_q']]),
        'attn_q_b': (repack.T_Q8_0, [g['q_lora_q'], g['heads'] * g['head_dim']]),
        'attn_kv': (repack.T_Q8_0, [g['hidden_size'], g['head_dim']]),
        'attn_kv_a_norm': (repack.T_F32, [g['head_dim']]),
        'attn_sinks': (repack.T_F32, [g['heads']]),
        'attn_output_a': (repack.T_Q8_0, [g['head_dim'] * (g['heads'] // g['o_groups']),
                                          g['o_groups'] * g['o_lora']]),
        'attn_output_b': (repack.T_Q8_0, [g['o_groups'] * g['o_lora'], g['hidden_size']]),
        'hc_ffn_fn': (repack.T_F32, [g['hidden_size'] * g['hc_mult'],
                                     2 * g['hc_mult'] + g['hc_mult'] ** 2]),
        'hc_ffn_scale': (repack.T_F32, [3]),
        'hc_ffn_base': (repack.T_F32, [2 * g['hc_mult'] + g['hc_mult'] ** 2]),
        'ffn_norm': (repack.T_F32, [g['hidden_size']]),
        'ffn_gate_inp': (repack.T_F32, [g['hidden_size'], g['experts']]),
        'ffn_gate_exps': (repack.T_MXFP4, [g['hidden_size'], g['ff_exp'], g['experts']]),
        'ffn_up_exps': (repack.T_MXFP4, [g['hidden_size'], g['ff_exp'], g['experts']]),
        'ffn_down_exps': (repack.T_MXFP4, [g['ff_exp'], g['hidden_size'], g['experts']]),
        'ffn_gate_shexp': (repack.T_Q8_0, [g['hidden_size'], g['ff_exp']]),
        'ffn_up_shexp': (repack.T_Q8_0, [g['hidden_size'], g['ff_exp']]),
        'ffn_down_shexp': (repack.T_Q8_0, [g['ff_exp'], g['hidden_size']]),
    }
    out = [('blk.%d.%s.weight' % (stage, role), dims, ttype,
            expert_payload if ttype == repack.T_MXFP4 else
            b'\0' * repack.nbytes(ttype, repack.product(dims)))
           for role, (ttype, dims) in rows.items()]
    out.append(('blk.%d.exp_probs_b.bias' % stage, [g['experts']], repack.T_F32,
                b'\0' * 4 * g['experts']))
    if stage == 0:
        out.append(('fc.weight', [g['stages'] * g['hidden_size'], g['hidden_size']],
                    repack.T_Q8_0, b'\0' * repack.nbytes(repack.T_Q8_0,
                                                         g['stages'] * g['hidden_size'] ** 2)))
        out.append(('enc.output_norm.weight', [g['hidden_size']], repack.T_F32,
                    b'\0' * 4 * g['hidden_size']))
    if stage == G['stages'] - 1 and markov:
        table = [3.0 if i % 7 else -1.5 for i in range(g['markov_rank'] * g['vocab_size'])]
        out.append(('output_norm.weight', [g['hidden_size']], repack.T_F32,
                    b'\0' * 4 * g['hidden_size']))
        out.append(('conf_proj.weight', [g['hidden_size'] + g['markov_rank'], 1],
                    repack.T_F32, b'\0' * 4 * (g['hidden_size'] + g['markov_rank'])))
        payload = bf16(table)
        out.append(('markov_w1.weight', [g['markov_rank'], g['vocab_size']],
                    repack.T_BF16, payload))
        out.append(('markov_w2.weight', [g['markov_rank'], g['vocab_size']],
                    repack.T_BF16, payload))
    return out


class SidecarRepackTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.expert_payload = bytes(range(17)) * (repack.nbytes(repack.T_MXFP4,
                                                               G['hidden_size'] * G['ff_exp']
                                                               * G['experts']) // 17)
        language = [
            ('general.architecture', STR, 'deepseek41'),
            ('general.source.revision', STR, REVISION),
            ('general.alignment', U32, 32),
            ('deepseek41.hidden_size', U32, G['hidden_size']),
            ('deepseek41.vocab_size', U32, G['vocab_size']),
            ('deepseek41.hc_mult', U32, G['hc_mult']),
            ('deepseek41.q_lora_rank', U32, G['q_lora_q']),
            ('deepseek41.num_attention_heads', U32, G['heads']),
            ('deepseek41.head_dim', U32, G['head_dim']),
            ('deepseek41.o_groups', U32, G['o_groups']),
            ('deepseek41.o_lora_rank', U32, G['o_lora']),
            ('deepseek41.moe_intermediate_size', U32, G['ff_exp']),
            ('deepseek41.n_routed_experts', U32, 384),
            ('deepseek41.num_experts_per_tok', U32, 6),
        ]
        self.language = write_gguf(self.root / 'language.gguf', language,
                                   [('blk.0.attn_norm.weight', [G['hidden_size']],
                                     repack.T_F32, b'\0' * 4 * G['hidden_size'])])
        sidecar = [
            ('general.architecture', STR, 'dflash'),
            ('general.name', STR, 'DeepSeek V4.1 Flash 1017'),
            ('general.version', STR, '1017'),
            ('dflash.block_count', U32, G['stages']),
            # A BOOL is one byte; decoding it as an int and re-encoding it as a uint32 is
            # what shifted every KV after it and made the written header unreadable.
            ('dflash.has_confidence_head', BOOL, True),
            ('dflash.swiglu_clamp_exp', ARR, [10.0, 10.0, 10.0]),
            ('tokenizer.ggml.tokens', ARR, ['a', 'bb', 'ccc']),
        ]
        tensors = []
        for stage in range(G['stages']):
            tensors += draft_tensors(stage, self.expert_payload,
                                     markov=stage == G['stages'] - 1)
        self.sidecar = write_gguf(self.root / 'dspark.gguf', sidecar, tensors)
        self.ds4_txt = self.root / 'ds4.txt'
        self.ds4_txt.write_text(
            'dspark_block_size 5\ndspark_markov_rank %d\ndspark_noise_token_id 128799\n'
            'dspark_target_layer_ids 37,38,39\ndspark_n_routed_experts %d\n'
            'dspark_num_experts_per_tok 3\n' % (G['markov_rank'], G['experts']),
            encoding='utf-8')

    def args(self, out=None):
        argv = [str(self.sidecar), str(self.language), '--ds4-txt', str(self.ds4_txt),
                '--reference-layout', str(self.root / 'absent.json')]
        if out is not None:
            argv += ['--out', str(out)]
        return argv

    def test_check_only_reports_no_gate_failure(self):
        self.assertEqual(0, repack.main(self.args() + ['--check-only']))

    def test_repacked_sidecar_reads_back_and_keeps_payloads(self):
        out = self.root / 'bound.gguf'
        self.assertEqual(0, repack.main(self.args(out=out)))
        fresh = {t['name']: t for t in repack.read_gguf(out)['tensors']}
        source = repack.read_gguf(self.sidecar)
        renamed, _, _, _ = repack.plan(source,
                                      repack.draft_hyperparameters(self.ds4_txt), REVISION)
        self.assertIn('mtp.0.main_proj.weight', fresh)
        self.assertIn('mtp.2.markov_head.markov_w1.weight', fresh)
        self.assertIn('mtp.2.confidence_head.proj.weight', fresh)
        self.assertNotIn('blk.0.attn_norm.weight', fresh)
        reencoded = set(repack.converted_types(source))
        with self.sidecar.open('rb') as old, out.open('rb') as new:
            for t in source['tensors']:
                moved = fresh[renamed[t['name']]]
                if t['name'] in reencoded:
                    self.assertEqual(repack.T_Q8_0, moved['type'], renamed[t['name']])
                    continue
                self.assertEqual(moved['type'], t['type'], renamed[t['name']])
                old.seek(t['abs'])
                new.seek(moved['abs'])
                self.assertEqual(old.read(t['bytes']), new.read(moved['bytes']),
                                 renamed[t['name']])

    def test_metadata_types_survive_the_repack(self):
        out = self.root / 'types.gguf'
        self.assertEqual(0, repack.main(self.args(out=out)))
        fresh = repack.read_gguf(out)
        self.assertEqual(repack.BOOL, fresh['kv']['dflash.has_confidence_head'][0])
        self.assertEqual(repack.ARR, fresh['kv']['tokenizer.ggml.tokens'][0])
        self.assertEqual(['a', 'bb', 'ccc'], fresh['values']['tokenizer.ggml.tokens'])
        self.assertEqual([10.0, 10.0, 10.0], fresh['values']['dflash.swiglu_clamp_exp'])
        self.assertEqual(5, fresh['values']['deepseek41.dspark.block_size'])
        self.assertEqual([37, 38, 39], fresh['values']['deepseek41.dspark.target_layer_ids'])
        self.assertEqual(REVISION, fresh['values']['general.source.revision'])
        self.assertEqual('dflash', fresh['values']['general.architecture'])

    def test_a_sidecar_for_another_release_is_refused(self):
        self.assertEqual(0, repack.main(self.args() + ['--check-only']))
        other = self.root / 'other-revision.gguf'
        write_gguf(other,
                   [('general.architecture', STR, 'deepseek41'),
                    ('general.source.revision', STR, '0' * 40),
                    ('general.alignment', U32, 32),
                    ('deepseek41.hidden_size', U32, G['hidden_size']),
                    ('deepseek41.vocab_size', U32, G['vocab_size']),
                    ('deepseek41.hc_mult', U32, G['hc_mult']),
                    ('deepseek41.q_lora_rank', U32, G['q_lora_q']),
                    ('deepseek41.num_attention_heads', U32, G['heads']),
                    ('deepseek41.head_dim', U32, G['head_dim']),
                    ('deepseek41.o_groups', U32, G['o_groups']),
                    ('deepseek41.o_lora_rank', U32, G['o_lora']),
                    ('deepseek41.moe_intermediate_size', U32, G['ff_exp']),
                    ('deepseek41.n_routed_experts', U32, 384)],
                   [('blk.0.attn_norm.weight', [G['hidden_size']],
                     repack.T_F32, b'\0' * 4 * G['hidden_size'])])
        self.assertEqual(1, repack.main([str(self.sidecar), str(other), '--ds4-txt',
                                         str(self.ds4_txt), '--reference-layout',
                                         str(self.root / 'absent.json'), '--check-only']))

    def test_missing_draft_hyperparameters_are_a_hard_error(self):
        empty = self.root / 'empty.txt'
        empty.write_text('vocab_size 129280\n', encoding='utf-8')
        with self.assertRaises(ValueError) as caught:
            repack.main([str(self.sidecar), str(self.language), '--ds4-txt', str(empty),
                         '--reference-layout', str(self.root / 'absent.json'), '--check-only'])
        self.assertIn('dspark_target_layer_ids', str(caught.exception))


class MarkovQuantiserTests(unittest.TestCase):
    def test_blocks_match_the_layout_ds4_decodes(self):
        import numpy as np
        values = [3.0, -2.5, 0.0, 1.25] * 16
        packed = repack.quantise_q8_0(bf16(values))
        self.assertEqual(2 * 34, len(packed))
        decoded = []
        for block in range(len(packed) // 34):
            scale = float(np.frombuffer(packed[block * 34:block * 34 + 2],
                                        dtype=np.float16)[0])
            codes = np.frombuffer(packed[block * 34 + 2:block * 34 + 34], dtype=np.int8)
            decoded.extend(scale * codes.astype(float))
        self.assertEqual(64, len(decoded))
        self.assertLess(max(abs(a - b) for a, b in zip(decoded, values)), 3.0 / 127 + 1e-6)


if __name__ == '__main__':
    unittest.main()
