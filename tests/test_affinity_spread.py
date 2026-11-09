"""The B5 routing calibration is offered only when it names the model being served.

Two measured failures sit behind this file. `presets/dsv41-affinity-spreads.txt` is calibrated for
a different V4.1 build than this box serves, and the launcher used to hand it to the producer
unconditionally; `ds4x_set_affinity` then refused the reasoning phase's bias at the first decoded
token - after a 34 s prefill, with the producer holding ~30 GiB of VRAM - as `Live bias refused`.
The second was found on 2026-11-09, by the check itself: the producer calibrated this box's file
and wrote `DS4AFFINITY1 162955640832 384 40 …`, which is `ds4_model.size` - the file minus its
disk-only Engram region, per `model_unmap_engram` - while `matches_model` compared the file's
length and refused the calibration the engine had just produced. So the launcher mirrors the four
numbers `cuda_affinity_model` compares: bound bytes, the 64 KiB FNV-1a header hash, experts, layers.
"""
import struct
import unittest
from pathlib import Path
import tempfile

from slstack.affinity_spread import (DSV41_EXPERTS, DSV41_LAYERS, ENGRAM_TENSOR, binding,
                                     declared, matches_model)
from slstack.single_runtime import affinity_spread

ROOT = Path(__file__).resolve().parents[1]
PACKAGED = ROOT / 'presets' / 'dsv41-affinity-spreads.txt'
# This fork's GGUF value types: STRING=8, UINT32=4 (engines/ds4/ds4.c GGUF_VALUE_*).
STR, U32 = 8, 4
F32, I8 = 0, 24


def emit(out, key, t, value):
    data = key.encode()
    out += struct.pack('<Q', len(data)) + data + struct.pack('<I', t)
    if t == STR:
        payload = value.encode()
        out += struct.pack('<Q', len(payload)) + payload
    else:
        out += struct.pack('<I', value)


def tensor(out, name, ttype, dims, rel):
    data = name.encode()
    out += struct.pack('<Q', len(data)) + data + struct.pack('<I', len(dims))
    for d in dims:
        out += struct.pack('<Q', d)
    out += struct.pack('<I', ttype) + struct.pack('<Q', rel)


def write_v41_model(directory, total_bytes, alignment=32, engram=True, encoding=True):
    """A V4.1-shaped GGUF: metadata, a resident tensor, an Engram table, then padding.

    `ds4_model.size` is the Engram table's offset, so that - not `total_bytes` - is what a
    calibration for this file must name.
    """
    out = bytearray(b'GGUF' + struct.pack('<IQQ', 3, 2 if engram else 1, 3 if encoding else 2))
    emit(out, 'general.architecture', STR, 'deepseek41')
    emit(out, 'general.alignment', U32, alignment)
    if encoding:
        emit(out, 'deepseek41.engram.encoding', STR, 'e4m3_e8m0_32_row264')
    resident_rel = 0
    resident_bytes = 8 * 4
    engram_rel = ((resident_rel + resident_bytes + alignment - 1) // alignment) * alignment
    tensor(out, 'blk.0.attn_norm.weight', F32, [8], resident_rel)
    if engram:
        tensor(out, ENGRAM_TENSOR, I8, [264, 16], engram_rel)
    data_start = (len(out) + alignment - 1) // alignment * alignment
    if engram and data_start + engram_rel > total_bytes:
        raise ValueError('the Engram tensor has to fit inside the file')
    out += b'\0' * (total_bytes - len(out))
    target = Path(directory) / f'model-{total_bytes}-{alignment}-{engram}-{encoding}.gguf'
    target.write_bytes(bytes(out))
    return target, data_start + (engram_rel if engram else 0)


def write_spread(directory, model_bytes, model, experts=DSV41_EXPERTS, layers=DSV41_LAYERS,
                 digest=None):
    if digest is None:
        digest = '%016x' % binding(model)[1]
    target = Path(directory) / f'spreads-{model_bytes}-{experts}-{layers}-{digest}.txt'
    target.write_text(f'DS4AFFINITY1 {model_bytes} {experts} {layers} {digest}\n0 0.654706478 61\n',
                      encoding='utf-8')
    return target


class SpreadHeaderTests(unittest.TestCase):
    def test_packaged_preset_names_the_model_it_was_measured_on(self):
        self.assertEqual(declared(PACKAGED), (315838021632, 384, 40, '9bebbe7c246be2dd'))

    def test_a_file_without_the_header_declares_nothing(self):
        with tempfile.TemporaryDirectory() as folder:
            plain = Path(folder) / 'model.bin'
            plain.write_bytes(b'\0' * 12)
            self.assertIsNone(declared(plain))

    def test_a_truncated_header_declares_nothing(self):
        with tempfile.TemporaryDirectory() as folder:
            broken = Path(folder) / 'broken.txt'
            broken.write_text('DS4AFFINITY1 12 384 40\n', encoding='utf-8')
            self.assertIsNone(declared(broken))


class BindingTests(unittest.TestCase):
    def test_the_bound_size_is_the_engram_offset_not_the_file_length(self):
        with tempfile.TemporaryDirectory() as folder:
            model, bound = write_v41_model(folder, 4096)
            self.assertEqual(binding(model), (bound, binding(model)[1]))
            self.assertLess(bound, model.stat().st_size)

    def test_a_file_without_an_engram_region_binds_its_length(self):
        with tempfile.TemporaryDirectory() as folder:
            model, _ = write_v41_model(folder, 4096, engram=False)
            self.assertEqual(binding(model)[0], model.stat().st_size)

    def test_a_non_gguf_file_binds_nothing(self):
        with tempfile.TemporaryDirectory() as folder:
            plain = Path(folder) / 'model.bin'
            plain.write_bytes(b'\0' * 4096)
            self.assertIsNone(binding(plain))

    def test_the_hash_follows_the_engine_s_own_recipe(self):
        # cuda_affinity_header_hash: FNV-1a over the first min(size, 65536) bytes, then xor size.
        with tempfile.TemporaryDirectory() as folder:
            model, bound = write_v41_model(folder, 4096)
            digest = 14695981039346656037
            for byte in model.read_bytes()[:min(bound, 1 << 16)]:
                digest = ((digest ^ byte) * 1099511628211) & ((1 << 64) - 1)
            self.assertEqual(binding(model)[1], digest ^ bound)


class MatchesModelTests(unittest.TestCase):
    def test_matching_bound_bytes_hash_and_geometry(self):
        with tempfile.TemporaryDirectory() as folder:
            model, bound = write_v41_model(folder, 4096)
            self.assertTrue(matches_model(write_spread(folder, bound, model), model))

    def test_a_spread_naming_the_file_length_is_refused(self):
        # The regression this fixes: the launcher refusing the calibration the engine wrote,
        # because the engine records ds4_model.size and the launcher compared st_size.
        with tempfile.TemporaryDirectory() as folder:
            model, bound = write_v41_model(folder, 4096)
            self.assertLess(bound, model.stat().st_size)
            self.assertFalse(matches_model(write_spread(folder, model.stat().st_size, model), model))

    def test_a_spread_with_another_header_hash_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            model, bound = write_v41_model(folder, 4096)
            other, _ = write_v41_model(folder, 4096, alignment=64)
            self.assertFalse(matches_model(write_spread(folder, bound, other), model))

    def test_other_geometry_does_not_match(self):
        with tempfile.TemporaryDirectory() as folder:
            model, bound = write_v41_model(folder, 4096)
            self.assertFalse(matches_model(write_spread(folder, bound, model, experts=256), model))
            self.assertFalse(matches_model(write_spread(folder, bound, model, layers=61), model))

    def test_the_packaged_preset_is_not_for_this_box(self):
        with tempfile.TemporaryDirectory() as folder:
            model, _ = write_v41_model(folder, 4096)
            self.assertFalse(matches_model(PACKAGED, model))

    def test_missing_files_match_nothing(self):
        with tempfile.TemporaryDirectory() as folder:
            model, _ = write_v41_model(folder, 512)
            self.assertFalse(matches_model(Path(folder) / 'absent.txt', model))
            self.assertFalse(matches_model(model, Path(folder) / 'absent.gguf'))
            self.assertFalse(matches_model(model, None))


class ResolveTests(unittest.TestCase):
    def test_nothing_configured_and_no_match_offers_no_calibration(self):
        with tempfile.TemporaryDirectory() as folder:
            model, _ = write_v41_model(folder, 4096)
            self.assertIsNone(affinity_spread({'model': str(model),
                                               'cache_affinity_spread_file': ''}))

    def test_a_calibration_written_for_this_model_is_accepted(self):
        with tempfile.TemporaryDirectory() as folder:
            model, bound = write_v41_model(folder, 4096)
            spread = write_spread(folder, bound, model)
            self.assertEqual(affinity_spread({'model': str(model),
                                              'cache_affinity_spread_file': str(spread)}), spread)

    def test_an_explicit_wrong_model_calibration_is_a_launch_error(self):
        with tempfile.TemporaryDirectory() as folder:
            model, _ = write_v41_model(folder, 4096)
            spread = write_spread(folder, 999999999, model)
            with self.assertRaises(ValueError) as caught:
                affinity_spread({'model': str(model), 'cache_affinity_spread_file': str(spread)})
            text = str(caught.exception)
            self.assertIn('calibrated for another model', text)
            self.assertIn('999999999 bytes', text)
            self.assertIn('DS4_CACHE_AFFINITY_CALIBRATE', text)

    def test_an_unreadable_explicit_calibration_is_refused_the_same_way(self):
        with tempfile.TemporaryDirectory() as folder:
            model, _ = write_v41_model(folder, 4096)
            junk = Path(folder) / 'junk.txt'
            junk.write_text('not a spread\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                affinity_spread({'model': str(model), 'cache_affinity_spread_file': str(junk)})


if __name__ == '__main__':
    unittest.main()
