"""The B5 routing calibration is offered only when it names the model being served.

The failure this bounds is measured, not hypothetical: `presets/dsv41-affinity-spreads.txt` is
calibrated for a 315838021632-byte V4.1 file, this box serves a 365713686528-byte one, and the
launcher used to hand the preset to the producer unconditionally. `ds4x_set_affinity` then refused
the reasoning phase's bias at the first decoded token - after a 34 s prefill, with the producer
holding ~30 GiB of VRAM - as `Live bias refused`.
"""
import unittest
from pathlib import Path
import tempfile

from slstack.affinity_spread import DSV41_EXPERTS, DSV41_LAYERS, declared, matches_model
from slstack.single_runtime import affinity_spread

ROOT = Path(__file__).resolve().parents[1]
PACKAGED = ROOT / 'presets' / 'dsv41-affinity-spreads.txt'


def write_spread(directory, model_bytes, experts=DSV41_EXPERTS, layers=DSV41_LAYERS,
                 digest='9bebbe7c246be2dd'):
    target = Path(directory) / f'spreads-{model_bytes}-{experts}-{layers}.txt'
    target.write_text(f'DS4AFFINITY1 {model_bytes} {experts} {layers} {digest}\n0 0.654706478 64\n',
                      encoding='utf-8')
    return target


def write_model(directory, size):
    target = Path(directory) / f'model-{size}.gguf'
    target.write_bytes(b'\0' * size)
    return target


class SpreadHeaderTests(unittest.TestCase):
    def test_packaged_preset_names_the_model_it_was_measured_on(self):
        self.assertEqual(declared(PACKAGED), (315838021632, 384, 40, '9bebbe7c246be2dd'))

    def test_a_file_without_the_header_declares_nothing(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertIsNone(declared(write_model(folder, 12)))

    def test_a_truncated_header_declares_nothing(self):
        with tempfile.TemporaryDirectory() as folder:
            broken = Path(folder) / 'broken.txt'
            broken.write_text('DS4AFFINITY1 12 384 40\n', encoding='utf-8')
            self.assertIsNone(declared(broken))


class MatchesModelTests(unittest.TestCase):
    def test_matching_bytes_and_geometry(self):
        with tempfile.TemporaryDirectory() as folder:
            model, spread = write_model(folder, 4096), write_spread(folder, 4096)
            self.assertTrue(matches_model(spread, model))

    def test_other_model_bytes_do_not_match(self):
        with tempfile.TemporaryDirectory() as folder:
            model, spread = write_model(folder, 4096), write_spread(folder, 8192)
            self.assertFalse(matches_model(spread, model))

    def test_other_geometry_does_not_match(self):
        with tempfile.TemporaryDirectory() as folder:
            model = write_model(folder, 4096)
            self.assertFalse(matches_model(write_spread(folder, 4096, experts=256), model))
            self.assertFalse(matches_model(write_spread(folder, 4096, layers=61), model))

    def test_the_packaged_preset_is_not_for_this_box(self):
        # The 5090's file is bigger than the one the preset was measured on.
        with tempfile.TemporaryDirectory() as folder:
            self.assertFalse(matches_model(PACKAGED, write_model(folder, 365713686528 % 4096)))

    def test_missing_files_match_nothing(self):
        with tempfile.TemporaryDirectory() as folder:
            model = write_model(folder, 64)
            self.assertFalse(matches_model(Path(folder) / 'absent.txt', model))
            self.assertFalse(matches_model(model, Path(folder) / 'absent.gguf'))
            self.assertFalse(matches_model(model, None))


class ResolveTests(unittest.TestCase):
    def test_nothing_configured_and_no_match_offers_no_calibration(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertIsNone(affinity_spread({'model': str(write_model(folder, 4096)),
                                               'cache_affinity_spread_file': ''}))

    def test_nothing_configured_and_the_packaged_preset_would_be_offered_on_its_own_model(self):
        # The preset's own model: 315838021632 bytes cannot be written here, so the check is
        # proved through an explicit file with the same header instead.
        with tempfile.TemporaryDirectory() as folder:
            model, spread = write_model(folder, 4096), write_spread(folder, 4096)
            self.assertEqual(affinity_spread({'model': str(model),
                                              'cache_affinity_spread_file': str(spread)}), spread)

    def test_an_explicit_wrong_model_calibration_is_a_launch_error(self):
        with tempfile.TemporaryDirectory() as folder:
            model, spread = write_model(folder, 4096), write_spread(folder, 999999999)
            with self.assertRaises(ValueError) as caught:
                affinity_spread({'model': str(model), 'cache_affinity_spread_file': str(spread)})
            text = str(caught.exception)
            self.assertIn('calibrated for another model', text)
            self.assertIn('999999999 bytes', text)
            self.assertIn('DS4_CACHE_AFFINITY_CALIBRATE', text)

    def test_an_unreadable_explicit_calibration_is_refused_the_same_way(self):
        with tempfile.TemporaryDirectory() as folder:
            model, junk = write_model(folder, 4096), write_model(folder, 32)
            with self.assertRaises(ValueError):
                affinity_spread({'model': str(model), 'cache_affinity_spread_file': str(junk)})


if __name__ == '__main__':
    unittest.main()
