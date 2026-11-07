"""CPU capability regressions; these do not stand in for the real Jeff proof."""
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
# The streamed trainer lives in the vendored engines tree, which is how the launcher
# puts it on a worker's PYTHONPATH (slstack/envvars.py's library_pythonpath).
MEDIA_PYTHONPATH = os.pathsep.join((str(ROOT), str(ROOT / 'vendor' / 'engines')))


class Dsv41RecipeTests(unittest.TestCase):
    def test_existing_streamed_trainer_accepts_dsv41_architecture(self):
        result = subprocess.run([sys.executable, '-X', 'utf8', '-B', '-m', 'cosmos_windows.gguf_qat.train_gguf_lora_streamed',
                                 '--arch', 'deepseek41', '--gguf', 'TEST-not-loaded.gguf', '--help'],
                                cwd=ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace',
                                env={**os.environ, 'PYTHONPATH': MEDIA_PYTHONPATH},
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_native_startup_accepts_lora_before_help_without_loading_weights(self):
        binary = ROOT / 'engines/ds4/ds4-server'
        self.assertTrue(binary.is_file(), 'Build the native test binary first')
        result = subprocess.run([str(binary), '--lora', 'TEST-not-loaded.safetensors', '--help'],
                                capture_output=True, text=True, encoding='utf-8', errors='replace',
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
