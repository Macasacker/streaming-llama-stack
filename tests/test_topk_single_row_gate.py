"""Build and run the on-device top-k gate (tests/dsv41_topk_single_row.cpp).

This is the check that proves the coverage rule equals what the CUDA backend actually dispatches —
the executor refuses a layer when the two disagree, and the rule is now shared
(ggml/include/ggml-dsv41-topk-plan.h). It needs the card, so it is opt-in: the rest of the suite
runs `tests/test_topk_coverage_gate.py`, which proves the same rule without a device.

    SLSTACK_GPU_GATE=1 env PYTHONPATH=.:tests python3 -B -m pytest tests/test_topk_single_row_gate.py -q

The `LLAMA_DSV4_TOPK_CUB_1ROW=1` A/B control is deliberately not run here: on this card
DeviceTopK's temporary storage for a 131072-column row trips
`ggml_cuda_capturing_dispatch` in `ggml-cuda.cu:1121`, which is that control path's own limit and
not the coverage rule's. The rule is asserted for both configurations that do run.
"""
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'tests' / 'dsv41_topk_single_row.cpp'
INCLUDE = ROOT / 'engines' / 'streaming-llama' / 'prefill_llama.cpp' / 'ggml' / 'include'
LIBS = ROOT / 'build' / 'llama' / 'ggml' / 'src'
CUDA_LIBS = LIBS / 'ggml-cuda'


def cuda_lib_dir():
    nvcc = shutil.which('nvcc')
    return Path(nvcc).parent.parent / 'lib64' if nvcc else Path('/usr/local/cuda/lib64')


def gpus_present():
    try:
        return subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True, timeout=20).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


class TopkSingleRowGateTests(unittest.TestCase):
    def build(self):
        if not os.environ.get('SLSTACK_GPU_GATE'):
            self.skipTest('this gate runs on the card; set SLSTACK_GPU_GATE=1 (docs/SESSION_FAILURE_PLAN.md)')
        if not gpus_present():
            self.skipTest('no CUDA device on this box')
        if not (LIBS / 'libggml.a').exists():
            self.skipTest('build ./build.sh first: the gate links against the built ggml')
        compiler = os.environ.get('SLSTACK_CXX') or shutil.which('c++') or shutil.which('g++') or shutil.which('clang++')
        if not compiler:
            self.skipTest('A C++ compiler is required for the on-device top-k gate')
        executable = Path(tempfile.mkdtemp()) / 'dsv41_topk_single_row'
        built = subprocess.run([compiler, '-std=c++17', '-O1', str(SOURCE), '-I', str(INCLUDE),
                                '-L', str(LIBS), '-L', str(CUDA_LIBS), '-L', str(cuda_lib_dir()),
                                '-Wl,--start-group', '-l:libggml.a', '-l:libggml-cuda.a', '-l:libggml-base.a',
                                '-l:libggml-cpu.a', '-Wl,--end-group',
                                '-lcudart', '-lcublas', '-lcublasLt', '-lnvrtc', '-lcuda', '-lgomp', '-lcudadevrt',
                                '-o', str(executable)],
                               capture_output=True, text=True, timeout=600)
        self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
        return executable

    def run_gate(self, executable, env):
        result = subprocess.run([str(executable)], capture_output=True, text=True, timeout=300,
                                env={**os.environ, **env})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('ALL PASS', result.stdout)
        self.assertNotIn('FAIL', result.stdout)
        return result.stdout

    def test_the_coverage_rule_equals_the_dispatched_kernels(self):
        out = self.run_gate(self.build(), {'LLAMA_DSV41_TOPK_RADIX': '1'})
        # The shape that aborted pi's session: one decode row over more index columns than the
        # single-kernel bitonic fits. The rule must count the kernels, not answer zero.
        self.assertIn('ncols=48645 k=512 rows=1: coverage rule == dispatched kernels dispatched=2 rule=2', out)

    def test_the_rule_still_holds_with_the_radix_kernel_off(self):
        self.run_gate(self.build(), {})


if __name__ == '__main__':unittest.main()
