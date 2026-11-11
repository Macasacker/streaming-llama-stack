"""Compile and run the top-k radix dispatch-count gate (tests/dsv41_topk_coverage.cpp).

The gate needs no CUDA: the dispatch rule is pure arithmetic in ggml-dsv41-topk-plan.h, so this is
the one engine-side self-check that can be regression-tested on a box whose card is busy. The two
runs are the row rule the backend can be configured with — deterministic (default) and the
LLAMA_DSV4_TOPK_CUB_1ROW control."""
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'tests' / 'dsv41_topk_coverage.cpp'
ENGINE = ROOT / 'engines' / 'streaming-llama' / 'prefill_llama.cpp'
INCLUDE = ENGINE / 'ggml' / 'include'
BACKEND = ENGINE / 'ggml' / 'src' / 'ggml-cuda' / 'top-k.cu'
EXECUTOR = ENGINE / 'src' / 'llama-dsv41-executor.cpp'


class TopkCoverageGateTests(unittest.TestCase):
    def compile(self):
        compiler = os.environ.get('SLSTACK_CXX') or shutil.which('c++') or shutil.which('g++') or shutil.which('clang++')
        if not compiler:
            self.skipTest('A C++ compiler is required for the top-k coverage gate')
        temporary = Path(tempfile.mkdtemp())
        executable = temporary / 'dsv41_topk_coverage'
        built = subprocess.run([compiler, '-std=c++17', '-Wall', '-O1', '-I', str(INCLUDE),
                                str(SOURCE), '-o', str(executable)],
                               capture_output=True, text=True, timeout=180)
        self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
        self.assertEqual(built.stderr.strip(), '', 'the gate compiles with warnings: ' + built.stderr)
        return executable

    def run_gate(self, executable, env):
        environment = {**os.environ, **env}
        result = subprocess.run([str(executable)], capture_output=True, text=True, timeout=60, env=environment)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('PASS all top-k radix coverage checks', result.stdout)
        self.assertNotIn('FAIL', result.stdout)
        return result.stdout

    def test_the_dispatch_count_holds_by_default(self):
        self.run_gate(self.compile(), {})

    def test_the_dispatch_count_holds_under_the_cub_single_row_control(self):
        out = self.run_gate(self.compile(), {'LLAMA_DSV4_TOPK_CUB_1ROW': '1'})
        self.assertIn('mode=CUB_1ROW control', out)

    def test_the_rule_is_read_from_one_header_by_both_sides(self):
        plan = (INCLUDE / 'ggml-dsv41-topk-plan.h').read_text(encoding='utf-8')
        self.assertIn('dsv41_topk_radix_dispatches', plan)
        backend = BACKEND.read_text(encoding='utf-8')
        executor = EXECUTOR.read_text(encoding='utf-8')
        self.assertIn('ggml-dsv41-topk-plan.h', backend)
        self.assertIn('ggml_backend_cuda_dsv41_topk_radix_dispatches', backend)
        self.assertIn('ggml_backend_cuda_dsv41_topk_radix_dispatches', executor)
        # The deleted rule must not come back: the executor must not re-derive the branch.
        self.assertNotIn('while(width>8192){width=(width+8191)/8192*512;++expected;}', executor)


if __name__ == '__main__':unittest.main()
