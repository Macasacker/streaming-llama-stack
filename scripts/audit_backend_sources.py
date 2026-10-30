"""Audit the retained Linux engine sources: provenance, licences, and file completeness.

The fork ships two engines. This script checks that each one carries its provenance record and
that the source inputs a build needs are present in the checkout. It does not build anything and
does not establish that a model runs - see docs/linux-port-plan.md for the gates that do.
"""
import hashlib
import json
import subprocess
from pathlib import Path


def normalized(path):
    return hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()


ENGINES = {
    'ds4': {
        'provenance': 'engines/ds4/PROVENANCE.md',
        'notices': ['engines/ds4/THIRD_PARTY_NOTICES.md'],
        'inputs': ['engines/ds4/Makefile', 'engines/ds4/ds4.c', 'engines/ds4/ds4_cuda.cu',
                   'engines/ds4/ds4_batch_io.cpp', 'engines/ds4/ds4_batch_io.h',
                   'engines/ds4/ds4_host_memory.c', 'engines/ds4/ds4_host_memory.h',
                   'engines/ds4/ds4_engram_iouring.c', 'engines/ds4/ds4_server.c',
                   'engines/ds4/cuda/mmq/mmq.cuh', 'engines/ds4/cuda/mmq/ds4_repack.cu'],
    },
    'streaming-llama': {
        'provenance': 'engines/streaming-llama/PROVENANCE.md',
        'inputs': ['engines/streaming-llama/prefill_llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu',
                   'engines/streaming-llama/prefill_llama.cpp/src/llama-weight-fetcher.cpp'],
    },
}


def manifest_hashes(root, engine):
    """Verify any checked-in PROVENANCE.json manifest for an engine directory."""
    manifest_path = root / 'engines' / engine / 'PROVENANCE.json'
    if not manifest_path.is_file():
        return []
    mismatches = []
    for row in json.loads(manifest_path.read_text(encoding='utf-8'))['files']:
        file = root / 'engines' / engine / row['path']
        if not file.is_file() or normalized(file) != row['export_sha256']:
            mismatches.append(row['path'])
    return mismatches


def audit(root=None):
    root = Path(root or Path(__file__).resolve().parents[1])
    rows = []
    for name, spec in ENGINES.items():
        missing = [p for p in [spec['provenance'], *spec.get('notices', []), *spec['inputs']]
                   if not (root / p).is_file()]
        rows.append(dict(engine=name, source_inputs_present=not missing, missing=missing,
                         qualification='Source audit only; does not establish a native build, '
                                       'model compatibility, or GPU output.'))
    hashes = {name: manifest_hashes(root, name) for name in ENGINES}
    untracked = []
    if (root / '.git').exists():
        tracked = set(subprocess.check_output(['git', '-C', str(root), 'ls-files', '--cached', '-z'])
                      .decode().split('\0'))
        for name, spec in ENGINES.items():
            untracked.extend(p for p in [spec['provenance'], *spec.get('notices', []), *spec['inputs']]
                             if p not in tracked)
    complete = not (any(row['missing'] for row in rows) or any(hashes.values()) or untracked)
    return dict(engines=rows, manifest_hash_mismatches=hashes, untracked_retained_files=untracked,
                all_retained_components_complete=complete,
                external_requirements={'ds4': 'CUDA 13.4 with nvcc for sm_120, liburing; '
                                              'GPUDirect Storage is not required (io_uring batch reads).',
                                       'streaming-llama': 'CUDA 13.4, a C++ toolchain, and OpenSSL; '
                                                          'the producer needs a qualified GGUF on disk to run.'})


if __name__ == '__main__':
    result = audit()
    print(json.dumps(result, indent=2))
    raise SystemExit(not result['all_retained_components_complete'])
