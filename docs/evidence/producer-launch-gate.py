"""Chunk 5.2's launch gate: run the real producer through the vendored Producer on Linux.

No model. The throwaway GGUF, receipt and four LE32 tokens take the CLI through its front
half, so this proves the launch chain rather than inference: the child is contained before
it execs, admission releases it, the three transport pipes carry it, its stderr drains into
the diagnostic buffer, its exit code is read, and the retirement certificate says the owned
tree is empty. Expected outcome on this box while another process holds the GPU: the abort
line `Selected GPU free bytes do not exceed the explicit reserve`, exit code 1, and
`launchers still blocked: 0`.

Run docs/linux-port-commands.md's cuda-headroom-probe in the same session: reaching that
refusal costs a CUDA context, and without one the producer aborts inside ggml CUDA init.
"""
import hashlib
import json
import os
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # this file sits two levels under the repo
sys.path.insert(0, str(ROOT))

from slstack.native_broker import vendor
vendor()
from studio.dsv41_session_process import Producer
from slstack.producer_runner import _Native

root = ROOT / 'build' / 'smoke'
out = root / 'out'
out.mkdir(parents=True, exist_ok=True)

model = root / 'model.gguf'
model.write_bytes(bytes(1 << 20))
digest = hashlib.sha256(model.read_bytes()).hexdigest()
st = os.stat(model)
receipt = root / 'copy-receipt.json'
receipt.write_text(json.dumps({'receipts': {'model.gguf': {
    'sha256': digest,
    'final_identity': {'dev': st.st_dev, 'ino': st.st_ino, 'size': st.st_size, 'mtime_ns': st.st_mtime_ns}}}}))
tokens = root / 'tokens.le32'
tokens.write_bytes(b''.join(struct.pack('<I', t) for t in (1, 2, 3, 4)))

binary = ROOT / 'build' / 'llama' / 'bin' / 'llama-dsv41-prefill'
build_sha = hashlib.sha256(binary.read_bytes()).hexdigest()
directory = root / 'owned'
directory.mkdir(exist_ok=True)
session = os.urandom(16).hex()

argv = [str(binary), '--model', str(model), '--copy-receipt', str(receipt), '--tokens-le32', str(tokens),
        '--cuda-device', '0', '--context', '8192', '--build-sha', build_sha,
        '--provider-device-bytes', str(1 << 30), '--provider-host-bytes', str(1 << 30),
        '--cache-bytes', str(4 << 30), '--graph-bytes', str(256 << 20),
        '--job-device-bytes', str(4 << 30), '--job-host-bytes', str(16 << 30),
        '--gpu-reserve-bytes', str(1 << 30),
        '--carry', str(out / 'carry.bin'), '--carry-file-bytes', str(4 << 30), '--carry-host-bytes', str(4 << 30),
        '--carry-outstanding-bytes', '0', '--output-outstanding-bytes', '0',
        '--payload', str(out / 'payload.bin'), '--receipt', str(out / 'receipt.json')]
env = dict(os.environ)   # CUDA_VISIBLE_DEVICES stays as the controller has it

native = _Native()
print('runner memory=%d GiB commit=%d/%d GiB' % (native.memory() >> 30, native.commit()[0] >> 30, native.commit()[1] >> 30))
from contextlib import ExitStack
with ExitStack() as leases:
    leases.enter_context(native.directory(directory))
    for held in (binary, model, receipt, tokens):
        leases.enter_context(native.read_lease(held))
    try:
        producer = Producer(session, argv, env, directory, host_bytes=1 << 30,
                            physical_floor=512 << 20, operation_seconds=120, idle_seconds=120,
                            native=native, native_mode=1, model_sha256=digest)
        print('UNEXPECTED: producer reached readiness')
    except BaseException as failure:
        print('producer refused: %s' % str(failure)[-700:])
        print('owned_cleanup=%s' % json.dumps(getattr(failure, 'owned_cleanup', None), default=str))
        print('owned_diagnostic tail: %s' % bytes(getattr(failure, 'owned_diagnostic', b''))[-1400:].decode(errors='replace'))
        process = getattr(failure, 'owned_process', None)
        job = getattr(process, 'job', None)
        if job is not None:
            print('job active after refusal=%d exit=%s scope=%s' % (job.active(), job.poll(), job.scope))
still = [p.name for p in Path('/proc').iterdir()
         if p.name.isdigit() and p.name != str(os.getpid())
         and b'producer_launch.py' in (p / 'cmdline').read_bytes()]
print('launchers still blocked: %d %s' % (len(still), still))
