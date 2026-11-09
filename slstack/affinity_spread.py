"""What a B5 routing calibration is bound to, and whether it matches this model.

`runtime/qualification/*.txt` and `presets/dsv41-affinity-spreads.txt` start with

    DS4AFFINITY1 <model bytes> <experts> <layers> <header hash>

and the engine refuses any nonzero expert-affinity bias unless that line matches the model it
loaded: `cuda_affinity_model` in `engines/ds4/ds4_cuda.cu` compares size, the FNV-1a hash of the
first 64 KiB of the mapping XORed with that size, and the expert and layer counts. Refusing at the
token boundary is the bad failure - a session dies mid-decode, after its prefill and with its VRAM
held - so the launcher checks the same four numbers before starting anything.

`<model bytes>` is **not the file length**. `ds4_model.size` is set to the start of the disk-only
Engram region by `model_unmap_engram` (`engines/ds4/ds4.c`), because those rows are deliberately
outside the mapping, and that is the number recorded in the header. Measured 2026-11-09: the
producer calibrated this box's 365 713 686 528-byte file and wrote
`DS4AFFINITY1 162955640832 384 40 …` - the offset of `blk.1.engram_embd.weight`. A launcher that
compared `st_size` refused its own calibration as wrong-model.

The hash is mirrored here as well rather than left to the engine: reading 64 KiB is cheap, and a
file that shares a length or an Engram offset with another build is exactly the case that would
otherwise die at the first token.
"""
from __future__ import annotations
import re
import struct
from pathlib import Path

HEADER = re.compile(r'^DS4AFFINITY1 (\d+) (\d+) (\d+) ([0-9a-f]{16})$')
# DeepSeek V4.1-Flash: 384 routed experts, 40 layers. The engine passes the same two constants to
# ds4_gpu_cache_affinity_prepare_model, so a spread naming anything else is not for this family.
DSV41_EXPERTS = 384
DSV41_LAYERS = 40
# The tensor whose offset ds4 uses as the bound size, and the metadata that makes it apply.
ENGRAM_TENSOR = 'blk.1.engram_embd.weight'
ARCHITECTURE = 'deepseek41'
ENGRAM_ENCODING = 'e4m3_e8m0_32_row264'
# This fork's GGUF value types (UINT8=0 through FLOAT64=12, STRING=8, ARRAY=9) and the tensor
# block sizes, both mirroring engines/ds4/ds4.c. Only what the directory walk needs.
_STRING, _ARRAY = 8, 9
_SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_FNV_OFFSET = 14695981039346656037
_FNV_PRIME = 1099511628211
_MASK64 = (1 << 64) - 1
_HEADER_WINDOW = 1 << 16          # cuda_affinity_header_hash reads the first 65 536 bytes
# Enough of the head to hold metadata plus the tensor directory. A V4.1 GGUF's tokenizer table is
# ~5 MB (129 280 tokens and their merges), so the window is generous rather than exact; a file
# whose directory runs past it reads as unverifiable, which refuses a bias rather than trusting
# a partial parse.
_DIRECTORY_WINDOW = 32 << 20


class _Cursor:
    def __init__(self, buf):
        self.buf = buf
        self.i = 0

    def u32(self):
        v = struct.unpack_from('<I', self.buf, self.i)[0]
        self.i += 4
        return v

    def u64(self):
        v = struct.unpack_from('<Q', self.buf, self.i)[0]
        self.i += 8
        return v

    def string(self):
        n = self.u64()
        if self.i + n > len(self.buf):
            raise ValueError('GGUF header is truncated')
        value = self.buf[self.i:self.i + n].decode('utf-8', 'replace')
        self.i += n
        return value

    def value(self, t, depth=0):
        if depth > 8:
            raise ValueError('metadata array nesting is too deep')
        if t == _STRING:
            return self.string()
        if t == _ARRAY:
            elem = self.u32()
            for _ in range(self.u64()):
                self.value(elem, depth + 1)
            return None
        self.i += _SCALAR[t]
        return None


def binding(model):
    """(model bytes, header hash) as `ds4` sees this file, or None when it cannot be read.

    The bytes are the file length for a plain GGUF and the Engram region's start for a V4.1 file,
    which is what `model_unmap_engram` leaves `ds4_model.size` at.
    """
    path = Path(model)
    try:
        size = path.stat().st_size
        with path.open('rb') as handle:
            head = handle.read(_HEADER_WINDOW + _DIRECTORY_WINDOW)
    except OSError:
        return None
    if len(head) < 24 or head[:4] != b'GGUF':
        return None
    try:
        return _bind(path, head, size)
    except (ValueError, KeyError, struct.error):
        # A directory that runs past the read window, an unknown value type, or a truncated
        # header: unverifiable, so no bias is offered rather than a partially parsed one.
        return None


def _bind(path, head, size):
    version, n_tensors, n_kv = struct.unpack_from('<IQQ', head, 4)
    if version != 3:
        return None
    c = _Cursor(head)
    c.i = 24
    architecture = encoding = None
    alignment = 32
    for _ in range(n_kv):
        key = c.string()
        t = c.u32()
        if key == 'general.architecture':
            architecture = c.value(t)
        elif key == 'deepseek41.engram.encoding':
            encoding = c.value(t)
        elif key == 'general.alignment':
            alignment = c.u32()
        else:
            c.value(t)
    engram_rel = None
    for _ in range(n_tensors):
        name = c.string()
        dims = [c.u64() for _ in range(c.u32())]
        c.u32()                        # tensor type; only the offsets matter here
        rel = c.u64()
        if name == ENGRAM_TENSOR:
            engram_rel = rel
    data_start = (c.i + alignment - 1) // alignment * alignment
    bound = data_start + engram_rel if (architecture == ARCHITECTURE and encoding == ENGRAM_ENCODING
                                        and engram_rel is not None) else size
    if bound > size:
        return None
    with path.open('rb') as handle:
        prefix = handle.read(min(bound, _HEADER_WINDOW))
    digest = _FNV_OFFSET
    for byte in prefix:
        digest = ((digest ^ byte) * _FNV_PRIME) & _MASK64
    return bound, digest ^ bound


def declared(spread):
    """(model bytes, experts, layers, hash) from a spread file's header line, or None."""
    try:
        with Path(spread).open('r', encoding='utf-8') as handle:
            match = HEADER.match(handle.readline().strip())
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    if not match:
        return None
    return int(match.group(1)), int(match.group(2)), int(match.group(3)), match.group(4)


def matches_model(spread, model):
    """True only when the spread names this model's bound bytes, hash and this family's geometry.

    A missing model path, a missing spread, a file that is not a readable GGUF v3, or a spread with
    no header is False: nothing unverified is offered, and an unbiased decode is the honest
    fallback."""
    bound = declared(spread)
    if not bound or not model:
        return False
    seen = binding(model)
    if seen is None:
        return False
    size, experts, layers, digest = bound
    try:
        # The header writes the hash with %llx, matching cuda_affinity_header_hash's uint64.
        declared_digest = int(digest, 16)
    except ValueError:
        return False
    return size == seen[0] and declared_digest == seen[1] and experts == DSV41_EXPERTS \
        and layers == DSV41_LAYERS
