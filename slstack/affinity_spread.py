"""What a B5 routing calibration is bound to, and whether it matches this model.

`presets/dsv41-affinity-spreads.txt` starts with

    DS4AFFINITY1 <model file bytes> <experts> <layers> <header hash>

and the engine refuses any nonzero expert-affinity bias unless that line matches the model it
loaded (`cuda_affinity_model` in `engines/ds4/ds4_cuda.cu`: file size, header FNV-1a, expert and
layer counts). Refusing at the token boundary is the bad failure - a session dies mid-decode, after
its prefill and with its VRAM held - and it happens whenever a launcher offers a calibration
measured on another file. The size and geometry check below is what the launcher can do before
starting anything; the header hash stays the engine's, so it is verified once, by the code that
also uses the numbers.
"""
from __future__ import annotations
import re
from pathlib import Path

HEADER = re.compile(r'^DS4AFFINITY1 (\d+) (\d+) (\d+) ([0-9a-f]{16})$')
# DeepSeek V4.1-Flash: 384 routed experts, 40 layers. The engine passes the same two constants to
# ds4_gpu_cache_affinity_prepare_model, so a spread naming anything else is not for this family.
DSV41_EXPERTS = 384
DSV41_LAYERS = 40


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
    """True only when the spread names this model file's bytes and this family's geometry.

    A missing model path, a missing spread, or a spread with no readable header is False: nothing
    unverified is offered, and an unbiased decode is the honest fallback."""
    bound = declared(spread)
    if not bound or not model:
        return False
    path = Path(model)
    if not path.is_file():
        return False
    size, experts, layers, _ = bound
    return size == path.stat().st_size and experts == DSV41_EXPERTS and layers == DSV41_LAYERS
