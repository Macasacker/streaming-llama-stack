# Negotiated exact-cone state

DSV4.1 ULTRA defaults select `yoco_keep=2541`. Explicit full prefill (`0`) and
approximate YOCO140 (`140`) stay supported. The native broker path, which hands
producer state to the ds4 server on the same machine, negotiates a state version with
it.

| Keep | Negotiated state version | Meaning |
| --- | --- | --- |
| 0 | 0 | Full prefill; a peer that omits both fields is read as this |
| 140 | 1 | Approximate bounded replay, explicit opt-in |
| 2541 | 2 | Exact dependency cone |

Version 2 reuses the version 1 mode3 export/import path. It does not truncate token
history or invent a second payload layout. Layers 0-20 publish every compressed and
index row. The padded cone preserves every final 128-row raw window, the complete
DSpark rings and the final logits. The canonical native payload, per-owner hashes,
token and logit digests, delta and rewind checks apply unchanged. Local snapshots keep
their separate local-state identity.

## Where it is enforced

`slstack/yoco.py` maps keep to version and refuses a pair that does not match.
`slstack/native_broker.py` requires the request's `yoco_keep` and `yoco_state_version`
to equal its own configuration, passes `--yoco-keep` and
`--yoco-remote-state-version` to the producer, and refuses a producer whose readiness
reports a different pair. Crossed pairs (`140/2`, `2541/1`), a missing cone provenance,
an unknown version and a malformed identity all fail closed, with no fallback to
approximate140.

`scripts/start-dsv41-single.sh --keep 2541|140|0` and
`examples/run_preset.py --single ultra-cone|ultra-yoco` are the launch surface.
`--approximate-yoco` selects keep 140.

CPU checks in the suite cover the request and header validation, the shared prefill
arguments between the single-engine and native-export shapes, and the launcher
defaults: `tests/test_shared_prefill_profile.py`, `tests/test_prefill_plan.py` and
`tests/test_dsv41_single_launch.py`. The image harness requires identical full and
cone logit hashes unless approximate mode is selected.

## What is not claimed here

The paired RTX-to-Mac proof was the only live measurement of this protocol, and it
went with the macOS half. Its receipts (`examples/dual-machine/dsv41-ultra-cone/proof/`)
are deleted, so the 105,054-token WebUI interval, the mode3 byte count, the Mac-side
follow-up rates and the state-parity hashes are no longer recorded anywhere in this
repository. The protocol and its CPU checks remain; the numbers do not.

`tests/cone_state_gate.py` is the GPU gate that compares local mode1 state with
negotiated mode3/v2 state on one machine. It needs a qualified model file and the
owner's exclusive GPU window, so it has not run since the port began.
