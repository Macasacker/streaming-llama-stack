# Engine source audit

`scripts/audit_backend_sources.py` checks the two retained engines for provenance
records, licence notices, the source inputs a build needs, any checked-in
`PROVENANCE.json` manifest hashes, and whether those files are tracked. It builds
nothing.

```sh
python -B scripts/audit_backend_sources.py
```

It exits non-zero on a missing input, a manifest whose recorded hash differs, or a
retained file Git does not track.

## Engine boundaries

| Engine | Interface | Source |
| --- | --- | --- |
| `ds4` | Native state broker and decode kernels | `engines/ds4`, CUDA with io_uring batch reads. See its `PROVENANCE.md` and `THIRD_PARTY_NOTICES.md`. |
| `streaming-llama` | Streaming prefill producer, `llama-server`, `mtmd` | `engines/streaming-llama/prefill_llama.cpp`, a llama.cpp fork at a pinned upstream, plus vendored FlashInfer sm120 kernels. |

External requirements the audit reports: CUDA 13.4 with `nvcc` for `sm_120` and
liburing for `ds4`; CUDA 13.4, a C++ toolchain and OpenSSL for the llama fork.
GPUDirect Storage is not required, because the batch reads go through io_uring. The
producer needs a qualified GGUF on disk before it can run at all.

HTTP compatibility alone does not provide KV injection. Model architecture, tensor
layout, quantisation and state format must match the selected backend.

`slstack decode --config local.json --check` checks the configured decoder's model
and executable paths, for `ds4` or `llama-server`. Set `decode.backend`,
`decode.model_path`, its bind address and backend-specific options in that file;
omit `--check` to run. A successful path check is not a model-load qualification.

## What an audit does not prove

A source audit does not establish a native build, model compatibility, or GPU output.
The build gate is `setup-backend` ([backend setup](BACKEND_SETUP.md)), and the only
gate that would establish output quality is the qualification run in chunk 8.1 of
[the port plan](linux-port-plan.md), which needs a model file in the accepted type
set. No such file exists on this box.

## DSV4.1 measurement context

These are inherited conditions for the README performance section, not measurements of
this fork. They were taken on the upstream Windows build; prefill and decode are
reported separately there.

The original model is `DeepSeek-V4.1-Flash-1017-Q4-v41.gguf`, 518,596,067,328 bytes,
SHA-256 `2cc6bc254f22a0238020e61e931c0cac865218aada1eed9c0036d9f4031c6e5f`. The
hardware was one 32 GB RTX 5090 with 32 GB DDR5 and a 25 GB/s RAID-0 NVMe array, with
explicit host-memory budgets. The single-machine route needs no second machine.

The rates for Kimi, GLM, MiniMax, Step and MiMo in
[the historical table](PERFORMANCE.md#other-models) were RTX prefill against
macOS decode, and neither half runs here any more. [Supported models](SUPPORTED_MODELS.md)
states what this fork serves.

## Publication checks

Vendored engine snapshots share the clean stack's Git history; development repository
histories are not imported with the files. Front Eagle remains the required author,
committer and tagger for the stack's commits. Original third-party copyright and
licence attribution stays intact.

`scripts/release_check.py` scans tracked and untracked-but-not-ignored blobs, commit
messages and, with `--history`, every reachable tree: private network literals,
credential patterns, machine paths, and the owner's name. Pattern scans cannot prove
an arbitrary unknown string is not a password.
