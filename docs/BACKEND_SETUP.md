# Backend installation

This fork builds two engines. `ds4` is the native CUDA decode engine in
`engines/ds4`; `llama-server` is the llama.cpp fork in
`engines/streaming-llama/prefill_llama.cpp`. Build inputs, compiler output and
receipts live under `runtime_root`; model weights and host settings stay in your own
config. Installation needs no Studio checkout and no virtual environment of yours.

Setup is Linux-only: `slstack/backend_setup.py` refuses any other platform rather
than building something it cannot verify.

```sh
./bin/slstack --config my.local.json setup-backend --backend ds4
./bin/slstack --config my.local.json setup-backend --backend llama-server
./bin/slstack --config my.local.json setup-backend --backend all --check
```

`--backend all` covers both. `--jobs` is 1 or 2 compiler jobs, each running under the
stack's host-memory guardian with `backend_build_budget_gib` (default 3 GiB) and at
least 4 GiB of available system memory; a lower budget can stop a linker before it
finishes. Setup requires 16 GiB free on the runtime volume for inputs, outputs and
headroom.

| Backend | What setup builds |
| --- | --- |
| `ds4` | `make -jN cuda CUDA_ARCH=sm_<arch>` over the staged engine tree, producing `ds4-server`. The architecture comes from `nvidia-smi --query-gpu=compute_cap`, falling back to `sm_120a` or `DS4_CUDA_ARCH`. |
| `llama-server` | cmake over the staged llama fork with `-DGGML_CUDA=ON`, `-DBUILD_SHARED_LIBS=OFF`, `-DLLAMA_CURL=OFF`, `-DLLAMA_OPENSSL=ON`, tests and examples off, building the `llama-server` target into `runtime_root/build/llama-<revision>/bin/`. |

The producer and the chat UI are not `setup-backend` targets. `./build.sh
[config] [all|llama|ds4|webui]` builds them: the WebUI with npm and vite, the ds4
engine with `make cuda`, and the llama tree with cmake including
`llama-dsv41-prefill` and `libslstack-weight-fetcher.so`. `./bin/slstack install
--all` runs `build.sh` too, alongside the media environments, and hashes the model
file once.

An OpenAI HTTP backend performs its own prefill. `llama-server` is that backend; the
`ds4` route is the native state broker and does the streaming prefill in the producer.

```sh
./bin/slstack --config my.local.json setup-backend --backend ds4 --build-only
```

`--build-only` compiles and leaves verification pending. A normal setup run also
executes `<binary> --help` as a CLI check before marking the backend ready.
`--fetch-only` downloads pinned source archives into `--cache-dir` and builds
nothing. `--offline` needs a `--wheelhouse` for the Python build tools
(cmake 4.4.4, setuptools 84.2.1, wheel 0.48.0) and a `--cache-dir` for sources.
Downloads are verified before extraction, and a divergent cache or build snapshot is
reported and preserved rather than overwritten.

Backend receipts at `runtime_root/state/backends/<name>.json` record the source
revision, the binary hashes and fingerprints of the retained stack sources the build
used. `--check` refuses a changed or unverified artifact, and `slstack decode`
refuses to launch one.

Installation and CLI checks establish that an engine is built and runs its own help
text. They do not qualify a model, an output, or a speed. Nothing in this fork has
loaded weights yet; see [the port plan](linux-port-plan.md) for the gates that would.

See [the source audit](BACKEND_SOURCE_AUDIT.md) for provenance and what a source
audit does and does not prove.
