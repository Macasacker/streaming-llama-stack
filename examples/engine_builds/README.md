# Building native engines

`./bin/slstack install --all` builds this fork's two engines and the chat UI by
running `build.sh`: the llama fork with cmake (`GGML_CUDA=ON`, `sm_120a`), producing
`llama-server`, `llama-export-lora`, `libslstack-weight-fetcher.so` and
`llama-dsv41-prefill`; the ds4 engine with `make cuda`, producing `ds4-server` and
`ds4-decode.so`; and the WebUI with npm and vite. `install --all --check` checks the
outputs without rebuilding. Python 3.12, CMake and Ninja, a CUDA toolkit with `nvcc`,
liburing and Node.js are the prerequisites.

`RESULTS.json` here is the inherited record from the deleted Windows and macOS
engines: durations, binary identities and resource usage measured before the port. The
Linux builds have no receipt of their own yet; `setup-backend` writes one per backend
under `runtime_root/state/backends/`. Model-output comparisons are described in
[engine parity](../engine_parity/README.md).
