# ds4 engine provenance (Linux)

Source lineage, inherited from the Windows engine this fork was ported from:

- [antirez/ds4 at `a0510c2ac2bec2cbffe730fa728fb7d3880f93ec`](https://github.com/antirez/ds4/commit/a0510c2ac2bec2cbffe730fa728fb7d3880f93ec), the merge-base between the exported Windows source ancestry and upstream ds4, MIT.
- [mishal-harbi/ds4-win-NovaStar at `8e3cfa00e60c9ba45acaf0c09e47054ca236521c`](https://github.com/mishal-harbi/ds4-win-NovaStar/commit/8e3cfa00e60c9ba45acaf0c09e47054ca236521c), the merge-base of the original NovaStar-derived checkout and its public branch.
- Selective V4.1 contributions from [IngeniousIdiocy/ds4-v41-m3ultra](https://github.com/IngeniousIdiocy/ds4-v41-m3ultra). The retained graph does not establish a direct V4.1 branch point. The historical `IngeniousIdiocy/stack-backend` repository could not be resolved to a verifiable upstream base.

The upstream licences, embedded-library attribution and third-party notices are retained in
`THIRD_PARTY_NOTICES.md`.

Changes made in this fork, all Linux-specific:

- Batched weight reads go through io_uring (`ds4_batch_io.cpp`) instead of DirectStorage. GPU
  destinations are filled by reading into a host mirror and then `cudaMemcpyAsync` on an upload
  stream with a `cudaEvent` for ordering; there is no D3D12 fence anywhere.
- Memory admission reads `/proc/meminfo`, `/proc/self/status`, and the cgroup v2
  `memory.current`/`memory.max` files (`ds4_host_memory.c`) instead of Windows job-object limits.
- Engram row reads use io_uring (`ds4_engram_iouring.c`).
- CUDA compute capability 120 (RTX 5090). Windows, macOS/Metal/MLX, and ROCm paths are deleted.

The port is inherited from the Windows engine's source, not from its compiled artifacts. The
packed FP4 and Q4_K half kernels came over as files; `cuda/mmq/mmq.cuh` is the Windows copy, which
keeps the K2304 row-stride guard so a partial K row never loads the next expert's codes or scales.
