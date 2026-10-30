CC ?= cc
NVCC ?= $(shell command -v nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc)
CUDA_HOME ?= $(shell if [ -x "$(NVCC)" ]; then dirname "$$(dirname "$(NVCC)")"; else printf '%s' /usr/local/cuda; fi)

NATIVE_CPU_FLAG ?= -march=native
DEBUG_FLAGS ?= -g
# -ffast-math is deliberate: the engine's reduction order is qualified against
# hash-pinned fixtures, not against IEEE-exact reference math.
CFLAGS ?= -O3 -ffast-math $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) -Wall -Wextra -std=c99 \
	-D_GNU_SOURCE -fno-finite-math-only
QUALITY_CFLAGS ?= -O3 $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) -Wall -Wextra -std=c11

LDLIBS ?= -lm -pthread -luring
DS4_TEST_MODEL ?= ds4flash.gguf
DS4_TEST_MTP ?= gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf
DS4_DSPARK_MODEL ?= $(DS4_TEST_MODEL)
DS4_DSPARK_SUPPORT ?= gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf

# One platform: NVIDIA RTX 50-series, compute capability 12.0. sm_120a is the
# arch nvcc needs for the FP4 tensor-core paths (mma.sync f4 with aarch).
CUDA_ARCH ?=
ifneq ($(strip $(CUDA_ARCH)),)
ifneq ($(filter sm_120 sm_120a,$(strip $(CUDA_ARCH))),)
NVCC_ARCH_FLAGS := -gencode arch=compute_120a,code=sm_120a -DDS4_CUDA_HAVE_MXF4=1
else
NVCC_ARCH_FLAGS := -arch=$(CUDA_ARCH)
endif
endif
NVCCFLAGS ?= -O3 -g -lineinfo --use_fast_math $(NVCC_ARCH_FLAGS) \
	-Xcompiler $(NATIVE_CPU_FLAG) -Xcompiler -pthread
CUDA_LDLIBS ?= -lm -Xcompiler -pthread -L$(CUDA_HOME)/lib -L$(CUDA_HOME)/lib64 -lcudart -lcublas -lcuda -luring
DS4_LINK ?= $(NVCC) $(NVCCFLAGS)
DS4_LINK_LIBS ?= $(CUDA_LDLIBS)

# Vendored llama.cpp mmq prefill tier (cuda/mmq/, see cuda/mmq/VENDOR.md).
MMQ_INCLUDES := -Icuda/mmq
MMQ_OBJS := cuda/mmq/ds4_ggml_stubs.o cuda/mmq/ds4_mmq.o cuda/mmq/ds4_mmq_d2r.o \
	cuda/mmq/quantize.o cuda/mmq/mmid.o cuda/mmq/mmvq.o cuda/mmq/ds4_repack.o \
	cuda/mmq/ds4_q4k_half.o
HOST_IO_OBJS = ds4_batch_io.o ds4_host_memory.o
CORE_OBJS = ds4.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_cuda.o ds4_layer_pack.o ds4_engram.o \
	ds4_engram_iouring.o ds4_image.o ds4_prompt_prefix.o $(HOST_IO_OBJS) $(MMQ_OBJS)
CPU_CORE_OBJS = ds4_cpu.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_layer_pack.o ds4_engram.o \
	ds4_engram_iouring.o ds4_image.o ds4_prompt_prefix.o ds4_host_memory_cpu.o
PIC_FLAGS := -fPIC

.PHONY: all help clean test cpu cuda cuda-generic cuda-decode cuda-regression \
	test-mxfp4-cuda test-kv-payload-roundtrip test-cpu-fp4 dspark-acceptance \
	dspark-verify-depth mtp-verify-depth

all: help

help:
	@echo "DS4 build targets (Linux, CUDA only):"
	@echo "  make cuda CUDA_ARCH=sm_120     Build CUDA ./ds4, ./ds4-server, ./ds4-bench, ./ds4-eval, ./ds4-agent"
	@echo "  make cuda-generic              Build CUDA for a generic local CUDA GPU"
	@echo "  make cuda-decode CUDA_ARCH=sm_120  Build ./ds4-decode.so for the producer's decode engine"
	@echo "  make cpu                       Build CPU-only ./ds4, ./ds4-server, ./ds4-bench, ./ds4-eval, ./ds4-agent"
	@echo "  make test                      Build and run the tests that need no model file"
	@echo "  make test-mxfp4-cuda           MXFP4 exactness test (needs tests/test_mxfp4_cuda.cu)"
	@echo "  make test-cpu-fp4              CPU mirror of the packed FP4 MoE path"
	@echo "  make dspark-verify-depth       DSpark speculative verification smoke if support GGUF is present"
	@echo "  make mtp-verify-depth          Legacy MTP speculative verification smoke if MTP GGUF is present"
	@echo "  make clean                     Remove build outputs"

cuda:
	@if [ -z "$(strip $(CUDA_ARCH))" ]; then \
		echo "error: specify CUDA_ARCH, for example: make cuda CUDA_ARCH=sm_120"; \
		echo "       or use make cuda-generic"; \
		exit 2; \
	fi
	$(MAKE) -B ds4 ds4-server ds4-bench ds4-eval ds4-agent CUDA_ARCH="$(CUDA_ARCH)"

cuda-generic:
	$(MAKE) -B ds4 ds4-server ds4-bench ds4-eval ds4-agent CUDA_ARCH=native

ds4: ds4_cli.o ds4_help.o linenoise.o ds4_gpu_args.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

ds4-server: ds4_server.o ds4_peer.o ds4_help.o ds4_kvstore.o rax.o ds4_gpu_args.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

ds4-bench: ds4_bench.o ds4_help.o ds4_gpu_args.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

ds4-eval: ds4_eval.o ds4_eval_cases.o ds4_help.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

ds4-agent: ds4_agent.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o ds4_gpu_args.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

# The producer loads this beside itself instead of ds4-decode.dll: it binds the
# ds4x_* entry points in ds4_decode_adapter.inc with dlopen.
ds4-decode.so: ds4.o ds4_cuda.o ds4_lora.o ds4_image.o $(MMQ_OBJS) $(HOST_IO_OBJS) ds4_layer_pack.o ds4_ssd.o ds4_distributed.o ds4_tp.o
	$(NVCC) $(NVCCFLAGS) $(PIC_FLAGS) -shared -o $@ $^ $(CUDA_LDLIBS)

cuda-decode:
	@if [ -z "$(strip $(CUDA_ARCH))" ]; then \
		echo "error: specify CUDA_ARCH, for example: make cuda-decode CUDA_ARCH=sm_120"; \
		exit 2; \
	fi
	$(MAKE) -B ds4-decode.so CUDA_ARCH="$(CUDA_ARCH)"

gguf-tools/quality-testing/score_official.o: gguf-tools/quality-testing/score_official.c ds4.h
	$(CC) $(filter-out -ffast-math,$(QUALITY_CFLAGS)) -I. -c -o $@ $<

gguf-tools/quality-testing/score_official: gguf-tools/quality-testing/score_official.o $(CORE_OBJS) rax.o ds4_gpu_args.o
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

cpu: ds4_cli_cpu.o ds4_server_cpu.o ds4_bench_cpu.o ds4_eval_cpu.o ds4_agent_cpu.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o rax.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS)
	$(CC) $(CFLAGS) -o ds4 ds4_cli_cpu.o ds4_help.o linenoise.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-server ds4_server_cpu.o ds4_peer.o ds4_help.o ds4_kvstore.o rax.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-bench ds4_bench_cpu.o ds4_help.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-eval ds4_eval_cpu.o ds4_eval_cases.o ds4_help.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-agent ds4_agent_cpu.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)

cuda-regression: tests/cuda_long_context_smoke
	./tests/cuda_long_context_smoke

ds4.o: ds4.c ds4.h ds4_ssd.h ds4_distributed.h ds4_gpu.h
	$(CC) $(CFLAGS) -c -o $@ ds4.c

ds4_ssd.o: ds4_ssd.c ds4_ssd.h
	$(CC) $(CFLAGS) -c -o $@ ds4_ssd.c

ds4_cli.o: ds4_cli.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h linenoise.h
	$(CC) $(CFLAGS) -c -o $@ ds4_cli.c

ds4_distributed.o: ds4_distributed.c ds4_distributed.h ds4.h ds4_ssd.h
	$(CC) $(CFLAGS) -c -o $@ ds4_distributed.c

ds4_tp.o: ds4_tp.c ds4_tp.h ds4.h ds4_ssd.h
	$(CC) $(CFLAGS) -c -o $@ ds4_tp.c

ds4_help.o: ds4_help.c ds4_help.h
	$(CC) $(CFLAGS) -c -o $@ ds4_help.c

ds4_gpu_args.o: ds4_gpu_args.c ds4_gpu_args.h ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -c -o $@ ds4_gpu_args.c

ds4_server.o: ds4_server.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h ds4_kvstore.h rax.h
	$(CC) $(CFLAGS) -c -o $@ ds4_server.c

ds4_bench.o: ds4_bench.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h
	$(CC) $(CFLAGS) -c -o $@ ds4_bench.c

ds4_eval.o: ds4_eval.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h
	$(CC) $(CFLAGS) -c -o $@ ds4_eval.c

ds4_agent.o: ds4_agent.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h ds4_kvstore.h ds4_web.h linenoise.h
	$(CC) $(CFLAGS) -c -o $@ ds4_agent.c

ds4_web.o: ds4_web.c ds4_web.h
	$(CC) $(CFLAGS) -c -o $@ ds4_web.c

ds4_kvstore.o: ds4_kvstore.c ds4_kvstore.h ds4.h ds4_ssd.h
	$(CC) $(CFLAGS) -c -o $@ ds4_kvstore.c

# io_uring expert-read transport and the host-memory admission guard: host C++/C,
# linked into the CUDA binaries. Only ds4_batch_io.c needs the CUDA headers.
ds4_batch_io.o: ds4_batch_io.cpp ds4_batch_io.h
	$(CXX) $(PIC_FLAGS) -O3 -g -std=c++17 -Wall -Wextra -I$(CUDA_HOME)/include -c -o $@ $<

ds4_host_memory.o: ds4_host_memory.c ds4_host_memory.h
	$(CC) $(CFLAGS) $(PIC_FLAGS) -c -o $@ $<

ds4_engram.o: ds4_engram.c ds4_engram.h
	$(CC) $(CFLAGS) $(PIC_FLAGS) -c -o $@ $<

ds4_engram_iouring.o: ds4_engram_iouring.c ds4_engram.h
	$(CC) $(CFLAGS) $(PIC_FLAGS) -c -o $@ $<

ds4_peer.o: ds4_peer.c ds4_peer.h
	$(CC) $(CFLAGS) -c -o $@ ds4_peer.c

ds4_image.o: ds4_image.c ds4_image.h
	$(CC) $(CFLAGS) -c -o $@ ds4_image.c

ds4_lora.o: ds4_lora.cpp ds4_lora.h
	$(CXX) $(PIC_FLAGS) -O3 -g -std=c++17 -c -o $@ ds4_lora.cpp

ds4_test.o: tests/ds4_test.c ds4_server.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h ds4_kvstore.h rax.h
	$(CC) $(CFLAGS) -Wno-unused-function -c -o $@ tests/ds4_test.c

ds4_agent_test.o: tests/ds4_agent_test.c ds4_agent.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h ds4_kvstore.h ds4_web.h linenoise.h
	$(CC) $(CFLAGS) -Wno-unused-function -c -o $@ tests/ds4_agent_test.c

tests/cuda_long_context_smoke.o: tests/cuda_long_context_smoke.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/cuda_long_context_smoke.c

rax.o: rax.c rax.h rax_malloc.h
	$(CC) $(CFLAGS) -c -o $@ rax.c

linenoise.o: linenoise.c linenoise.h
	$(CC) $(CFLAGS) -c -o $@ linenoise.c

ds4_cpu.o: ds4.c ds4.h ds4_ssd.h ds4_distributed.h ds4_gpu.h
	$(CC) $(CFLAGS) -Wno-unused-function -DDS4_NO_GPU -c -o $@ ds4.c

ds4_host_memory_cpu.o: ds4_host_memory.c ds4_host_memory.h
	$(CC) $(CFLAGS) -c -o $@ ds4_host_memory.c

ds4_cli_cpu.o: ds4_cli.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h linenoise.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_cli.c

ds4_gpu_args_cpu.o: ds4_gpu_args.c ds4_gpu_args.h ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_gpu_args.c

ds4_server_cpu.o: ds4_server.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h ds4_kvstore.h rax.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_server.c

ds4_bench_cpu.o: ds4_bench.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_bench.c

ds4_eval_cpu.o: ds4_eval.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_eval.c

ds4_agent_cpu.o: ds4_agent.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h ds4_kvstore.h ds4_web.h linenoise.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_agent.c

ds4_cuda.o: ds4_cuda.cu ds4_gpu.h ds4_gpu_mgpu.h ds4_iq2_tables_cuda.inc cuda/mmq/ds4_mmq.h
	$(NVCC) $(NVCCFLAGS) -c -o $@ ds4_cuda.cu

# Vendored mmq pieces (see cuda/mmq/VENDOR.md).  ds4_mmq.cu transitively
# pulls in mmq.cuh which has heavy template instantiation -- each piece
# compiles in its own TU and links in.
cuda/mmq/ds4_ggml_stubs.o: cuda/mmq/ds4_ggml_stubs.cu cuda/mmq/ds4_ggml_stubs.h cuda/mmq/common.cuh
	$(NVCC) $(NVCCFLAGS) -std=c++17 $(MMQ_INCLUDES) -c -o $@ $<

cuda/mmq/ds4_mmq.o: cuda/mmq/ds4_mmq.cu cuda/mmq/ds4_mmq.h cuda/mmq/ds4_mmq_d2r.cuh cuda/mmq/mmq.cuh cuda/mmq/common.cuh cuda/mmq/ds4_ggml_stubs.h cuda/mmq/quantize.cuh cuda/mmq/mmid.cuh cuda/mmq/vecdotq.cuh cuda/mmq/mma.cuh cuda/mmq/ds4_mmq_fp4_packed.cuh
	$(NVCC) $(NVCCFLAGS) -std=c++17 $(MMQ_INCLUDES) -c -o $@ $<

cuda/mmq/ds4_mmq_d2r.o: cuda/mmq/ds4_mmq_d2r.cu cuda/mmq/ds4_mmq_d2r.cuh cuda/mmq/mmq.cuh cuda/mmq/common.cuh cuda/mmq/ds4_ggml_stubs.h cuda/mmq/vecdotq.cuh cuda/mmq/mma.cuh
	$(NVCC) $(NVCCFLAGS) -std=c++17 $(MMQ_INCLUDES) -c -o $@ $<

cuda/mmq/quantize.o: cuda/mmq/quantize.cu cuda/mmq/quantize.cuh cuda/mmq/common.cuh cuda/mmq/ds4_ggml_stubs.h cuda/mmq/mmq.cuh
	$(NVCC) $(NVCCFLAGS) -std=c++17 $(MMQ_INCLUDES) -c -o $@ $<

cuda/mmq/ds4_q4k_half.o: cuda/mmq/ds4_q4k_half.cu cuda/mmq/ds4_q4k_half_common.cuh cuda/mmq/common.cuh cuda/mmq/mma.cuh cuda/mmq/mmid.cuh
	$(NVCC) $(NVCCFLAGS) -std=c++17 $(MMQ_INCLUDES) -c -o $@ $<

cuda/mmq/mmid.o: cuda/mmq/mmid.cu cuda/mmq/mmid.cuh cuda/mmq/common.cuh cuda/mmq/ds4_ggml_stubs.h
	$(NVCC) $(NVCCFLAGS) -std=c++17 $(MMQ_INCLUDES) -c -o $@ $<

cuda/mmq/mmvq.o: cuda/mmq/mmvq.cu cuda/mmq/mmvq.cuh cuda/mmq/common.cuh cuda/mmq/ds4_ggml_stubs.h cuda/mmq/quantize.cuh cuda/mmq/vecdotq.cuh cuda/mmq/unary.cuh
	$(NVCC) $(NVCCFLAGS) -std=c++17 $(MMQ_INCLUDES) -c -o $@ $<

cuda/mmq/ds4_repack.o: cuda/mmq/ds4_repack.cu cuda/mmq/ds4_repack.h
	$(NVCC) $(NVCCFLAGS) -std=c++17 -c -o $@ $<

tests/cuda_long_context_smoke: tests/cuda_long_context_smoke.o ds4_cuda.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_layer_pack.o: tests/test_layer_pack.c ds4_layer_pack.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_layer_pack: tests/test_layer_pack.o ds4_layer_pack.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_gpu_args.o: tests/test_gpu_args.c ds4_gpu_args.h ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -I. -DDS4_NO_GPU -c -o $@ $<

tests/test_gpu_args: tests/test_gpu_args.o ds4_gpu_args_cpu.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

ds4_cpu_test_hooks.o: ds4.c ds4.h ds4_gpu.h ds4_gpu_mgpu.h ds4_layer_pack.h
	$(CC) $(CFLAGS) -Wno-unused-function -DDS4_NO_GPU -DDS4_TEST_HOOKS -c -o $@ ds4.c

tests/test_engine_mgpu_placement.o: tests/test_engine_mgpu_placement.c ds4.h ds4_gpu_mgpu.h ds4_layer_pack.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_engine_mgpu_placement: tests/test_engine_mgpu_placement.o ds4_cpu_test_hooks.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_layer_pack.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_kv_payload_roundtrip.o: tests/test_kv_payload_roundtrip.c ds4.h ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_kv_payload_roundtrip: tests/test_kv_payload_roundtrip.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-kv-payload-roundtrip: tests/test_kv_payload_roundtrip
	DS4_TEST_MODEL="$(DS4_TEST_MODEL)" ./tests/test_kv_payload_roundtrip --all-formats

tests/test_cpu_fp4: tests/test_cpu_fp4.cpp ds4_cpu_fp4.cpp ds4.h
	$(CXX) -O2 -Wall -Wextra -std=c++17 -o $@ tests/test_cpu_fp4.cpp ds4_cpu_fp4.cpp -lm
	./tests/test_cpu_fp4

test-cpu-fp4: tests/test_cpu_fp4

tests/test_quality_phase.o: tests/test_quality_phase.c ds4_quality_phase.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_quality_phase: tests/test_quality_phase.o ds4.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_layer_pack.o $(MMQ_OBJS) ds4_cuda.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

.PHONY: test-quality-phase
test-quality-phase: tests/test_quality_phase
	./tests/test_quality_phase

ifneq ($(wildcard tests/test_mxfp4_cuda.cu),)
tests/test_mxfp4_cuda: tests/test_mxfp4_cuda.cu $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -std=c++17 $(MMQ_INCLUDES) -o $@ $^ $(CUDA_LDLIBS)

test-mxfp4-cuda: tests/test_mxfp4_cuda
	./tests/test_mxfp4_cuda
else
test-mxfp4-cuda:
	@echo "test-mxfp4-cuda: skipped, tests/test_mxfp4_cuda.cu is not in the published tree"
endif

ds4_test: ds4_test.o ds4_help.o ds4_kvstore.o rax.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ ds4_test.o ds4_help.o ds4_kvstore.o rax.o $(CORE_OBJS) $(CUDA_LDLIBS)

ds4_agent_test: ds4_agent_test.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ ds4_agent_test.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o $(CORE_OBJS) $(CUDA_LDLIBS)

test: ds4-eval tests/test_layer_pack tests/test_gpu_args \
	tests/test_engine_mgpu_placement tests/test_quality_phase tests/test_cpu_fp4
	./ds4-eval --self-test-extractors
	./tests/test_layer_pack
	./tests/test_gpu_args
	./tests/test_engine_mgpu_placement
	./tests/test_gpu_args_cli.sh

dspark-acceptance: ds4
	DS4_DSPARK_MODEL="$(DS4_DSPARK_MODEL)" \
	DS4_DSPARK_SUPPORT="$(DS4_DSPARK_SUPPORT)" \
	sh tests/dspark_acceptance_fixture.sh

dspark-verify-depth: ds4_test
	@if [ ! -f "$(DS4_TEST_MODEL)" ]; then \
		echo "dspark-verify-depth: skipped, missing model $(DS4_TEST_MODEL)"; \
	elif [ ! -f "$(DS4_DSPARK_SUPPORT)" ]; then \
		echo "dspark-verify-depth: skipped, missing DSpark support $(DS4_DSPARK_SUPPORT)"; \
		echo "dspark-verify-depth: run ./download_model.sh ds4f-dspark or set DS4_DSPARK_SUPPORT=FILE"; \
	else \
		DS4_TEST_MODEL="$(DS4_TEST_MODEL)" DS4_TEST_DSPARK="$(DS4_DSPARK_SUPPORT)" ./ds4_test --dspark-verify-depth; \
	fi

mtp-verify-depth: ds4_test
	@if [ ! -f "$(DS4_TEST_MODEL)" ]; then \
		echo "mtp-verify-depth: skipped, missing model $(DS4_TEST_MODEL)"; \
	elif [ ! -f "$(DS4_TEST_MTP)" ]; then \
		echo "mtp-verify-depth: skipped, missing MTP support $(DS4_TEST_MTP)"; \
		echo "mtp-verify-depth: set DS4_TEST_MTP=FILE to a legacy support GGUF"; \
	else \
		DS4_TEST_MODEL="$(DS4_TEST_MODEL)" DS4_TEST_MTP="$(DS4_TEST_MTP)" ./ds4_test --mtp-verify-depth; \
	fi

q4k-dot-test: tests/test_q4k_dot.c
	$(CC) -O2 -Wall -Wextra -std=c99 -o tests/test_q4k_dot tests/test_q4k_dot.c -lm -pthread
	./tests/test_q4k_dot

mxfp4-dot-test: tests/test_mxfp4_dot.c
	$(CC) -O2 -Wall -Wextra -std=c99 -o tests/test_mxfp4_dot tests/test_mxfp4_dot.c -lm
	./tests/test_mxfp4_dot

clean:
	rm -f ds4 ds4-server ds4-bench ds4-eval ds4-agent ds4-decode.so ds4_cpu ds4_native \
		ds4_server_test ds4_test ds4_agent_test \
		gguf-tools/quality-testing/score_official gguf-tools/quality-testing/score_official.o \
		tests/test_q4k_dot tests/test_mxfp4_dot tests/test_mxfp4_cuda tests/test_cpu_fp4 \
		tests/test_kv_payload_roundtrip tests/test_quality_phase \
		tests/test_gpu_xdev tests/test_gpu_model_cache tests/test_gpu_lookup_cache_strict \
		tests/test_engine_mgpu_refusal tests/test_engine_mgpu_runtime tests/test_engine_correctness \
		tests/test_sampling tests/test_cuda_session_batch tests/test_cuda_mixed_batch \
		tests/cuda_long_context_smoke tests/cuda_long_context_smoke.o tests/*.o *.o \
		cuda/mmq/*.o
