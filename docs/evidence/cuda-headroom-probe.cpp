// Whether this box can create a CUDA context right now, and whether a change is
// to blame when it cannot. Strata's serve process holds ~30 GiB of the 32 GiB
// and the browser GPU process holds a few hundred MB more, so the margin is
// roughly 120 MB when it exists at all and it moves within a session. Two calls
// that need no allocation answer the question: cudaFree(0) initialises the
// context and cudaMemGetInfo reports what is left. When both return
// "out of memory" every CUDA gate here is unrun rather than failing, including
// the producer's no-model receipt smoke, which aborts inside
// ggml_backend_cuda_get_device_memory before its reserve refusal. Nothing in
// this file allocates more than 8 MiB, so it is safe to run while Strata is up.
#include <cstdio>
#include <cuda_runtime.h>
int main(){
    const cudaError_t init=cudaFree(0);
    size_t free_bytes=0,total_bytes=0;
    const cudaError_t info=cudaMemGetInfo(&free_bytes,&total_bytes);
    std::printf("cudaFree(0)=%s cudaMemGetInfo=%s free=%zu MiB total=%zu MiB\n",
        cudaGetErrorString(init),cudaGetErrorString(info),free_bytes>>20,total_bytes>>20);
    void *block=nullptr;
    const cudaError_t alloc=cudaMalloc(&block,8u<<20);
    std::printf("cudaMalloc(8 MiB)=%s\n",cudaGetErrorString(alloc));
    if(alloc==cudaSuccess)cudaFree(block);
    return init!=cudaSuccess||info!=cudaSuccess||alloc!=cudaSuccess;
}
