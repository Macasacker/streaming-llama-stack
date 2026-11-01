/* Optional displaced-expert promotion. Single serialized foreground owner,
 * one LOW DS read batch, existing VRAM slots; no second cache or worker thread. */
static bool cuda_affinity_background_prepare(uint64_t,uint64_t) {return true;}
static void cuda_affinity_background_finish(void) {}
static bool cuda_affinity_background_harvest(void) {return true;}
static void cuda_affinity_background_note(const cuda_router_affinity_metrics &,const cuda_router_residency &,
        const void *,uint64_t,uint32_t,uint32_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t) {}
extern "C" int ds4_gpu_cache_affinity_background_tick(void) {return 1;}
