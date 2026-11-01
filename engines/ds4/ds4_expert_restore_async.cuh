/* DS4COMBO asynchronous expert-cache restore (DS4COMBO_RESTORE_ASYNC, default 1).
 *
 * The restore list is read on the existing LOW-priority DirectStorage background
 * queue (ds4_ds_background_submit/poll, the queue the affinity promotion uses)
 * into this module's own pinned staging, while decode step 0 fetches what it
 * needs on demand. Single foreground owner, no worker thread, ticked between
 * decode layers.
 *
 * Slot safety: the restore never reads into a cache slot. A batch is published
 * only after its DS read has completed, under a device synchronize, with the
 * slot bytes copied synchronously; the slot is registered (valid=1) only after
 * its bytes have landed. An expert already resident at publication (demand
 * fetched its own copy) is skipped, so at most one writer ever owns a slot.
 *
 * Priority: items are read hottest first; each item carries the age the
 * synchronous restore would have given it (list position), reserved on the
 * tier tick at start, so demand admissions stay newer. When a tier has no free
 * slot, an item replaces the least recent slot only when that slot is older
 * than the item; otherwise newer demand residency wins and the item is counted
 * as displaced (it would have been evicted by the same admissions anyway). */
extern "C" int ds4_gpu_expert_restore_async_available(char *why,size_t n) {snprintf(why,n,"not implemented in this build");return 0;}
extern "C" void ds4_gpu_expert_restore_async_reset(void) {}
extern "C" int ds4_gpu_expert_restore_async_add(const ds4_gpu_stream_expert_table *,uint32_t,int,uint32_t) {return 0;}
extern "C" int ds4_gpu_expert_restore_async_start(uint32_t,char *err,size_t n) {snprintf(err,n,"not implemented in this build");return 0;}
extern "C" int ds4_gpu_expert_restore_async_tick(void) {return 1;}
extern "C" int ds4_gpu_expert_restore_async_drain(const char *) {return 1;}
extern "C" int ds4_gpu_expert_restore_async_state(char *,size_t) {return 0;}
