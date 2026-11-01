/* Handoff glue between two engine processes, kept as the refusal stubs the Linux
 * build has always compiled: the combo lifecycle is not implemented here, so
 * starting it reports the refusal and its HTTP control op answers 409. The
 * server still honours the authenticated /decode/shutdown op, which leaves the
 * accept loop, drains workers and closes the engine (ds4_gpu_cleanup frees
 * pinned, registered and managed memory). */
static volatile bool ds4_combo_shutdown_requested=false;
static bool ds4_combo_start(server*s,const ds4_engine_options*o,char*e,size_t n){(void)s;(void)o;(void)e;(void)n;const char*v=getenv("DS4_COMBO_ENABLED");return !v||!strcmp(v,"0");}
static void ds4_combo_http(server*s,int fd,const http_request*r){(void)r;http_error(fd,s->enable_cors,409,"combo handoff is not implemented in this build");}
