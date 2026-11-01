/* Handoff glue between two engine processes, kept as the stubs the Linux build
 * always compiled. Nothing here is active: ds4_combo_active is never set, so the
 * /decode/* routes fall through to the server's 404 and
 * ds4_combo_shutdown_requested stays false, which means shutdown always writes
 * the KV disk cache. Asking for the combo (DS4_COMBO_ENABLED=1) fails startup
 * with ds4_combo_start's reason instead of starting a handoff. */
static volatile bool ds4_combo_shutdown_requested=false;
static bool ds4_combo_start(server*s,const ds4_engine_options*o,char*e,size_t n){(void)s;(void)o;(void)n;const char*v=getenv("DS4_COMBO_ENABLED");if(v&&strcmp(v,"0")){if(e&&n)snprintf(e,n,"combo handoff is not implemented in this build");return false;}return true;}
static void ds4_combo_http(server*s,int fd,const http_request*r){(void)r;http_error(fd,s->enable_cors,409,"combo handoff is not implemented in this build");}
