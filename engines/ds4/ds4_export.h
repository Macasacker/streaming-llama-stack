/* Export marker for the entry points the producer binds with dlopen.
 *
 * Only the decode library (built with -DDS4_EXTERNAL_DECODE=1) exports them; a normal
 * engine build has no ds4x_* surface at all. */
#ifndef DS4_EXPORT_H
#define DS4_EXPORT_H

#ifdef DS4_EXTERNAL_DECODE
#define DS4X_EXPORT __attribute__((visibility("default")))
#endif

#endif
