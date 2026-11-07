# E15: RTX 5090 power and clocks

Inherited measurement. Every number here came from the upstream Windows build (DirectStorage on an RTX 5090 with 32 GB of system RAM) or from the macOS decode half this fork deleted. Nothing has been re-measured on Linux, and chunk 8.1 of the port plan is the run that would replace them. See [the port plan](linux-port-plan.md).

Verdict: reject changing the power limit. Current, default and maximum
supported limits all report575W. No supported increase is available.

An independent45-second NVML observation during the original-Q4 indexer
campaign contains27 samples at GPU utilization80% or higher. Active clocks
are2737/2760/2797MHz and power423.37/495.52/516.89W (minimum/median/maximum).
Software power capping is active in16 of those27 samples; hardware and
software thermal slowdown are inactive in every active sample.

These independent one-second readings can miss short events. They do not
prove continuous boost or zero throttling. A sampled average below575W does
not exclude transient software power capping. No power or clock setting was
changed, and no clock-intervention speed gain is claimed. Earlier32K readings
also showed the575W limit; the machine was not running with the reference
system's reported400W limit.
