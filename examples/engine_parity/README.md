# Native engine comparisons

These receipts are inherited. They compare 384 full-vocabulary logit vectors and greedy
choices from the Windows build against a zero-bias control, and a complete 32-token
state from the deleted macOS engine against an independent core build with identical
settings, normalizing CRLF line endings.

The carry-bank check inspects every page of a 4,536,143,872-byte anonymous locked bank
and covers release and allocation failure. The producer comparison hashes the complete
10,559,816-byte state for 67 tokens in Accurate mode. These bounded comparisons do not
establish equality across all arithmetic profiles, models or contexts, and none of them
was run on the Linux fork.

The `reference_sha256` values are the ones chunk 8.1 of
[the port plan](../../docs/linux-port-plan.md) compares a Linux run against. No model
weights or raw states are included. See [performance](../../docs/PERFORMANCE.md) for
timing measurements and [YOCO](../../docs/ULTRAYOCO_QUALIFICATION.md) for its separate
approximation error.
