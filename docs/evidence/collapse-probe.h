// Sweep input for scripts/port/collapse_conditionals.py - what each decision path must do.
//
//   python3 scripts/port/collapse_conditionals.py --tokens DEAD_ONE,DEAD_TWO,__linux__ \
//       --never-defined DEAD_ONE,DEAD_TWO --apply build/42/probe.h
//   gcc -E -fdirectives-only -P -x c docs/evidence/collapse-probe.h > before.i
//   gcc -E -fdirectives-only -P -x c build/42/probe.h               > after.i
//   diff before.i after.i                                            # must print nothing
//
// The last command is the whole gate: the collapse may only remove text the preprocessor already
// removed, so the preprocessed text of the applied copy has to be identical to this file's.
// `__linux__` is in the tool's true-platform set, so it stands for a macro this build defines;
// DEAD_ONE and DEAD_TWO are pinned false by --never-defined.
//
// Cases, in order:
//   - provably-true `&&` terms and provably-false `||` terms are dropped from a mixed guard.
//   - a provably-true operand of `||` is NOT dropped (`X || true` is true; dropping it would
//     leave X). The whole guard collapses instead, because `true || anything` is true.
//   - a provably-false operand of `&&` is NOT dropped (`false && X` is false).
//   - a multi-line directive whose terms are all dead goes whole, continuation lines included.
//   - a multi-line directive that cannot be decided stays byte-for-byte, and is reported.
//   - a dead `#if` whose next branch's condition the tool cannot parse is reported, not rewritten.
//   - a dead `#if` with a live `#else` keeps the else body and loses both directive lines.
//   - a dead `#if` with a live `#elif` promotes the `#elif` to `#if` and KEEPS the `#endif`,
//     because a guard was introduced, not removed.
//   - a conditional nested inside the surviving branch of a dead outer one is work too.
//   - an expression whose grammar the tokenizer cannot follow (here `defined(A) + 1`) is reported
//     and left alone, even though the compiler would happily evaluate it.

#if !defined(DEAD_ONE) && !defined(DEAD_TWO) && CUDART_VERSION >= 11070
    drop_two_true_terms();
#endif

#if CUDART_VERSION >= 12000 || defined(DEAD_ONE)
    drop_false_or_term();
#endif

#if defined(__linux__) && CUDART_VERSION >= 11070
    true_and_term_dropped();
#endif

#if defined(__linux__) || CUDART_VERSION >= 2
    true_or_term_whole_guard_is_live();
#endif

#if !defined(__linux__) || CUDART_VERSION >= 3
    false_or_term_dropped();
#endif

#if !defined(__linux__) && CUDART_VERSION >= 4
    false_and_term_must_stay_false();
#endif

#if defined(DEAD_ONE) || \
    defined(DEAD_TWO)
    multiline_all_dead_goes_whole();
#endif

#if defined(UNKNOWN_THING) || \
    defined(DEAD_ONE)
    multiline_undecided_must_stay();
#endif

#if defined(DEAD_ONE)
    dead_branch();
#elif CUDART_VERSION >= 11010
    live_elif_after_a_dead_if_needs_a_human();
#else
    unreachable();
#endif

#if defined(DEAD_ONE)
    dead_branch_two();
#else
    live_else();
#endif

#if defined(DEAD_ONE)
    dead_before_live_elif();
#elif !defined(DEAD_TWO)
    live_via_elif_keeps_the_endif();
#else
    unreachable_else();
#endif

#if defined(DEAD_ONE)
    dead_outer();
#else
#if defined(DEAD_TWO)
    dead_nested();
#else
    live_nested();
#endif
    live_after_nested();
#endif

#if defined(DEAD_ONE) + 1
    unparsable_stays();
#endif
