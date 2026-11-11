"""Whether a failed engine request is worth retrying, and how to say so.

Pi retries a stream error three times. A producer abort raised by an engine self-check is a
property of the tensor shapes, not of the moment, so the same request reproduces it — and each
attempt retires the producer, so every retry re-prefills the whole conversation. On
2026-11-11 that cost pi's session three cold launches against the same abort
(docs/SESSION_FAILURE_PLAN.md).

Nothing here changes the client's policy; it states the classification where a client can read it:
the `x-should-retry` header on a buffered error (which pi honours), and `retryable: false` in the
error object, which is the only channel left once an SSE stream has already sent its 200.
"""
import re

# The engine's own invariant wording — every such message states that two computed quantities
# differ, and they are computed from the graph, so they recur.
DETERMINISTIC = re.compile(r'\bdiffers?\b|\bdiffered\b|invariant|exceeds its admitted bound|not a total order')
# Conditions of the moment: memory, cancellation, a peer going away. A later attempt can differ.
TRANSIENT = re.compile(r'cancelled|bad_alloc|out of memory|free VRAM|pipe closed|Address already in use|'
                       r'not idle enough|timeout|timed out|Connection reset|Broken pipe')


def is_retryable(message):
    """True when retrying the same request could plausibly meet different conditions.

    Transient wins over deterministic wording: a message that names both (a self-check reached
    while the card ran out) is a resource problem, and a retry may well succeed.
    """
    text = message or ''
    if TRANSIENT.search(text):return True
    return not DETERMINISTIC.search(text)


def describe(message):
    """The failure, with the reason a retry will or will not help attached."""
    if is_retryable(message):return str(message)
    return (str(message) + ' — engine self-check: retrying this request unchanged reproduces it, '
            'and each attempt retires the producer, so the conversation is re-prefilled')


def error(message, *, type='single_engine_refused'):
    """The error object an engine reply carries, marked non-retryable when it should be."""
    body = dict(message=describe(message), type=type)
    if not is_retryable(message):body['retryable'] = False
    return dict(error=body)


def retry_header(message):
    """The `x-should-retry` value for a buffered error reply, or None to leave it unset."""
    return 'false' if not is_retryable(message) else None
