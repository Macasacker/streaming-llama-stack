"""One decoder, many callers: the bounded queue in front of a single-owner engine.

A resident engine holds one producer and one conversation, so two requests cannot run at once.
Refusing the second one costs the client a retry against a cold engine — the whole conversation
re-prefills — so requests wait behind whoever holds the decoder. The queue is bounded because a
backlog on a decoder this slow is invisible to the caller, and an unbounded one turns one stuck
request into a hundred timed-out ones.
"""
import threading

DEFAULT_QUEUE_DEPTH = 4


class QueueFull(RuntimeError):
    """More requests are already waiting for the decoder than the queue admits."""


class QueueTimeout(RuntimeError):
    """The decoder stayed busy for the whole admission window."""


class DecoderQueue:
    """Admission accounting around a lock the engine also uses directly.

    The lock stays the engine's: retirement, checkpoints and the idle sweep take it too, and
    they must keep seeing one owner. `waiting` counts callers blocked on it, not the one running.
    """
    def __init__(self, lock, depth=DEFAULT_QUEUE_DEPTH):
        if depth < 1:raise ValueError('the decoder queue depth must be at least 1')
        self.lock = lock;self.depth = depth
        self.queue_lock = threading.Lock();self.waiting = 0

    def enter(self, timeout):
        with self.queue_lock:
            if self.waiting >= self.depth:
                raise QueueFull(f'{self.depth} requests are already waiting for the decoder')
            self.waiting += 1
        try:
            if not self.lock.acquire(timeout=timeout):
                raise QueueTimeout(f'the decoder stayed busy for the whole {timeout:g} s admission window')
        finally:
            with self.queue_lock:self.waiting -= 1

    def exit(self):
        self.lock.release()
