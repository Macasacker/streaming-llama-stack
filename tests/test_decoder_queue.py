"""The queue in front of the single-owner decoder (slstack.decoder_queue, combo_single, combo_proxy).

A second request used to be refused with 409 "Another request owns this decoder", which costs the
client a retry against a cold engine; it now waits, and only overflows when the queue is full."""
import json
import os
import socket
import sys
import tempfile
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT));sys.path.insert(0, str(ROOT / 'tests'))
from slstack import combo_single as s
from slstack.decoder_queue import DecoderQueue, QueueFull, QueueTimeout
from test_combo_single import FakeProducer, Harness, char_tokenizer, ids


class QueueUnitTests(unittest.TestCase):
    def test_a_waiter_is_served_after_the_holder_releases(self):
        queue = DecoderQueue(threading.Lock(), depth=2)
        queue.enter(5)
        got = []
        def waiter():
            queue.enter(5);got.append('served');queue.exit()
        thread = threading.Thread(target=waiter);thread.start()
        time.sleep(.2)
        self.assertEqual(got, [])                      # it waits rather than being refused
        queue.exit();thread.join(5)
        self.assertEqual(got, ['served'])
        self.assertEqual(queue.waiting, 0)

    def test_overflow_names_the_depth(self):
        queue = DecoderQueue(threading.Lock(), depth=2)
        queue.enter(5)                                  # the request running now
        served = []
        def worker():
            queue.enter(30);served.append('served');time.sleep(.05);queue.exit()
        waiters = [threading.Thread(target=worker) for _ in range(2)]
        for thread in waiters:thread.start()
        deadline = time.monotonic() + 5
        while queue.waiting < 2 and time.monotonic() < deadline:time.sleep(.02)
        self.assertEqual(queue.waiting, 2)              # two waiting, one running
        with self.assertRaises(QueueFull) as caught:
            queue.enter(.2)                             # the third in line is refused, not queued
        self.assertIn('2 requests are already waiting', str(caught.exception))
        queue.exit()
        for thread in waiters:thread.join(10)
        self.assertEqual(len(served), 2)
        self.assertEqual(queue.waiting, 0)

    def test_a_window_that_elapses_is_a_timeout_not_a_refusal(self):
        queue = DecoderQueue(threading.Lock(), depth=4)
        queue.enter(5)
        with self.assertRaises(QueueTimeout) as caught:
            queue.enter(.2)
        self.assertIn('0.2 s admission window', str(caught.exception))
        self.assertEqual(queue.waiting, 0)
        queue.exit()

    def test_a_queue_that_admits_nothing_is_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            DecoderQueue(threading.Lock(), depth=0)


class BlockingProducer(FakeProducer):
    """decode() parks until the test opens the gate, so the decoder is provably held."""
    def __init__(self, tokens, script, directory, gate):
        super().__init__(tokens, script, directory);self.gate = gate
    def decode(self, max_tokens, stop_tokens=()):
        self.gate.wait()
        return super().decode(max_tokens, stop_tokens)


def post(address, body, token):
    data = json.dumps(body).encode()
    client = socket.create_connection(address, timeout=30)
    client.sendall(b'POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer ' + token.encode() +
                   b'\r\nContent-Type: application/json\r\nContent-Length: ' + str(len(data)).encode() + b'\r\n\r\n' + data)
    head = b''
    while b'\r\n\r\n' not in head:head += client.recv(4096)
    headers = head.split(b'\r\n\r\n')[0].decode()
    status = int(headers.split('\r\n')[0].split(' ')[1])
    length = next((int(line.split(':')[1]) for line in headers.split('\r\n') if line.lower().startswith('content-length')), 0)
    payload = head.split(b'\r\n\r\n')[1]
    while len(payload) < length:payload += client.recv(4096)
    client.close()
    return status, payload.decode()


class QueueServingTests(unittest.TestCase):
    def setUp(self):
        os.environ['DS4_SERVER_AUTH_TOKEN'] = 't' * 40
        self.token = 't' * 40

    def harness(self, scripts, depth):
        h = Harness(scripts);gate = threading.Event();h.gate = gate
        def factory(tokens):
            producer = BlockingProducer(tokens, scripts, h.dir, gate);h.launched.append(producer);return producer
        h.engine.producer_factory = factory
        # rather than on the proxy's copy of the configuration.
        h.engine.queue.depth = depth
        cfg = dict(context=4096, lease_minutes=19, runtime_root=h.dir, model_name='DeepSeek-V4.1-Flash',
                   request_timeout_s=30, single_engine_queue_depth=depth)
        proxy = s.SingleProxy(cfg, 'run', None, engine=h.engine)
        server = ThreadingHTTPServer(('127.0.0.1', 0), s.handler(proxy))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        h.server = server
        return h

    def body(self, text='q'):
        return dict(messages=[dict(role='user', content=text)], thinking=False, max_tokens=8)

    def test_a_second_request_waits_and_is_served(self):
        script = ids(char_tokenizer(), 'a') + [s.EOS_ID]
        h = self.harness([list(script), list(script)], 4)
        try:
            first = {};second = {}
            def run(slot):
                started = time.monotonic()
                slot['result'] = post(h.server.server_address, self.body(), self.token)
                slot['elapsed'] = time.monotonic() - started
            a = threading.Thread(target=run, args=(first,));a.start()
            deadline = time.monotonic() + 10
            while not h.engine.lock.locked() and time.monotonic() < deadline:time.sleep(.05)
            b = threading.Thread(target=run, args=(second,));b.start()
            time.sleep(.3)
            self.assertFalse(second, 'the second request must wait, not answer early')
            self.assertEqual(h.engine.queue.waiting, 1)
            h.gate.set()
            a.join(30);b.join(30)
            self.assertEqual(first['result'][0], 200)
            self.assertEqual(second['result'][0], 200, second['result'][1][:200])
            self.assertGreater(second['elapsed'], .2)   # it waited behind the first, then was served
        finally:
            h.gate.set();h.server.shutdown();h.server.server_close();h.engine.close()

    def test_a_full_queue_answers_429_and_says_so(self):
        script = ids(char_tokenizer(), 'a') + [s.EOS_ID]
        h = self.harness([list(script) for _ in range(3)], 1)
        try:
            answers = []
            def request():answers.append(post(h.server.server_address, self.body(), self.token))
            running = threading.Thread(target=request);running.start()
            deadline = time.monotonic() + 10
            while not h.engine.lock.locked() and time.monotonic() < deadline:time.sleep(.05)
            queued = threading.Thread(target=request);queued.start()
            deadline = time.monotonic() + 10
            while h.engine.queue.waiting < 1 and time.monotonic() < deadline:time.sleep(.05)
            self.assertEqual(h.engine.queue.waiting, 1)
            status, payload = post(h.server.server_address, self.body(), self.token)   # the third in line
            h.gate.set()
            running.join(30);queued.join(30)
            self.assertEqual(status, 429)
            error = json.loads(payload)['error']
            self.assertEqual(error['type'], 'engine_queue_full')
            self.assertIn('1 requests are already waiting', error['message'])
            self.assertEqual([code for code, _ in answers], [200, 200])   # the two it did admit were served
        finally:
            h.gate.set();h.server.shutdown();h.server.server_close();h.engine.close()

    def test_a_configuration_without_a_timeout_still_serves(self):
        # The admission window used to be read as cfg['request_timeout_s']; on a partial
        # configuration that raised mid-request and the stream died without a status line.
        script = ids(char_tokenizer(), 'answer') + [s.EOS_ID]
        h = Harness([list(script)])
        cfg = dict(context=4096, lease_minutes=19, runtime_root=h.dir, model_name='DeepSeek-V4.1-Flash')
        proxy = s.SingleProxy(cfg, 'run', None, engine=h.engine)
        server = ThreadingHTTPServer(('127.0.0.1', 0), s.handler(proxy))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            status, payload = post(server.server_address, self.body(), self.token)
            self.assertEqual(status, 200, payload[:200])
            self.assertEqual(json.loads(payload)['choices'][0]['message']['content'], 'answer')
        finally:
            server.shutdown();server.server_close();h.engine.close()

    def test_the_queue_wait_is_recorded_for_the_request(self):
        script = ids(char_tokenizer(), 'a') + [s.EOS_ID]
        h = self.harness([list(script), list(script)], 4)
        try:
            first = threading.Thread(target=lambda: post(h.server.server_address, self.body(), self.token));first.start()
            deadline = time.monotonic() + 10
            while not h.engine.lock.locked() and time.monotonic() < deadline:time.sleep(.05)
            h.gate.set()
            status, _ = post(h.server.server_address, self.body(), self.token)
            first.join(30)
            self.assertEqual(status, 200)
            waits = []
            for name in Path(h.dir).glob('single-request-*.json'):
                row = json.loads(name.read_text())
                if 'queue_wait_seconds' in row:waits.append(row['queue_wait_seconds'])
            self.assertTrue(any(w > 0 for w in waits), f'no queue wait recorded in {waits}')
        finally:
            h.gate.set();h.server.shutdown();h.server.server_close();h.engine.close()


class ConfigTests(unittest.TestCase):
    def test_the_queue_depth_is_a_bounded_integer(self):
        from slstack import config
        for bad in (0, 65, 4.5, 'four'):
            with self.assertRaises(ValueError):
                config.check_decoder_queue_depth(bad)
        for good in (1, 4, 64):
            self.assertIsNone(config.check_decoder_queue_depth(good))


if __name__ == '__main__':unittest.main()
