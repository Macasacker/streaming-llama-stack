"""A deterministic engine abort is reported as one (slstack.request_error, combo_single).

Pi retried the top-k coverage abort three times; each attempt retired the producer and re-prefilled
the conversation. The stack cannot set the client's policy, but it can say what it knows: the
`x-should-retry` header pi honours on a buffered reply, and `retryable: false` in the error object,
which is all that is left once an SSE stream has already sent its 200."""
import json
import os
import socket
import sys
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT));sys.path.insert(0, str(ROOT / 'tests'))
from slstack import combo_single as s
from slstack import request_error
from test_combo_single import FakeProducer, Harness

ABORT = ('Persistent producer aborted (exception, sequence 3): Actual exact radix top-k first/merge '
         'dispatch coverage differs; producer exit code 1 (0x00000001); LLAMADECODE_PRODUCER_ABORT '
         'kind=exception code=0x00000000 sequence=3 reason=Actual exact radix top-k first/merge '
         'dispatch coverage differs')


class ClassificationTests(unittest.TestCase):
    def test_an_engine_self_check_is_not_worth_repeating(self):
        self.assertFalse(request_error.is_retryable(ABORT))
        for reason in ('Actual completed original-Q8 BF16 final stores differ',
                       'Actual F16 route-weight store coverage differs',
                       'Routed decode selected-expert geometry differs',
                       'Producer frontier 12 differs from the live token count 11',
                       'Request body exceeds its admitted bound'):
            self.assertFalse(request_error.is_retryable(reason), reason)

    def test_a_condition_of_the_moment_is(self):
        for reason in ('Physical free VRAM fell below the decode VRAM reserve',
                       'cancelled: Producer cancelled; incomplete output retained and not certified',
                       'Session parent input pipe closed',
                       'std::bad_alloc: a host allocation failed',
                       'RTX is not idle enough for exclusive admission; no other process was ejected',
                       '[Errno 98] Address already in use'):
            self.assertTrue(request_error.is_retryable(reason), reason)

    def test_a_self_check_reached_while_resources_ran_out_reads_as_transient(self):
        self.assertTrue(request_error.is_retryable('Coverage differs after free VRAM fell below the reserve'))

    def test_the_saying_names_the_cost(self):
        #    The message has to explain why repeating it does not help, not just that it failed.
        text = request_error.describe(ABORT)
        self.assertIn('engine self-check', text)
        self.assertIn('re-prefilled', text)
        self.assertIn('Actual exact radix top-k', text)
        self.assertEqual(request_error.describe('free VRAM fell'), 'free VRAM fell')
        self.assertEqual(request_error.retry_header(ABORT), 'false')
        self.assertIsNone(request_error.retry_header('free VRAM fell'))

    def test_the_error_object_carries_the_flag(self):
        #    A streamed error cannot carry a header, so the body has to say it.
        body = request_error.error(ABORT)
        self.assertFalse(body['error']['retryable'])
        self.assertEqual(body['error']['type'], 'single_engine_refused')
        self.assertNotIn('retryable', request_error.error('free VRAM fell')['error'])


class AbortingProducer(FakeProducer):
    def __init__(self, tokens, script, directory, message):
        super().__init__(tokens, script, directory);self.message = message
    def decode(self, max_tokens, stop_tokens=()):
        raise RuntimeError(self.message)


def request(address, token):
    body = json.dumps(dict(messages=[dict(role='user', content='q')], thinking=False, max_tokens=8)).encode()
    client = socket.create_connection(address, timeout=20)
    client.sendall(b'POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer ' + token.encode() +
                   b'\r\nContent-Type: application/json\r\nContent-Length: ' + str(len(body)).encode() + b'\r\n\r\n' + body)
    head = b''
    while b'\r\n\r\n' not in head:
        chunk = client.recv(4096)
        if not chunk:break
        head += chunk
    headers, rest = head.split(b'\r\n\r\n')[0].decode().split('\r\n'), b''
    status = int(headers[0].split(' ')[1])
    length = next((int(line.split(':')[1]) for line in headers if line.lower().startswith('content-length')), 0)
    payload = head.split(b'\r\n\r\n')[1]
    while len(payload) < length:payload += client.recv(4096)
    client.close()
    return status, {line.split(':')[0].lower(): line.split(':', 1)[1].strip() for line in headers[1:]}, json.loads(payload)


class HandlerReplyTests(unittest.TestCase):
    def serve(self, message):
        os.environ['DS4_SERVER_AUTH_TOKEN'] = 't' * 40
        h = Harness([[]])
        h.engine.producer_factory = lambda tokens: AbortingProducer(tokens, [], h.dir, message)
        cfg = dict(context=4096, lease_minutes=19, runtime_root=h.dir, model_name='DeepSeek-V4.1-Flash',
                   request_timeout_s=30)
        proxy = s.SingleProxy(cfg, 'run', None, engine=h.engine)
        server = ThreadingHTTPServer(('127.0.0.1', 0), s.handler(proxy))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return h, server

    def answer(self, message):
        h, server = self.serve(message)
        try:
            return request(server.server_address, 't' * 40)
        finally:
            server.shutdown();server.server_close();h.engine.close()

    def test_a_self_check_reply_tells_the_client_not_to_retry(self):
        status, headers, payload = self.answer(ABORT)
        self.assertEqual(status, 503)
        self.assertEqual(headers.get('x-should-retry'), 'false')
        self.assertFalse(payload['error']['retryable'])
        self.assertIn('engine self-check', payload['error']['message'])

    def test_a_transient_reply_leaves_the_policy_to_the_client(self):
        status, headers, payload = self.answer('Physical free VRAM fell below the decode VRAM reserve')
        self.assertEqual(status, 503)
        self.assertNotIn('x-should-retry', headers)
        self.assertNotIn('retryable', payload['error'])
        self.assertEqual(payload['error']['message'], 'Physical free VRAM fell below the decode VRAM reserve')

    def test_the_request_record_keeps_the_classification(self):
        h, server = self.serve(ABORT)
        try:
            request(server.server_address, 't' * 40)
        finally:
            server.shutdown();server.server_close();h.engine.close()
        from pathlib import Path as P
        rows = [json.loads(name.read_text()) for name in P(h.dir).glob('single-request-*.json')]
        self.assertTrue(rows)
        self.assertFalse(all(row['retryable'] for row in rows))


if __name__ == '__main__':unittest.main()
