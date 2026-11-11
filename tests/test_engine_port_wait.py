"""The pre-flight on the ports the stack binds (slstack.port_wait).

`[Errno 98] Address already in use` reached a client as a 503 because nothing checked the port
before binding, and a force-killed producer keeps its socket occupied for seconds without ever
reaching LISTEN. These tests hold a port with a real child process, which is the only way to
test the attribution rather than guess it."""
import json
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slstack.port_wait import PortHeld, held_ports, port_holder, socket_inodes, stack_ports, wait_for_ports


def free_port():
    probe = socket.socket()
    probe.bind(('127.0.0.1', 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


def holder(port, seconds=30):
    """A child that binds `port` and keeps it — a LISTEN socket, like a live engine."""
    return subprocess.Popen([sys.executable, '-c',
                             'import socket,time;s=socket.socket();s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);'
                             f's.bind(("127.0.0.1",{port}));s.listen(1);time.sleep({seconds})'])

def held_by(child, port, seconds=15):
    """Wait until the child's socket is actually visible, then return its pid."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if port_holder(port) == child.pid:return child.pid
        time.sleep(.05)
    raise AssertionError(f'child {child.pid} never came up on port {port}')



class StackPortsTests(unittest.TestCase):
    def test_both_bound_ports_are_listed_once(self):
        cfg = dict(port=8088, ds4_url='http://127.0.0.1:8089')
        self.assertEqual(stack_ports(cfg), [8088, 8089])

    def test_a_port_repeated_by_the_decode_url_is_not_listed_twice(self):
        self.assertEqual(stack_ports(dict(port=8089, ds4_url='http://127.0.0.1:8089')), [8089])

    def test_a_setting_that_is_not_a_port_is_ignored(self):
        self.assertEqual(stack_ports(dict(port='nope', ds4_url='http://x')), [])
        self.assertEqual(stack_ports(dict(ds4_url='no scheme')), [])


class TableTests(unittest.TestCase):
    def test_a_short_table_line_is_skipped_not_fatal(self):
        path = Path(tempfile.mkdtemp()) / 'tcp'
        # Two LISTEN rows (ports 8329 and 8123) with a header and a truncated line between them.
        path.write_text(
            '  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n'
            '   0: 0100007F:2089 00000000:0000 0A 00000000:00000000 00:00000000 00:000000  1000        0 2768955 1 0 0 0 0 0\n'
            '   1: junk\n'
            '   2: 0100007F:1FBB 00000000:0000 0A 00000000:00000000 00:00000000 00:000000  1000        0 2768999 1 0 0 0 0 0\n')
        self.assertEqual(socket_inodes(8329, str(path)), {'2768955'})
        self.assertEqual(socket_inodes(8123, str(path)), {'2768999'})
        self.assertEqual(socket_inodes(9999, str(path)), set())

    def test_a_missing_table_reads_as_empty(self):
        self.assertEqual(socket_inodes(8089, str(Path(tempfile.mkdtemp()) / 'absent')), set())


class HolderTests(unittest.TestCase):
    def test_a_free_port_has_no_holder(self):
        self.assertIsNone(port_holder(free_port()))

    def test_the_holder_is_the_process_holding_the_port(self):
        port = free_port()
        child = holder(port)
        try:
            deadline = time.monotonic() + 10
            while port_holder(port) is None and time.monotonic() < deadline:time.sleep(.05)
            self.assertEqual(port_holder(port), child.pid)
            self.assertEqual(held_ports(dict(port=port)), [(port, child.pid)])
        finally:
            child.terminate();child.wait(10)

    def test_a_holder_is_attributed_after_the_wait_when_it_is_gone(self):
        port = free_port()
        child = holder(port, 3)
        try:
            self.assertEqual(held_by(child, port), child.pid)
            started = time.monotonic()
            wait_for_ports(dict(port=port), 20)           # the child exits after 3 s; the wait covers it
            self.assertGreaterEqual(time.monotonic() - started, .5)
        finally:
            child.terminate();child.wait(10)

    def test_a_holder_that_stays_is_refused_by_pid_and_says_nothing_was_killed(self):
        port = free_port()
        child = holder(port)
        held_by(child, port)
        try:
            with self.assertRaises(PortHeld) as caught:
                wait_for_ports(dict(port=port), .5)
            text = str(caught.exception)
            self.assertIn(f'port {port} held by {child.pid}', text)
            self.assertIn('no process was killed', text)
            self.assertIn('gpu_handoff', text)            # points at the handoff, not at a kill
        finally:
            child.terminate();child.wait(10)

    def test_the_wait_ends_when_the_owner_stops_the_stack(self):
        port = free_port()
        child = holder(port)
        held_by(child, port)
        stopping = threading.Event();stopping.set()
        try:
            started = time.monotonic()
            with self.assertRaises(PortHeld):
                wait_for_ports(dict(port=port), 600, stopping=stopping)
            self.assertLess(time.monotonic() - started, 2)
        finally:
            child.terminate();child.wait(10)


if __name__ == '__main__':unittest.main()
