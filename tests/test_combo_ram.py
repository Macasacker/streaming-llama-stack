"""The parent-owned checkpoint tier: bounded transfers, a name another process attaches to,
and a page lock reported as what the kernel actually granted."""
import contextlib
import ctypes
import errno
import hashlib
import io
import os
import subprocess
import sys
from pathlib import Path
import unittest

from slstack.combo_ram import (KERNEL_PAGE, LockedRam, attach, kernel_api, memlock_allowance_bytes,
                               reap_stale_sections, runtime_min_free_gib, section_name,
                               wait_for_host_capacity)
from slstack.working_set import wait_owned_initialization


class RuntimeFloorTest(unittest.TestCase):
    def test_decode_budget_stays_inside_the_aggregate_and_reuses_prefill_ceiling(self):
        from slstack.combo_producer import decode_host_budget
        G=1<<30
        self.assertEqual(decode_host_budget({},6*G),6*G)
        cfg=dict(host_budget_gib=13,producer_decode_host_budget_gib=12.5)
        self.assertEqual(decode_host_budget(cfg,6*G),int(12.5*G))
        with self.assertRaises(MemoryError):decode_host_budget(dict(cfg,producer_decode_host_budget_gib=12.6),6*G)
        for bad in (True,0,-1,float('nan'),float('inf'),'12'):
            with self.subTest(value=bad),self.assertRaises(ValueError):
                decode_host_budget(dict(cfg,producer_decode_host_budget_gib=bad),6*G)

    def test_default_is_half_and_zero_disables(self):
        self.assertEqual(runtime_min_free_gib({}),0.5)
        self.assertEqual(runtime_min_free_gib(dict(runtime_min_free_gib=0)),0.0)
        self.assertEqual(runtime_min_free_gib(dict(runtime_min_free_gib=6.5)),6.5)

    def test_invalid_floors_fail_before_launch(self):
        for value in (-1,float('nan'),float('inf'),True,'4',None,2**34):
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,'runtime_min_free_gib'):
                runtime_min_free_gib(dict(runtime_min_free_gib=value))


class CapacityTest(unittest.TestCase):
    def test_waits_for_exact_physical_capacity_and_reports_elapsed(self):
        now=[0.];readings=iter((2,5))
        def sleep(seconds):now[0]+=seconds
        result=wait_for_host_capacity(3,2,lambda:False,available=lambda:next(readings),clock=lambda:now[0],sleep=sleep)
        self.assertEqual(result['required_bytes'],5);self.assertEqual(result['available_bytes'],5)
        self.assertAlmostEqual(result['seconds'],.1)

    def test_timeout_retains_required_and_available_without_launch(self):
        now=[0.]
        def sleep(seconds):now[0]+=seconds
        with self.assertRaisesRegex(MemoryError,'available=2, required=5') as caught:
            wait_for_host_capacity(3,2,lambda:False,timeout_s=.2,available=lambda:2,clock=lambda:now[0],sleep=sleep)
        self.assertAlmostEqual(caught.exception.admission_receipt['seconds'],.2)

    def test_cancel_does_not_wait_or_admit(self):
        with self.assertRaisesRegex(RuntimeError,'cancelled'):
            wait_for_host_capacity(3,2,lambda:True,available=lambda:5)

    def test_slow_but_rising_reclamation_waits_past_the_window(self):
        # Available RAM rises by 1 byte every 0.5 s for 60 s (> the 30 s window): progress renews it.
        now=[0.]
        def sleep(seconds):now[0]+=seconds
        result=wait_for_host_capacity(100,20,lambda:False,available=lambda:min(120,int(now[0]/.5)),clock=lambda:now[0],sleep=sleep)
        self.assertEqual(result['available_bytes'],120);self.assertGreater(result['seconds'],59)


class _Job:
    def __init__(self,code=None):self.code=code;self.pid=4242
    def poll(self):return self.code


class OwnedInitializationTest(unittest.TestCase):
    """Owner rule: no fixed deadline. Alive + growing RSS keeps waiting; only exit or a
    no-progress window fails, with the real reason."""
    def clock(self):
        now=[0.]
        def sleep(seconds):now[0]+=seconds
        return now,sleep

    def test_slow_growth_past_the_old_two_second_deadline(self):
        now,sleep=self.clock()   # 8 MiB reached only after ~12 s of slow, steady growth
        row=wait_owned_initialization(_Job(),lambda:int(now[0]*700_000),clock=lambda:now[0],sleep=sleep)
        self.assertGreaterEqual(row['rss_bytes'],8<<20);self.assertGreater(row['seconds'],11)

    def test_no_progress_window_fails_with_the_reason(self):
        now,sleep=self.clock()
        with self.assertRaisesRegex(TimeoutError,r'no progress for 30\.\d+s: rss=1000 of 8388608, process alive'):
            wait_owned_initialization(_Job(),lambda:1000,clock=lambda:now[0],sleep=sleep)
        self.assertLess(now[0],31)

    def test_stall_after_growth_is_measured_from_the_last_growth(self):
        now,sleep=self.clock()   # grows for 20 s, then flat: fails ~30 s after the last growth
        with self.assertRaises(TimeoutError):
            wait_owned_initialization(_Job(),lambda:int(min(now[0],20)*1000),clock=lambda:now[0],sleep=sleep)
        self.assertGreater(now[0],49);self.assertLess(now[0],51)

    def test_exit_fails_at_once_with_the_code(self):
        now,sleep=self.clock()
        with self.assertRaisesRegex(RuntimeError,r'exited \(code 3\)'):
            wait_owned_initialization(_Job(3),lambda:0,clock=lambda:now[0],sleep=sleep)
        self.assertEqual(now[0],0)


@unittest.skipUnless(sys.platform=='win32','Windows shared RAM contract')
class LockedRamTest(unittest.TestCase):
    def test_locked_shared_pages_and_bounded_transfer(self):
        with LockedRam(8<<20,session='1'*32,host_budget_gib=.5,min_available_gib=4) as ram:
            ram.write(0,b'checkpoint-owned-RAM')
            k=kernel_api();handle=k.OpenFileMappingW(6,False,ram.name)
            self.assertTrue(handle);address=k.MapViewOfFile(handle,6,0,0,ram.capacity)
            self.assertTrue(address)
            try:
                self.assertEqual(ctypes.string_at(address,20),b'checkpoint-owned-RAM')
                ctypes.memmove(address+20,b'!',1)
                self.assertEqual(ram.read(0,21),b'checkpoint-owned-RAM!')
            finally:k.UnmapViewOfFile(address);k.CloseHandle(handle)
            self.assertTrue(ram.locked)
            self.assertEqual(ram.sha256(0,21),hashlib.sha256(b'checkpoint-owned-RAM!').hexdigest())
            for offset,count in ((0,65537),(ram.capacity,1),(-1,4)):
                with self.assertRaises(ValueError):ram.read(offset,count)
            with ram.lease(),self.assertRaises(RuntimeError):ram.close()
        self.assertIsNone(ram.address)
        self.assertIsNone(ram.handle)
        with self.assertRaises(RuntimeError):ram.read(0,1)

    def test_budget_and_floor_refuse_before_allocation(self):
        with self.assertRaises(MemoryError):LockedRam(1<<20,session='1'*32,host_budget_gib=.0001,min_available_gib=4)
        with self.assertRaises(MemoryError):LockedRam(1<<20,session='1'*32,host_budget_gib=.5,min_available_gib=64)

    def test_cross_process_reads_the_same_section(self):
        with LockedRam(1<<20,session='2'*32,host_budget_gib=.5,min_available_gib=4) as ram:
            ram.write(123,b'native-to-ultra')
            program='from slstack.combo_ram import kernel_api;import ctypes,sys;k=kernel_api();h=k.OpenFileMappingW(4,False,sys.argv[1]);assert h;p=k.MapViewOfFile(h,4,0,0,1048576);assert p;assert ctypes.string_at(p+123,15)==b"native-to-ultra";assert k.UnmapViewOfFile(p);assert k.CloseHandle(h);print("SHARED_RAM_CHILD_PASS")'
            result=subprocess.run([sys.executable,'-B','-c',program,ram.name],capture_output=True,text=True,timeout=15)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('SHARED_RAM_CHILD_PASS',result.stdout)


@unittest.skipUnless(os.name!='nt','Linux shared checkpoint contract')
class SharedCheckpointTests(unittest.TestCase):
    """The same tier on `/dev/shm`: bounded transfers, one name two processes share, and a
    page lock that reports only what the kernel granted."""
    def test_shared_pages_and_bounded_transfer(self):
        with LockedRam(8<<20,session='1'*32,host_budget_gib=.5,min_available_gib=4) as ram:
            ram.write(0,b'checkpoint-owned-RAM')
            self.assertTrue(ram.locked)
            attached=attach(ram.name,ram.capacity)
            try:
                self.assertEqual(attached.read(0,20),b'checkpoint-owned-RAM')
                attached.write(20,b'!')
                self.assertEqual(ram.read(0,21),b'checkpoint-owned-RAM!')
            finally:
                attached.close()
            self.assertEqual(ram.sha256(0,21),hashlib.sha256(b'checkpoint-owned-RAM!').hexdigest())
            for offset,count in ((0,65537),(ram.capacity,1),(-1,4),(ram.capacity-1,2)):
                with self.subTest(offset=offset,count=count),self.assertRaises(ValueError):
                    ram.read(offset,count)
            with ram.lease(),self.assertRaises(RuntimeError):ram.close()
            self.assertEqual(section_name(ram.name).read_bytes()[:20],b'checkpoint-owned-RAM')
        self.assertIsNone(ram.address)
        self.assertIsNone(ram.handle)
        self.assertIsNone(ram.mapping)
        self.assertFalse(section_name(ram.name).exists(),'the name must not outlive the owner')
        with self.assertRaises(RuntimeError):ram.read(0,1)

    def test_the_page_lock_reports_only_what_the_kernel_granted(self):
        allowance=memlock_allowance_bytes()
        capacity=16<<20 if allowance<16<<20 else 1<<20
        captured=Capture()
        with captured:
            ram=LockedRam(capacity,session='3'*32,host_budget_gib=.5,min_available_gib=4)
        try:
            self.assertLessEqual(ram.locked_bytes,capacity)
            self.assertEqual(ram.locked_bytes%KERNEL_PAGE,0,'lock chunks must stay whole pages')
            short=ram.locked_bytes<capacity
            self.assertEqual('DS4COMBO_LOCKED_RAM_PARTIAL' in captured.text(),short)
            if short:
                self.assertIn('"shortfall_bytes":'+str(capacity-ram.locked_bytes),captured.text())
                self.assertIn('"memlock_allowance_bytes":'+str(allowance),captured.text())
                self.assertEqual(ram.lock_reason,errno.ENOMEM,
                                 'this kernel answers a lock past RLIMIT_MEMLOCK with ENOMEM, not EAGAIN')
        finally:
            ram.close()
        self.assertFalse(section_name(ram.name).exists())

    def test_budget_and_floor_refuse_before_allocation(self):
        with self.assertRaises(MemoryError):LockedRam(1<<20,session='1'*32,host_budget_gib=.0001,min_available_gib=4)
        with self.assertRaises(MemoryError):LockedRam(1<<20,session='1'*32,host_budget_gib=.5,min_available_gib=64)
        with self.assertRaises(ValueError):LockedRam(3*1024**3,session='1'*32,host_budget_gib=8,min_available_gib=4)
        with self.assertRaises(ValueError):LockedRam(1<<20,session='not-a-nonce',host_budget_gib=.5,min_available_gib=4)

    def test_only_one_creator_can_take_a_name(self):
        with LockedRam(1<<20,session='4'*32,host_budget_gib=.5,min_available_gib=4) as ram:
            location=section_name(ram.name)
            self.assertTrue(location.exists())
            with self.assertRaises(FileExistsError):
                os.close(os.open(str(location),os.O_CREAT|os.O_EXCL|os.O_RDWR,0o600))
            with self.assertRaises(FileNotFoundError):attach('DS4COMBO-'+'7'*32+'-1-'+'e'*32,1<<20)

    def test_cross_process_reads_the_same_section(self):
        with LockedRam(1<<20,session='2'*32,host_budget_gib=.5,min_available_gib=4) as ram:
            ram.write(123,b'native-to-ultra')
            program=('from slstack.combo_ram import attach;import sys;'
                     'a=attach(sys.argv[1],1<<20);'
                     'assert a.read(123,15)==b"native-to-ultra";'
                     'a.write(138,b"-child");'
                     'assert a.read(123,21)==b"native-to-ultra-child";'
                     'a.close();print("SHARED_RAM_CHILD_PASS")')
            result=subprocess.run([sys.executable,'-B','-c',program,ram.name],capture_output=True,text=True,timeout=15,
                                  env=dict(os.environ,PYTHONPATH=os.pathsep.join([os.getcwd(),'tests'])))
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('SHARED_RAM_CHILD_PASS',result.stdout)
            self.assertEqual(ram.read(123,21),b'native-to-ultra-child')

    def test_a_name_that_could_escape_the_shared_memory_directory_is_refused(self):
        for name in ('../passwd','DS4COMBO-../../x','plain',''):
            with self.subTest(name=name),self.assertRaises(ValueError):section_name(name)

    def test_a_section_whose_creating_process_is_gone_is_reaped(self):
        parent=os.getpid()
        stale=section_name('DS4COMBO-'+'5'*32+'-999999-'+hashlib.sha256(b'stale').hexdigest()[:32])
        stale.write_bytes(b'left behind by a killed controller')
        self.assertEqual(stale.stat().st_uid,os.getuid())
        self.assertIn(stale.name,reap_stale_sections())
        self.assertFalse(stale.exists())
        live=section_name('DS4COMBO-'+'6'*32+'-'+str(parent)+'-'+hashlib.sha256(b'live').hexdigest()[:32])
        live.write_bytes(b'held by a controller that is still here')
        try:
            self.assertNotIn(live.name,reap_stale_sections())
            self.assertTrue(live.exists())
        finally:
            live.unlink()


class Capture:
    """Captures what the tier prints, so a shortfall is asserted rather than eyeballed."""
    def __enter__(self):
        self.buffer=io.StringIO()
        self.redirect=contextlib.redirect_stdout(self.buffer)
        self.redirect.__enter__()
        return self

    def __exit__(self,*args):self.redirect.__exit__(*args)

    def text(self):return self.buffer.getvalue()


if __name__=='__main__':unittest.main()
