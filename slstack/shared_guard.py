"""Stack adapter around the pinned Studio ownership and paging implementation."""
import json
import hashlib
from pathlib import Path
import sys
import threading
import time
import psutil
from .journal import atomic_json
from .lease import HoldWatcher
from .native_broker import vendor
from .config import ROOT
from .progress import Window, activity

vendor()
from studio.job_guard import GuardedJob
from studio.job_paging import GLOBAL_PAGING_LIMIT

GiB = 2**30


class CleanupUnconfirmed(RuntimeError):
    owned_cleanup_uncertain = True


def containment_report(spec):
    """What the launcher's cgroup scope held, read from the report it wrote beside the spec."""
    report = Path(spec['_spec_path']).with_suffix('.scope.json')
    if not report.is_file():
        return {'memory_scope':{'unavailable':'the shared guard launcher wrote no scope report'},
                'scope_cleanup':None}
    return json.loads(report.read_text())


def supervise(spec):
    for name, pin in {'job_guard.py':'56d96b556f6b95ba72054100bc8c27a70937cba83a15577a476a1fd1cff87648',
                      'job_paging.py':'0387ed7fbade07d088c6149332c239f7ddfd3b2878f48d3f154c5d5742800c91'}.items():
        source = (ROOT/'vendor/studio/studio'/name).read_bytes().split(b'\n',2)[2].replace(b'\r\n',b'\n')
        if hashlib.sha256(source).hexdigest() != pin:
            raise RuntimeError('Pinned shared guard source changed: ' + name)
    watch = spec.get('hold_watch')
    holds = HoldWatcher(watch['path'],watch['identity'],task=watch.get('task','DS4STACK')) if watch else None
    parent = psutil.Process(spec['owner_pid'])
    if abs(parent.create_time()-spec['owner_created']) > .01:
        raise RuntimeError('Owning process identity changed before launch')
    class Adapter(GuardedJob):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.retirement_lock=threading.RLock()
            self.retirement_proof=None

        def _cooperative_ownership(self):
            # Prevent pipe-loss from bypassing cooperative retirement: the bootstrap
            # stays the group anchor until this handoff byte says the lease is done.
            if self.child and self.child.stdin and not self.child.stdin.closed:
                self.child.stdin.write(b'H')
                self.child.stdin.flush()
                self.child.stdin.close()

        def start(self,*args,**kwargs):
            result=super().start(*args,**kwargs)
            self._cooperative_ownership()
            return result

        def _stop(self):
            from .shutdown import ProgressOwnedProcesses
            with self.retirement_lock:
                if self.retirement_proof is not None:return self.retirement_proof
                # Preserve the pinned provider's identity ledger and paging checks.
                self._cooperative_ownership()
                owned=ProgressOwnedProcesses.from_owned(self.owned,
                    {**spec,'stop_reason':self.failure or 'shared guarded job retirement'})
                cleanup=owned.stop()
                # A straggler born between ownership polls stays inside the launcher's
                # cgroup scope, whose ceiling survives this adapter; an unconfirmed
                # cleanup holds the lease instead of releasing it.
                self.retirement_proof=cleanup
                return cleanup

        def close(self):
            from .shutdown import wait_for_thread,gpu_free_reader
            self.finished.set()
            if self.monitor and self.monitor is not threading.current_thread():
                try:wait_for_thread(self.monitor,self.child,diagnostic=spec['log'],
                    gpu_free=gpu_free_reader(spec.get('gpu_uuid') or (spec.get('gpu') or {}).get('gpu_uuid')))
                except RuntimeError as error:raise CleanupUnconfirmed(str(error)) from error
            return super().close()

        def _priority(self):
            super()._priority()
            if Path(spec['stop']).exists():
                raise RuntimeError('stop requested')
            if not parent.is_running() or abs(parent.create_time()-spec['owner_created']) > .01:
                raise RuntimeError('owning process exited')
            if holds and (reason := holds.reason()):
                raise RuntimeError('Yielding to shared hold: ' + reason)
            if spec.get('runtime_min_free_gib',0):
                from .memory_counters import available_physical_bytes
                if available_physical_bytes()<spec['runtime_min_free_gib']*GiB:
                    raise MemoryError('Runtime physical RAM floor crossed')
            if self.child is not None:
                progress.observe(activity(self.child.pid, spec['log']))
                if progress.expired():
                    raise TimeoutError('Owned operation made no progress within its stall window')
                self.deadline = progress.last + progress.seconds
    receipt = Path(spec['receipt'])
    certificate = receipt.with_suffix('.shared.json')
    host = spec['host_budget_gib'] * GiB
    progress = Window(spec['timeout_s'])
    # Linux has no private-commit charge: the launcher's cgroup scope is the hard
    # ceiling and RSS/floor checks are the budget, so both limits are the host budget.
    job = Adapter(rss_limit=host,commit_limit=host,floor=spec['min_available_gib']*GiB,
                  device_commit_allowance=0,
                  seconds=spec['timeout_s'],receipt=certificate)
    started = time.monotonic()
    code, reason = 1, None
    try:
        with open(spec['log'],'ab',buffering=0) as log:
            job.start([sys.executable,'-B','-m','slstack.guard_child',spec['_spec_path']],
                      cwd=spec['cwd'],env=spec['env'],stdout=log)
            code = job.wait()
    except BaseException as error:
        reason = str(error)
    finally:
        if 'ended' not in job.record:
            job.close()
        certain = job.child is None or bool(job.record.get('release_allowed'))
        record = {'exit_code':code,'reason':reason,'peak_tree_rss':job.record['peak_rss'],
                  'min_available':job.record['minimum_available'],'elapsed_s':time.monotonic()-started,
                  'host_budget_gib':spec['host_budget_gib'],'floor_gib':spec['min_available_gib'],
                  'runtime_min_free_gib':spec.get('runtime_min_free_gib',0),
                  'no_progress_window_s':spec['timeout_s'],
                  'job_commit_gib':host/GiB,
                  'child_pid':job.child.pid if job.child else None,
                  'shared_guard_commit':'2774636281ab32bf871497f0a8caae23c6b6cdc7',
                  'stack_retirement_policy':'cooperative request, progress-aware grace, logged force fallback',
                  'shared_guard_certificate':str(certificate),'cleanup_confirmed':certain,
                  'paging_policy':job.watcher.details(),
                  'owned_paging':{'attribution':job.watcher.observer.attribution,
                    'observed_paged_bytes':job.watcher.job_paged,
                    'observed_processes':job.watcher.observer.last_processes,
                    'scanned_pages':job.watcher.observer.scanned_pages,
                    'completed_scans':job.watcher.observer.completed_scans},
                  'system_pagefile_pressure':{'maximum_growth_bytes':job.watcher.max_global_growth,
                    'kill_above_bytes':GLOBAL_PAGING_LIMIT,'scope':'global; not process attribution'},
                  'paging_interpretation':'Owned observation is bounded and incremental. Zero observed pages is not an exhaustive zero-I/O proof.',
                  **containment_report(spec)}
        atomic_json(receipt,record)
        if not certain:
            identities = {(item['pid'],item['created']) for item in job.record.get('cleanup',{}).get('identities',[])}
            for file in (Path(spec['runtime_root'])/'state/leases').glob('*.json'):
                row = json.loads(file.read_text())
                if (row.get('guardian_pid'),row.get('guardian_created')) in identities and not row.get('released'):
                    atomic_json(file,{**row,'cleanup_uncertain':True,'cleanup_certificate':str(certificate)})
            raise CleanupUnconfirmed('Owned exit unconfirmed; resource lease must remain held: ' + str(certificate))
    return 0 if code==0 and reason is None else 1
