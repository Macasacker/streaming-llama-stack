"""Hold a launched child before its exec until its controller admits it.

The vendored Windows runner creates the producer with `CREATE_SUSPENDED`, assigns it to its
Job, and only then calls `ResumeThread`, so no engine instruction runs before the tree is
contained. Linux has no create-suspended call, so this launcher is the child that waits:

* it takes the process group the controller will signal (`setsid`), so a cooperative
  SIGINT and the force fallback reach every descendant the engine later spawns;
* it publishes its pid once that is done, which is the controller's confirmation that
  containment exists before anything is claimed about the child;
* it execs the engine only when the controller writes to the control descriptor.

If the controller goes away first, the launcher leaves with `EXIT_UNSTARTED` having run no
engine code - the counterpart of closing a Job handle while `KILL_ON_JOB_CLOSE` was set.
The parent-death signal covers the same window and `execv` clears it, which is why the
resumed tree survives its controller exactly as the released job handle let it.

Run standalone (`python -I slstack/producer_launch.py <wait_fd> <ready_fd> <argv...>`)
with no import of this package: the target's environment and working directory are already
the controller's, and the two descriptors are the only extra fds the child inherits.
"""
import ctypes
import os
import signal
import sys

EXIT_UNSTARTED = 125        # the controller retired before admission; no engine code ran
EXIT_EXEC_FAILED = 127      # the admitted target could not be execed
PR_SET_PDEATHSIG = 43


def claim_process_group():
    """Become the leader of the group the controller signals, before exec."""
    os.setsid()


def refuse_parent_exit():
    """Retire an unstarted child when its controller goes away.

    Cleared by `execv`, so this covers only the window the Job's kill-on-close covered:
    before the engine ran. After exec the tree is owned cooperatively and survives.
    """
    ctypes.CDLL(None).prctl(PR_SET_PDEATHSIG, signal.SIGKILL)


def publish_ready(ready_fd, pid):
    """Say who this is, then hand the descriptor back: the engine must not inherit it."""
    os.write(ready_fd, str(pid).encode('ascii'))
    os.close(ready_fd)


def wait_for_admission(wait_fd):
    """One byte means resume; an EOF means the controller's write end is gone."""
    try:
        return os.read(wait_fd, 1)
    finally:
        os.close(wait_fd)


def report_exec_failure(target, error):
    """One diagnostic line on the producer's stderr pipe, then the exit code."""
    try:
        os.write(2, b'DSV41 LAUNCHER: cannot exec %s: %s\n'
                 % (os.fsencode(target), str(error).encode('utf-8', errors='replace')))
    except OSError:
        pass


def main(arguments):
    wait_fd, ready_fd = int(arguments[1]), int(arguments[2])
    target = arguments[3:]

    claim_process_group()
    refuse_parent_exit()
    publish_ready(ready_fd, os.getpid())

    if not wait_for_admission(wait_fd):
        return EXIT_UNSTARTED
    try:
        os.execv(target[0], target)
    except OSError as error:
        report_exec_failure(target[0], error)
        return EXIT_EXEC_FAILED


if __name__ == '__main__':
    sys.exit(main(sys.argv))
