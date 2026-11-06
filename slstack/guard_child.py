"""Pass an existing guarded command its bounded stdin without buffering stdout.

This is the launcher of the pinned shared-guard path, so it is also where that path's
kernel containment is created: the guarded command is moved into its scope between fork
and exec, and its descendants inherit membership. What the scope held and whether it was
removed is written beside the spec for the guardian's receipt.
"""
import json
from pathlib import Path
import subprocess
import sys


def main():
    from slstack.envvars import restore_secrets
    from slstack.cgroup_scope import open_containment
    spec = restore_secrets(json.loads(Path(sys.argv[1]).read_text()))
    scope, scope_absent = open_containment(spec)
    child = subprocess.Popen(spec['command'], cwd=spec['cwd'], env=spec['env'],
                             stdin=subprocess.PIPE if 'stdin' in spec else subprocess.DEVNULL,
                             preexec_fn=scope.pre_exec if scope else None)
    try:
        child.communicate(spec['stdin'].encode() if 'stdin' in spec else None)
    finally:
        # Read what the scope held before removing it, so the peak and the OOM
        # accounting survive the directory.
        report = ({'memory_scope': scope.receipt(), 'scope_cleanup': scope.close()}
                  if scope else {'memory_scope': {'unavailable': scope_absent}, 'scope_cleanup': None})
        Path(sys.argv[1]).with_suffix('.scope.json').write_text(json.dumps(report, indent=2))
    return child.returncode


if __name__ == '__main__':
    sys.exit(main())
