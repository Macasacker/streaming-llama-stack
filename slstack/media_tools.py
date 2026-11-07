"""Video delivery tools on Linux: the system FFmpeg, verified by running it.

The retired Windows setup downloaded a pinned FFmpeg archive into the runtime root, hashed
every file of it and refused a changed tool; the archive held `.exe` binaries, so nothing of
that belongs in a Linux-only fork. Here the distribution owns `ffmpeg` and `ffprobe`, and
readiness is what the two programs say about themselves - a name on `PATH` that cannot start
is not ready, because the media worker would fail mid-delivery instead of at setup.
"""
import shutil
import subprocess

TOOLS = ('ffmpeg', 'ffprobe')


def locate():
    return {name: shutil.which(name) for name in TOOLS}


def readiness():
    found = locate()
    tools = {}
    for name, program in found.items():
        if not program:
            tools[name] = dict(path=None, ready=False, reason='not on PATH')
            continue
        try:
            result = subprocess.run([program, '-version'], stdin=subprocess.DEVNULL,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, timeout=30)
        except (OSError, subprocess.SubprocessError) as exc:
            tools[name] = dict(path=program, ready=False, reason=f'{type(exc).__name__}: {exc}')
            continue
        if result.returncode:
            tools[name] = dict(path=program, ready=False,
                               reason=f'exits {result.returncode}: {(result.stdout or "").splitlines()[:1]}')
            continue
        first = (result.stdout or '').splitlines()
        tools[name] = dict(path=program, ready=True, version=first[0] if first else '')
    return dict(ready=all(row['ready'] for row in tools.values()), tools=tools,
                missing=[name for name, row in tools.items() if not row['ready']])
