"""Install isolated Python media runtimes and build the native engines."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil
from .config import ROOT, path
from .journal import atomic_json
from .process import OwnedProcess


def python_path(root):
    return root / 'bin/python'

def media_profile(cfg, profile, *, check=False):
    from .media_profiles import PROFILES
    spec=PROFILES[profile]
    runtime = Path(cfg['runtime_root']) / spec['directory']
    binary = python_path(runtime)
    lock = ROOT / 'locks' / spec['lock']
    lock_sha = hashlib.sha256(lock.read_bytes()).hexdigest()
    marker = runtime / 'slstack-installed.json'
    install_cfg={**cfg,'host_budget_gib':min(3,cfg['host_budget_gib'])}
    ready = binary.is_file() and marker.is_file() and json.loads(marker.read_text())['lock_sha256'] == lock_sha
    if not check and not ready:
        destination=runtime
        while not destination.exists():destination=destination.parent
        if shutil.disk_usage(destination).free < 24 * 1024**3:
            raise RuntimeError('Media setup needs 24 GiB free for packages, download/build staging and headroom; choose a runtime on a larger volume')
        if not binary.exists():
            OwnedProcess(install_cfg, [sys.executable, '-B', '-m', 'venv', str(runtime)], 'install-venv', timeout=180).wait()
        OwnedProcess(install_cfg, [str(binary), '-B', '-m', 'pip', 'install', '--disable-pip-version-check', '-r', str(lock)],
                     'install-media', timeout=7200).wait()
        OwnedProcess(install_cfg, [str(binary), '-B', '-m', 'pip', 'check'], 'install-verify', timeout=120).wait()
        ready = True
    if binary.is_file():
        probe_cfg={**cfg,'host_budget_gib':min(3,cfg['host_budget_gib'])}
        probe=OwnedProcess(probe_cfg,[str(binary),'-B','-m','slstack.media_runtime',profile],'install-media-capabilities',timeout=180)
        try:
            failure=None
            try:probe.wait()
            except RuntimeError as exc:failure=exc
            rows=[json.loads(line.removeprefix('SLSTACK_MEDIA_RUNTIME ')) for line in probe.log.read_text(errors='replace').splitlines() if line.startswith('SLSTACK_MEDIA_RUNTIME ')]
            capabilities=rows[-1] if rows else dict(ready=False,missing=[dict(error='Media import verification did not finish; see '+str(probe.log))])
            ready=ready and capabilities['ready'] and failure is None
            if not ready and not check:raise RuntimeError('Media runtime verification failed: '+json.dumps(capabilities)) from failure
        finally:probe.stop()
    if not check and ready:
        atomic_json(marker, {'lock_sha256': lock_sha, 'python': str(binary), 'profile':profile,
                             'weights_downloaded': False,'capabilities_verified':True})
    return dict(profile=profile,ready=ready,python=str(binary),lock=spec['lock'],
                capabilities=capabilities if binary.is_file() else None)


def install(cfg, *, image=False, video=False, all_engines=False, check=False):
    if sys.version_info[:2] != (3, 12):
        raise ValueError('Install Python 3.12, then run the launcher with that interpreter')
    result = {'engines': []}
    if not (image or video or all_engines):all_engines=True
    from .media_assets import install as install_assets
    result['media_assets']=install_assets(cfg['runtime_root'],check=check)
    from .media_tools import install as install_tools
    result['video_tools']=install_tools(cfg['runtime_root'],check=check)
    profiles=['studio','qwen21'] if image or all_engines else ['studio']
    result['media_runtimes']=[media_profile(cfg,p,check=check) for p in profiles]
    ready=all(row['ready'] for row in result['media_runtimes']) and result['media_assets']['ready'] and result.get('video_tools',{'ready':True})['ready']
    runtime=Path(cfg['runtime_root'])
    binaries = [path(cfg['ds4_command'][0]), path(cfg['llama_producer']),
                path(cfg.get('weight_fetcher', 'build/llama/bin/slstack-weight-fetcher.so')), ROOT / 'build/llama/bin/llama-server',
                ROOT / 'build/llama/bin/llama-export-lora', path(cfg['llama_producer']).parent / 'ds4-decode.so']
    if not check and all_engines:
        # The documented install step prepares the native engines and UI too.
        # Builds never need model state or a GPU reservation.
        # Each guarded compiler step uses request_timeout_s; a cold CUDA build
        # exceeds the 900 s chat-request default, so give steps the build bound.
        build_cfg = {**cfg, 'host_budget_gib': min(3, cfg['host_budget_gib']),
                     'min_available_gib': max(4, cfg['min_available_gib']),
                     'request_timeout_s': 7200}
        build_config = Path(cfg['runtime_root']) / 'install-build.local.json'
        atomic_json(build_config, build_cfg)
        outer = {**build_cfg, 'host_budget_gib': build_cfg['host_budget_gib'] + .5}
        print('Building native engines and chat UI; progress is recorded in the runtime jobs directory.', flush=True)
        OwnedProcess(outer, ['sh', str(ROOT / 'build.sh'), str(build_config), 'all'],
                     'install-build', timeout=7200).wait()
    if all_engines:
        result['build'] = {'ready': all(p.is_file() for p in binaries)
                          and (path(cfg['webui_root']) / 'index.html').is_file()}
    for kind, enabled in (('image', image or all_engines), ('video', video or all_engines)):
        if enabled:
            result['engines'].append({'engine': kind, 'ready': ready, 'runtime': str(runtime),
                                      'note': 'Native weight-fetcher shared build is also required; flash-attn is an explicit optional prerequisite for H3 flash_attn2'})
    if not check:
        # The one-time full model SHA-256 runs here, not on the first prompt.
        from .check import verify_model_receipt
        receipt = verify_model_receipt(cfg)
        if receipt is not None:
            result['model_receipt'] = receipt
    return result
