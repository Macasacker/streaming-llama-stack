"""Reproducible backend builds in private runtime directories.

Linux only. Two backends are installable here: `ds4` (the CUDA engine in `engines/ds4`) and
`llama-server` (the llama.cpp fork in `engines/streaming-llama`). The macOS MLX receivers, the
Swift mference server, the Zig mlx-serve backend, oMLX, and the cross-machine peer were deleted
with the platform they ran on.
"""
from __future__ import annotations
import hashlib
import json
import os
import platform
import shutil
import sys
from pathlib import Path
from .backend_cache import Cache,digest,snapshot,tree_files
from .backends import backend
from .config import ROOT
from .journal import atomic_json
from .process import OwnedProcess


class Build:
    def __init__(self,cfg,*,cache_dir=None,offline=False,wheelhouse=None,jobs=2):
        if jobs not in (1,2):raise ValueError('Backend builds support one or two compiler jobs')
        self.cfg=cfg;self.runtime=Path(cfg['runtime_root']);self.runtime.mkdir(parents=True,exist_ok=True)
        self.cache=Cache(cfg,directory=cache_dir,offline=offline);self.offline=offline;self.jobs=jobs
        self.wheelhouse=Path(wheelhouse).expanduser().resolve() if wheelhouse else None
        if self.wheelhouse and not self.wheelhouse.is_dir():raise ValueError('wheelhouse must be a directory')
        self.pip_options=['--no-index','--find-links',str(self.wheelhouse)] if self.wheelhouse else []
        if offline and not self.wheelhouse:raise ValueError('Offline builds require a wheelhouse for Python build tools')
        self.launch={**cfg,'host_budget_gib':cfg.get('backend_build_budget_gib',3),
                     'min_available_gib':max(4,cfg.get('min_available_gib',4))}
        self.python=self.runtime/'backend-build-venv'/'bin/python'
        self.steps=[];self.retained_inputs={}

    def run(self,argv,name,*,cwd=ROOT):
        job=OwnedProcess(self.launch,[str(v) for v in argv],'backend-'+name,cwd=cwd,timeout=3600)
        self.steps.append(dict(name=name,log=str(job.log),receipt=str(job.receipt)))
        job.wait();return job.log.read_text(errors='replace')

    def tools(self):
        if not self.python.is_file():self.run([sys.executable,'-B','-m','venv',self.python.parent.parent],'build-venv')
        cmake=self.python.with_name('cmake')
        if not cmake.is_file():
            self.run([self.python,'-B','-m','pip','install','--disable-pip-version-check',*self.pip_options,
                      'cmake==4.4.4','setuptools==84.2.1','wheel==0.48.0'],'build-tools')
        return str(cmake)

    def source(self,name):
        return self.cache.unpack(name,self.runtime/'downloads/unpacked')

    def stage(self,source,label,*,transform=None):
        source=Path(source)
        if any(source.is_relative_to(ROOT/name) for name in ('engines','vendor')):
            from .backend_cache import fingerprint
            self.retained_inputs[source.relative_to(ROOT).as_posix()]=fingerprint(tree_files(source))
        recipe=hashlib.sha256()
        for file in ('backend_setup.py','backend_cache.py'):
            recipe.update((ROOT/'slstack'/file).read_bytes().replace(b'\r\n',b'\n'))
        receipt=json.dumps(dict(recipe_sha256=recipe.hexdigest(),platform=sys.platform,
                               machine=platform.machine(),python=list(sys.version_info[:2]))).encode()
        def stage_files(files):
            return [*(transform(files) if transform else files),('STACK_BUILD_RECIPE.json',receipt)]
        return snapshot(source,self.runtime,label,transform=stage_files)

    def cmake(self,source,build,options,targets=None,*,install=False):
        cmake=self.tools()
        self.run([cmake,'-S',source,'-B',build,'-DCMAKE_BUILD_TYPE=Release',*options],build.name+'-configure')
        command=[cmake,'--build',build,'--parallel',self.jobs]
        if targets:command+=['--target',*targets]
        self.run(command,build.name+'-compile')
        if install:self.run([cmake,'--install',build],build.name+'-install')

    def report(self,name,revision,binaries,**extra):
        missing=[str(p) for p in binaries if not Path(p).is_file()]
        if missing:raise RuntimeError('Backend build outputs are missing: '+', '.join(missing))
        result=dict(backend=name,built=True,ready=False,verification_pending=True,source_sha256=revision,
                    binaries={str(p):digest(p) for p in binaries},steps=self.steps,
                    retained_inputs=self.retained_inputs,**extra)
        atomic_json(self.runtime/'state/backends'/(name+'.json'),result)
        return result


def cuda_arch():
    """The GPU this machine reports, as a Makefile arch. Falls back to sm_120a."""
    import re,subprocess
    try:
        text=subprocess.check_output(['nvidia-smi','--query-gpu=compute_cap','--format=csv,noheader'],
                                     text=True,timeout=30).splitlines()[0].strip()
        return 'sm_%s%s' % (text.replace('.', ''), 'a')
    except Exception:
        return os.environ.get('DS4_CUDA_ARCH', 'sm_120a')


def native_ds4(build,name='ds4'):
    source,revision=build.stage(ROOT/'engines'/name,name)
    build.run(['make','-j'+str(build.jobs),'cuda','CUDA_ARCH='+cuda_arch()],'ds4-compile',cwd=source)
    binary=source/'ds4-server'
    return build.report(name,revision,[binary],binary=str(binary),
                        renderer_sha256=digest(ROOT/'engines'/name/'ds4_server.c'))


def native_llama(build,name='llama-server'):
    source,revision=build.stage(ROOT/'engines/streaming-llama/prefill_llama.cpp',name)
    output=build.runtime/'build'/('llama-'+revision[:16])
    options=['-DLLAMA_BUILD_TESTS=OFF','-DLLAMA_BUILD_EXAMPLES=OFF','-DLLAMA_BUILD_TOOLS=ON',
             '-DLLAMA_BUILD_SERVER=ON','-DLLAMA_CURL=OFF','-DLLAMA_OPENSSL=ON','-DBUILD_SHARED_LIBS=OFF',
             '-DGGML_CUDA=ON']
    build.cmake(source,output,options,['llama-server'])
    binary=output/'bin'/'llama-server'
    return build.report(name,revision,[binary],binary=str(binary))


def inputs_for(name):
    return []


def check(cfg,name):
    path=Path(cfg['runtime_root'])/'state/backends'/(name+'.json')
    if not path.is_file():return dict(backend=name,built=False,ready=False,reason='Run setup-backend for this backend')
    result=json.loads(path.read_text())
    from .backend_cache import fingerprint
    artifacts=result.get('binaries',{})
    result['built']=bool(artifacts) and all(Path(p).is_file() and digest(p)==sha for p,sha in artifacts.items())
    result['built']=result['built'] and all(fingerprint(tree_files(ROOT/p))==sha for p,sha in result.get('retained_inputs',{}).items())
    result['ready']=result['built'] and not result.get('verification_pending',True)
    return result


def setup(cfg,name,*,check_only=False,fetch_only=False,build_only=False,cache_dir=None,offline=False,wheelhouse=None,jobs=2):
    name=backend(name).name
    if check_only:return check(cfg,name)
    if fetch_only:return dict(backend=name,downloaded=[],built=False,ready=False)
    if sys.platform!='linux':raise ValueError('Engine setup is Linux-only in this fork')
    volume=Path(cfg['runtime_root']).resolve()
    while not volume.exists():volume=volume.parent
    if shutil.disk_usage(volume).free<16*1024**3:raise OSError('Backend setup requires 16 GiB free for inputs, build outputs and safety headroom')
    build=Build({**cfg,'backend_build_budget_gib':cfg.get('backend_build_budget_gib',3)},
                cache_dir=cache_dir,offline=offline,wheelhouse=wheelhouse,jobs=jobs)
    result=native_ds4(build,name) if name=='ds4' else native_llama(build,name)
    if not build_only:
        binary=result.get('binary')
        if binary:build.run([binary,'--help'],name+'-cli-check')
        result.update(ready=True,verification_pending=False)
        atomic_json(build.runtime/'state/backends'/(name+'.json'),result)
    return result
