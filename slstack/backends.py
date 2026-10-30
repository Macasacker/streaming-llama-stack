"""Decoder commands and transport capabilities for the Linux engine surface.

Two decode backends survive: `ds4`, the native CUDA engine, and `llama-server`, the llama.cpp
fork in `engines/streaming-llama`. The MLX receivers, mlx-serve, oMLX, mtp-shim, mference and the
cross-machine peer profile were deleted with macOS.
"""
from __future__ import annotations
from dataclasses import dataclass
import ipaddress
from pathlib import Path
from .config import ROOT
from .native_broker import vendor


@dataclass(frozen=True)
class Backend:
    name: str
    transport: str
    setup: str


BACKENDS = {name: Backend(name, transport, setup) for name, transport, setup in (
    ('ds4', 'ds4-native', 'ds4'),
    ('llama-server', 'openai', 'llama'),
)}
ALIASES = {'dwarfstar': 'ds4', 'ds4-server': 'ds4', 'llama.cpp': 'llama-server',
           'remote-llama': 'llama-server', 'llama-upstream': 'llama-server'}


def backend(value):
    name=ALIASES.get(str(value).lower(),str(value).lower())
    if name not in BACKENDS:raise ValueError('Unknown decode backend; choose '+', '.join(BACKENDS))
    return BACKENDS[name]


def bind_address(value):
    address=ipaddress.ip_address(str(value))
    if address.is_unspecified or address.is_multicast:raise ValueError('Choose a specific local interface address')
    return str(address)


def python_for(runtime,profile='decode'):
    return str(Path(runtime)/(profile+'-venv')/'bin/python')


def installed_binary(runtime,name,fallback):
    receipt=Path(runtime)/'state/backends'/(name+'.json')
    if receipt.is_file():
        from .backend_setup import check
        result=check({'runtime_root':str(runtime)},name)
        if not result.get('ready'):
            raise ValueError('Backend installation is unverified or changed; run setup-backend --backend '+name)
        value=result.get('binary')
        if value:return value
    return str(fallback)


def capabilities(value):
    spec=backend(value)
    return dict(backend=spec.name,transport=spec.transport,rtx_prefill=spec.transport!='openai',
                openai=spec.transport=='openai',setup=spec.setup)


def command(config,*,runtime,python=None,write_settings=True):
    """Build on the serving host. Model paths and binaries stay in local config."""
    vendor()
    spec=backend(config.get('backend','ds4'))
    model=str(config.get('model_path') or '')
    if not model:raise ValueError('Set the decode model_path in your local config')
    host=bind_address(config.get('bind_address','127.0.0.1'))
    port=int(config.get('port',8181));context=int(config.get('ctx',262144))
    if not 1<=port<=65535 or not 1<=context<=1048576:raise ValueError('Invalid decoder port or context')
    if spec.name=='ds4':
        from studio.mac_argv import ds4_server_argv
        return ds4_server_argv(model,binary=str(config.get('binary') or
            installed_binary(runtime,spec.name,ROOT/'engines/ds4/ds4-server')),
            chdir=str(ROOT/'engines/ds4'),port=port,ctx=context,dspark=bool(config.get('dspark',False)),
            mtp_model=config.get('mtp_model'),mtp_draft=config.get('mtp_draft'),batched_session=1)
    binary=str(config.get('binary') or installed_binary(runtime,spec.name,ROOT/'build/llama/bin/llama-server'))
    argv=[binary,'-m',model,'--host',host,'--port',str(port),'-c',str(context),
          '-ngl',str(config.get('ngl',99)),'--jinja']
    from studio.preset_samplers import launch_args
    argv+=launch_args(config)
    if config.get('jinja_file'):argv.extend(('--chat-template-file',str(config['jinja_file'])))
    for key,flag in (('batch','-b'),('ubatch','-ub'),('ctk','-ctk'),('ctv','-ctv'),('fa','-fa')):
        if config.get(key) is not None:argv.extend((flag,str(config[key])))
    return argv


def commands(config,*,runtime,python=None,write_settings=True):
    return [command(config,runtime=runtime,python=python,write_settings=write_settings)]
