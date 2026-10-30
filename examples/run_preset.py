"""Portable example launcher for one Linux machine. Local paths and credentials come from --config."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def prepare(args, local):
    spec = importlib.util.spec_from_file_location('single_preset', ROOT / 'scripts/dsv41_single.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.overlay(args.context, local, profile=args.single)
    result['webui'] = bool(args.webui)
    return result, 'start'


def missing_settings(value, prefix=''):
    if isinstance(value, dict):
        return [name for key, item in value.items()
                for name in missing_settings(item, prefix + '.' + key if prefix else key)]
    if isinstance(value, str) and value.startswith('SET_'):
        return [prefix]
    return []


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--single', choices=('accurate', 'fast', 'ultra', 'ultra-yoco', 'ultra-cone'), required=True)
    parser.add_argument('--context', choices=('262k', '1m'), default='262k')
    parser.add_argument('--webui', action='store_true')
    parser.add_argument('--config', required=True, help='Your private config; never a file containing model weights')
    parser.add_argument('--dry-run', action='store_true', help='Show the generated config; start nothing and write nothing')
    parser.add_argument('--setup', action='store_true', help='Build a native engine (ds4 or llama-server)')
    parser.add_argument('--backend', choices=('ds4', 'llama-server'), default='ds4')
    parser.add_argument('--cache-dir', help='Pinned source cache directory')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--build-only', action='store_true', help='Compile without GPU verification')
    args = parser.parse_args(argv)
    local = json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
    cfg, command = prepare(args, local)
    if args.setup:
        command = 'setup-backend'
    if (args.cache_dir or args.offline or args.build_only) and command != 'setup-backend':
        parser.error('--cache-dir, --offline and --build-only apply to --setup')
    missing = missing_settings(cfg)
    if args.dry_run:
        from slstack.envvars import redact_cfg
        print(json.dumps(dict(command=command, config=redact_cfg(cfg), missing_local_settings=missing), indent=2))
        return 0
    if missing:
        parser.error('Set these fields in your private config: ' + ', '.join(missing))
    runtime = Path(cfg.get('runtime_root', 'runtime')).expanduser()
    runtime = (runtime if runtime.is_absolute() else ROOT / runtime).resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    effective = runtime / ('example-' + args.single + '.local.json')
    effective.write_text(json.dumps(cfg, indent=2) + '\n')
    argv = [sys.executable, '-B', str(ROOT / 'scripts/bootstrap.py'), '--config', str(effective), command]
    if command == 'setup-backend':
        argv.extend(['--backend', args.backend])
    if args.cache_dir:
        argv.extend(['--cache-dir', str(Path(args.cache_dir).resolve())])
    if args.offline:
        argv.append('--offline')
    if args.build_only:
        argv.append('--build-only')
    return subprocess.call(argv, cwd=ROOT)


if __name__ == '__main__':
    raise SystemExit(main())
