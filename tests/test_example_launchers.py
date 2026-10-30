"""Example dry runs must be safe and preserve the selected execution mode."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('examples_launcher',ROOT/'examples/run_preset.py')
launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)


class Launchers(unittest.TestCase):

    def test_all_single_machine_dry_runs_are_read_only(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime=Path(folder)/'must-not-be-created'
            local=Path(folder)/'test.local.json'
            local.write_text(json.dumps(dict(model='model.gguf',runtime_root=str(runtime),webui=True,
                                            producer_profile='accurate',yoco_keep=140,api_key='example-placeholder')))
            for profile in ('accurate','fast','ultra','ultra-cone','ultra-yoco'):
                for context in ('262k','1m'):
                    with self.subTest(profile=profile,context=context):
                        run=subprocess.run([sys.executable,'-B',str(ROOT/'examples/run_preset.py'),
                                            '--single',profile,'--context',context,'--config',str(local),'--dry-run','--webui'],
                                           capture_output=True,text=True,timeout=30,cwd=ROOT)
                        self.assertEqual(run.returncode,0,run.stderr)
                        result=json.loads(run.stdout);cfg=result['config']
                        self.assertEqual(cfg['producer_profile'],'ultra' if profile in ('ultra-cone','ultra-yoco') else profile)
                        self.assertEqual(cfg['yoco_keep'],{'ultra-cone':2541,'ultra-yoco':140}.get(profile,0))
                        self.assertEqual(cfg['env_overrides']['LLAMADECODE_APPEND_DECODE_MAX'],'0')
                        self.assertEqual(cfg['webui'],True)
                        self.assertNotIn('example-placeholder',run.stdout)
                        self.assertFalse(runtime.exists())

    def test_single_machine_ultra_cone_dry_run_keeps_the_exact_cone(self):
        with tempfile.TemporaryDirectory() as folder:
            local=Path(folder)/'test.local.json';local.write_text('{}')
            run=subprocess.run([sys.executable,'-B',str(ROOT/'examples/run_preset.py'),
                                '--single','ultra-cone','--context','1m','--config',str(local),'--dry-run'],
                               capture_output=True,text=True,timeout=30,cwd=ROOT)
            self.assertEqual(run.returncode,0,run.stderr)
            cfg=json.loads(run.stdout)['config']
            self.assertEqual((cfg['producer_profile'],cfg['yoco_keep']),('ultra',2541))
            self.assertEqual(cfg['env_overrides'].get('LLAMADECODE_SINGLEPASS_VRAM'),'1')

if __name__=='__main__':unittest.main()
