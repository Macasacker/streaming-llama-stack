import importlib.util
import json
from pathlib import Path
import re
import struct
import tempfile
import unittest
from unittest.mock import patch
from slstack.config import ROOT
from slstack.backends import BACKENDS,command,commands
from slstack.native_broker import vendor

vendor()
from studio.general_launch import stream_arch_id,canonical_streaming_rtx,streaming_prefill_env


class Exports(unittest.TestCase):
    def test_all_backend_builders_and_real_receiver_options(self):
        receiver=(ROOT/'engines/streaming-llama/prefill_llama.cpp/tools/decode-receiver/main.cpp').read_text()
        allowed=set(re.findall(r'arg == "([^"]+)"',receiver))
        for name in BACKENDS:
            with self.subTest(backend=name):
                cfg=dict(backend=name,model_path='models/model',bind_address='127.0.0.1',fa='on',
                         binary='backend-binary',ctx=262144,mtp_mode='off')
                with tempfile.TemporaryDirectory() as temp:
                    argv=command(cfg,runtime=temp,python='decode-python',write_settings=False)
                    self.assertTrue(argv)
                    if name in ('llama-decode','llama-decode-upstream'):
                        self.assertTrue(set(x for x in argv[1:] if x.startswith('-'))<=allowed)
                        self.assertEqual(argv[argv.index('--kv-host')+1],'127.0.0.1')
                        self.assertEqual(argv[argv.index('--token-stream-host')+1],'127.0.0.1')
                        self.assertIn('--flash-attn',argv)
                    if name=='mtp-shim':self.assertEqual(len(commands(cfg,runtime=temp,python='decode-python')),2)

    def test_glm_family_and_flash_are_not_conflated(self):
        for name in ('GLM-5.1','GLM-5.2','GLM-5.3','GLM5.1','GLM5.2','GLM5.3'):
            self.assertEqual(stream_arch_id(name),'glm-dsa')
        self.assertEqual(stream_arch_id('GLM-5.3-Flash'),'glm5-next')
        self.assertEqual(stream_arch_id('', 'models/GLM-5.3-Flash-Q4.gguf'),'glm5-next')
        cfg=canonical_streaming_rtx(dict(arch='glm-dsa',fa='on',ubatch=2048,wf_gpu_bufs=2,dsa_sparse=True))
        self.assertEqual((cfg['fa'],cfg['ubatch']),('off',512))
        env=streaming_prefill_env(cfg)
        self.assertEqual(env['LLAMA_MLA_SPARSE'],'1')
        self.assertEqual(env['LLAMA_DSA_SCORE_CUBLAS'],'1')



    def test_omlx_never_imports_global_settings(self):
        from studio.backend_argv import omlx_write_settings
        with tempfile.TemporaryDirectory() as temp:
            with patch('studio.backend_argv.os.path.expanduser',side_effect=AssertionError('unexpected global settings read')):
                target=omlx_write_settings(temp,262144)
            self.assertEqual(json.loads(Path(target).read_text())['sampling']['max_context_window_policy'],262144)

    def test_decode_preflight_checks_paths_for_the_linux_backends(self):
        from slstack.decode import inspect,settings
        with tempfile.TemporaryDirectory() as temp:
            model=Path(temp)/'model';model.mkdir()
            cfg=dict(runtime_root=temp,decode=dict(backend='ds4',model_path=str(model)))
            result=inspect(cfg)
            self.assertEqual(result['backend'],'ds4')
            cfg['decode']['binary']=str(Path(temp)/'not-built-yet')
            self.assertFalse(inspect(cfg)['ready'])
            self.assertTrue(any('not-built-yet' in item for item in inspect(cfg)['missing']))
            self.assertEqual(settings(cfg,selected='llama-server')['backend'],'llama-server')
            with self.assertRaises(ValueError):settings(dict(runtime_root=temp))
            with self.assertRaises(ValueError):settings(dict(runtime_root=temp,model=str(model),selected='mlx-receiver'))
            cfg['decode'].update(backend='llama-server',binary=__import__('sys').executable)
            self.assertTrue(inspect(cfg)['ready'])

    def test_linux_engine_sources_and_provenance_survive_export(self):
        spec=importlib.util.spec_from_file_location('engine_audit',ROOT/'scripts/audit_backend_sources.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        result=module.audit()
        self.assertEqual([row['engine'] for row in result['engines']],['ds4','streaming-llama'])
        self.assertTrue(result['all_retained_components_complete'],result)
        self.assertTrue((ROOT/'engines/ds4/PROVENANCE.md').is_file())
        self.assertTrue((ROOT/'engines/ds4/THIRD_PARTY_NOTICES.md').is_file())

if __name__=='__main__':unittest.main()
