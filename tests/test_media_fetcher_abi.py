"""The media weight transport's ctypes mirror must match the fetcher header it binds.

`vendor/engines/cosmos_windows/weight_streaming/ds_backend.py` declares `wf_params`,
`wf_weight_span` and `wf_stats` field by field, in header order, because ctypes hands the
struct to `wf_create` as raw memory: a field the C header no longer has shifts every later
one, and the fetcher reads an integer where it expects the GPU backend pointer. Chunk 3.1e
deleted the three Windows relay fields, and this check is what notices the next time.

The library arm reads the built shared library and asserts what it exports, so the refusal
the transport raises on this platform is the documented one rather than an attribute error.
No GPU, no torch, no model.
"""
import ctypes
import re
import subprocess
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / 'engines/streaming-llama/prefill_llama.cpp/src/llama-weight-fetcher.h'
LIBRARY = ROOT / 'build/llama/bin/libslstack-weight-fetcher.so'
CTYPES = {'const char *': lambda: ctypes.c_char_p, 'const char **': lambda: ctypes.POINTER(ctypes.c_char_p),
          'size_t': lambda: ctypes.c_size_t, 'int': lambda: ctypes.c_int, 'bool': lambda: ctypes.c_bool,
          'void *': lambda: ctypes.c_void_p, 'uint64_t': lambda: ctypes.c_uint64,
          'float': lambda: ctypes.c_float, 'wf_transfer_mode': lambda: ctypes.c_int,
          'wf_moe_mode': lambda: ctypes.c_int, 'wf_post_prefill_mode': lambda: ctypes.c_int}


def ds_backend():
    if str(ROOT / 'vendor/engines') not in sys.path:
        sys.path.insert(0, str(ROOT / 'vendor/engines'))
    from cosmos_windows.weight_streaming import ds_backend
    return ds_backend


def header_source():
    return HEADER.read_text(encoding='utf-8')


def struct_fields(source, name):
    """The declared field list of `} <name>;`, as (field, ctypes type) pairs in header order."""
    body = re.search(r'typedef struct \{([^{}]*)\}\s*' + name + ';', source, re.S)
    if not body:
        raise AssertionError('no ' + name + ' struct in ' + str(HEADER))
    fields = []
    for line in body.group(1).splitlines():
        line = re.sub(r'//.*', '', line).strip()
        if not line or line.startswith('#'):
            continue
        parts = line.rstrip(';').rsplit(None, 1)
        if len(parts) != 2:
            continue
        type_name, field = parts[0].strip(), parts[1].strip()
        stars = type_name.count('*') + field.count('*')
        base = type_name.replace('*', '').strip()
        type_name = base + ' ' + '*' * stars if stars else base
        field = field.replace('*', '').strip()
        if type_name not in CTYPES:
            raise AssertionError('unmapped header type ' + type_name + ' in ' + name)
        fields.append((field, CTYPES[type_name]()))
    return fields


def python_fields(struct):
    return list(struct._fields_)


class FetcherAbiMirror(unittest.TestCase):
    def test_every_struct_mirrors_the_header_field_for_field(self):
        module, source = ds_backend(), header_source()
        for name, struct in (('wf_params', module.wf_params), ('wf_weight_span', module.wf_weight_span),
                             ('wf_stats', module.wf_stats)):
            self.assertEqual(struct_fields(source, name), python_fields(struct),
                             name + ' no longer mirrors ' + str(HEADER))

    def test_the_transport_asks_for_a_mode_the_fetcher_can_give(self):
        # The header retires mode 3, so a mirror that still offered it would ask the fetcher
        # for a transport it folds away, and every caller's requirement would then fail.
        module, source = ds_backend(), header_source()
        self.assertFalse(hasattr(module, 'WF_MODE_DSTORAGE'))
        declared = dict(re.findall(r'(WF_MODE_[A-Z_]+)\s*=\s*(\d+)', source))
        self.assertIn('WF_MODE_DSTORAGE', declared)
        for name, value in declared.items():
            if name == 'WF_MODE_DSTORAGE':
                continue
            self.assertEqual(int(value), getattr(module, name), name + ' moved')

    def test_streaming_is_the_relay_and_neither_mmap_nor_the_retired_mode(self):
        module = ds_backend()
        for mode in (module.WF_MODE_DIRECT_IO, module.WF_MODE_GDS):
            self.assertTrue(module.is_streaming_mode(mode))
        for mode in (module.WF_MODE_MMAP, module.WF_MODE_AUTO, 3, None):
            self.assertFalse(module.is_streaming_mode(mode))

    def test_the_symbols_the_transport_requires_are_declared_by_the_header(self):
        module, source = ds_backend(), header_source()
        declared = set(re.findall(r'\b(wf_[a-z_0-9]+)\s*\(', source))
        missing = set(module.REQUIRED_WF_SYMBOLS) - declared
        self.assertEqual(set(), missing, 'the transport requires a fetcher function the header '
                                         'does not declare')

    def test_no_media_transport_still_requires_the_retired_mode(self):
        offenders = []
        for path in sorted((ROOT / 'vendor/engines/cosmos_windows').rglob('*.py')):
            if 'domainshuttle_vendor' in path.parts:
                continue
            text = path.read_text(encoding='utf-8')
            if re.search(r"mode[^=\n]*[!=]= *3\b", text) or 'WF_MODE_DSTORAGE' in text:
                offenders.append(path.relative_to(ROOT).as_posix())
        self.assertEqual([], offenders)


class FetcherLibrarySurface(unittest.TestCase):
    """What the built shared library exports, and whether the transport agrees with it.

    The export table is read with `nm`, not by loading the library: this one is linked
    against cuFile, CUDA runtime and liburing, and dlopening it inside the test process
    changes what the cgroup and memory checks in other files measure.
    """

    @classmethod
    def setUpClass(cls):
        if not LIBRARY.is_file():
            raise unittest.SkipTest('build the llama tree first (build.sh llama)')
        cls.module = ds_backend()
        listed = subprocess.run(['nm', '-D', '--defined-only', str(LIBRARY)],
                                stdout=subprocess.PIPE, text=True)
        if listed.returncode:
            raise unittest.SkipTest('nm could not read ' + str(LIBRARY))
        cls.exported = {line.split()[-1] for line in listed.stdout.splitlines() if line.split()}
        cls.missing = [name for name in cls.module.REQUIRED_WF_SYMBOLS if name not in cls.exported]

    def test_a_process_that_binds_the_library_gets_the_documented_answer(self):
        # `load_wf` has to open the library to read its symbols, and that library is linked
        # against cuFile, the CUDA runtime and liburing - so the binding is proved in a child
        # process, never inside this one.
        child = subprocess.run(
            [sys.executable, '-B', '-c',
             "import sys; sys.path.insert(0, %r); "
             "from cosmos_windows.weight_streaming.ds_backend import load_wf; "
             "lib = load_wf(%r); print('BOUND')" % (str(ROOT / 'vendor/engines'), str(LIBRARY))],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if self.missing:
            self.assertNotEqual(0, child.returncode)
            self.assertIn('does not export the weight-fetcher API', child.stderr)
            for name in self.missing[:4]:
                self.assertIn(name, child.stderr)
        else:
            self.assertEqual(0, child.returncode)
            self.assertIn('BOUND', child.stdout)

    def test_the_export_table_is_complete_or_absent_never_partial(self):
        # A library that exports some but not all of the API is a stale build, and binding it
        # would fail inside a worker rather than at the door.
        exported = len(self.module.REQUIRED_WF_SYMBOLS) - len(self.missing)
        self.assertIn(exported, (0, len(self.module.REQUIRED_WF_SYMBOLS)),
                      f'{LIBRARY} exports {exported} of {len(self.module.REQUIRED_WF_SYMBOLS)} '
                      'required fetcher symbols; rebuild the llama tree')
        print('SLSTACK_MEDIA_FETCHER_ABI ' + f'library={LIBRARY.name} exported={exported} '
              f'of {len(self.module.REQUIRED_WF_SYMBOLS)}', flush=True)

    def test_an_unconfigured_library_is_a_value_error(self):
        with self.assertRaises(ValueError):
            self.module.load_wf('')


if __name__ == '__main__':
    unittest.main()
