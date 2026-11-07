"""Chunk 7.1d: the inventory must see a read through a wrapper, and must not count a
read that the wrapper discards. The helpers are proved on synthetic source, and the
tree is asked what it actually holds."""
import unittest

from slstack.envvars import (READ, discarded_alias_reads, inventory, rows, _env_wrappers,
                             _literal_reads, _newline_offsets, _source_texts, _strip_comments, _wrapper_pattern,
                             _wrapper_reads)

ALIAS = '''
static bool glm_graph_env_present(const char *rocm_name, const char *cuda_name) {
    (void)rocm_name;
    return cuda_name && getenv(cuda_name) != NULL;
}

static bool stream_readahead_enabled(void) {
    return glm_graph_env_present("DS4_ROCM_ENABLE_STREAMING_READAHEAD",
                                 "DS4_METAL_ENABLE_STREAMING_READAHEAD");
}
'''

TERNARY = '''
static bool queue_step(void *g) {
    return !getenv(((int)g == 2 ? "DS4_METAL_DISABLE_V41_TP_DECODE_QUEUE"
                                : "DS4_METAL_DISABLE_V41_DECODE_QUEUE"));
}
'''

FORWARD = '''
static int cli_env_flag_enabled(const char *name, bool defval) {
    const char *v = getenv(name);
    return v && v[0] ? v[0] != '0' : defval;
}

static bool greedy_splitkv(void) {
    if (cli_env_flag_enabled("DS4_CUDA_NO_SPLITKV_SPEC", false)) return false;
    return cli_env_flag_enabled("DS4_CUDA_SPLITKV_SPEC", false);
}
'''

PROTOTYPE = '''
static long long cache_bytes_from_env(const char *name, long long fallback);

static long long cache_bytes_from_env(const char *name, long long fallback) {
    const char *value = getenv(name);
    return value ? value[0] : fallback;
}

static void use(void) { cache_bytes_from_env("DS4_KV_TEST_BYTES", 8); }
'''


def scan(text):
    """What the inventory would report for one synthetic file: the net name set, the
    wrapper reads, the discarded alias reads and the wrapper registry."""
    src = _strip_comments(text)
    texts = {'synthetic.c': (text, src)}
    wrappers, definitions = _env_wrappers(texts)
    newlines = _newline_offsets(text)
    literal, confirmed = set(), set()
    for name, _line, direct in _literal_reads(text, src, newlines):
        (confirmed if direct else literal).add(name)
    wrapped, dropped, _ = _wrapper_reads(text, src, wrappers, definitions, 'synthetic.c',
                                         _wrapper_pattern(wrappers), newlines)
    net = literal | confirmed | {n for n, _ in wrapped}
    dropped_only = {n for n, _ in dropped} - confirmed - {n for n, _ in wrapped}
    return net - dropped_only, {n for n, _ in wrapped}, dropped_only, wrappers


def line_bound(text):
    """The rule the inventory used to have: one regex per physical line. A call whose
    argument is on the next line is invisible to it, which is the bug chunk 7.1d fixed.
    Asserting this returns nothing is only evidence because the same text, scanned over
    the file, returns names."""
    found = set()
    for line in text.splitlines():
        for match in READ.finditer(line):
            found.add(match.group(1) or match.group(2))
    return found


class HelperTests(unittest.TestCase):
    def test_wrapper_argument_is_found_and_its_discarded_alias_is_reported_not_read(self):
        net, wrapped, dropped, wrappers = scan(ALIAS)
        self.assertEqual(wrappers['glm_graph_env_present'], (1,))
        self.assertIn('DS4_METAL_ENABLE_STREAMING_READAHEAD', wrapped)
        self.assertIn('DS4_ROCM_ENABLE_STREAMING_READAHEAD', dropped)
        self.assertEqual(net, {'DS4_METAL_ENABLE_STREAMING_READAHEAD'})
        # The old line-bound regex saw the *wrong* one of the pair: the discarded alias
        # is on the same line as the call, the name that reaches getenv is not.
        self.assertEqual(line_bound(ALIAS), {'DS4_ROCM_ENABLE_STREAMING_READAHEAD'})
        self.assertEqual(line_bound(TERNARY), set())

    def test_both_names_inside_a_conditional_getenv_are_reads(self):
        net, wrapped, dropped, _ = scan(TERNARY)
        self.assertEqual(net, {'DS4_METAL_DISABLE_V41_TP_DECODE_QUEUE',
                               'DS4_METAL_DISABLE_V41_DECODE_QUEUE'})
        self.assertEqual(dropped, set())
        self.assertEqual(wrapped, set())

    def test_a_forwarding_helper_qualifies_as_a_wrapper(self):
        net, wrapped, dropped, wrappers = scan(FORWARD)
        self.assertEqual(wrappers['cli_env_flag_enabled'], (0,))
        self.assertEqual(wrapped, {'DS4_CUDA_NO_SPLITKV_SPEC', 'DS4_CUDA_SPLITKV_SPEC'})
        self.assertEqual(net, {'DS4_CUDA_NO_SPLITKV_SPEC', 'DS4_CUDA_SPLITKV_SPEC'})
        self.assertEqual(dropped, set())

    def test_a_prototype_is_not_a_call_and_a_real_call_still_counts(self):
        net, wrapped, dropped, _ = scan(PROTOTYPE)
        self.assertEqual(wrapped, {'DS4_KV_TEST_BYTES'})
        self.assertEqual(net, {'DS4_KV_TEST_BYTES'})
        self.assertEqual(dropped, set())

    def test_a_read_through_a_variable_is_reported_for_a_human(self):
        text = '''
static void set_env_value(const char *name, const char *value) { (void)value; getenv(name); }
static const char *const names[] = {"DS4_A", "DS4_B"};
static void use(void) { set_env_value(names[0], "1"); }
'''
        src = _strip_comments(text)
        texts = {'synthetic.c': (text, src)}
        wrappers, definitions = _env_wrappers(texts)
        _hit, _dropped, miss = _wrapper_reads(text, src, wrappers, definitions, 'synthetic.c',
                                              _wrapper_pattern(wrappers), _newline_offsets(text))
        self.assertEqual([(name, arg) for name, _, arg in miss], [('set_env_value', 'names[0]')])


class TreeTests(unittest.TestCase):
    """The claims chunk 7.1d makes about the tree, checked against the tree."""

    @classmethod
    def setUpClass(cls):
        cls.reads = inventory()

    def test_a_wrapper_only_knob_is_in_the_inventory(self):
        # Read as the second argument of glm_graph_env_present(), which is the only
        # argument that reaches getenv().
        self.assertIn('DS4_METAL_DISABLE_STREAMING_READAHEAD', self.reads)
        self.assertIn('DS4_METAL_STREAMING_PREFILL_LAYER_PREPARE_THREADS', self.reads)

    def test_a_conditional_getenv_names_both_knobs(self):
        self.assertIn('DS4_METAL_DISABLE_V41_TP_DECODE_QUEUE', self.reads)
        self.assertIn('DS4_METAL_DISABLE_V41_DECODE_QUEUE', self.reads)

    def test_a_discarded_alias_is_not_a_read_even_though_the_source_names_it(self):
        self.assertNotIn('DS4_ROCM_DISABLE_STREAMING_READAHEAD', self.reads)
        self.assertIn('DS4_ROCM_DISABLE_STREAMING_READAHEAD', discarded_alias_reads())

    def test_every_read_is_in_a_scanned_tree(self):
        for name, references in self.reads.items():
            with self.subTest(name=name):
                for reference in references:
                    path, _, line = reference.rpartition(':')
                    self.assertTrue(path.startswith('engines/') or path.startswith('vendor/'), reference)
                    self.assertGreater(int(line), 0)

    def test_no_row_keeps_a_read_the_wrapper_discards(self):
        # 277 rows have no read at all and stay: those are launcher-owned names the
        # scanned trees never read. A row must not keep a position that the wrapper
        # ignores, which is the alias class chunk 7.1d removed.
        dropped = discarded_alias_reads()
        for row in rows():
            for reference in row['reads']:
                with self.subTest(var=row['var'], read=reference):
                    self.assertNotIn(reference, dropped.get(row['var'], []))

    def test_the_table_covers_every_read(self):
        documented = {row['var'] for row in rows()}
        self.assertEqual(sorted(set(self.reads) - documented), [])

    def test_the_old_line_bound_rule_found_nothing_that_is_now_unexplained(self):
        # The rule chunk 7.1d replaced, run over the tree: every name it saw must still
        # be a read, or be an alias the wrapper discards - the class it documented by
        # mistake. Anything else would be a name the new passes lost.
        seen = set()
        for text, _src in _source_texts().values():
            for line in text.splitlines():
                for match in READ.finditer(line):
                    seen.add(match.group(1) or match.group(2))
        unexplained = sorted(seen - set(self.reads) - set(discarded_alias_reads()))
        self.assertEqual(unexplained, [])


if __name__ == '__main__':
    unittest.main()
