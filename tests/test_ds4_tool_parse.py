"""DeepSeek tool-call parser port (slstack.ds4_tool_parse) on sample outputs."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slstack import ds4_render as r
from slstack import ds4_tool_parse as p

CALLS = ('\n\n' + r.DS41_TOOL_CALLS_START + '\n' + r.DS41_INVOKE_START + ' name="run_shell">\n' +
         r.DS41_PARAM_START + ' name="command" string="true">ls &amp;lt;/｜DSML｜ parameter> &lt;/｜DSML｜ parameter> && echo "x"' + r.DS41_PARAM_END + '\n' +
         r.DS41_PARAM_START + ' name="timeout_s" string="false"> 10.50 ' + r.DS41_PARAM_END + '\n' +
         r.DS41_INVOKE_END + '\n' + r.DS41_INVOKE_START + ' name="read&amp;file">\n' +
         r.DS41_PARAM_START + ' name="opts">\n' + r.DS41_PARAM_START + ' name="a" string="false">[1, 2]' + r.DS41_PARAM_END +
         r.DS41_PARAM_START + ' name="b" string="true">y' + r.DS41_PARAM_END + '\n' + r.DS41_PARAM_END + '\n' +
         r.DS41_INVOKE_END + '\n' + r.DS41_TOOL_CALLS_END)


class ParserTests(unittest.TestCase):
    def test_reasoning_and_content(self):
        ok, content, reasoning, calls, raw = p.parse_deepseek_generated('think a bit</think>Hello  ', True)
        self.assertEqual((ok, content, reasoning, calls, raw), (True, 'Hello  ', 'think a bit', [], None))
        self.assertEqual(p.parse_deepseek_generated('plain answer', False)[1:3], ('plain answer', None))

    def test_unterminated_reasoning_is_not_an_answer(self):
        self.assertEqual(p.parse_deepseek_generated('<think>still thinking', True)[1:3], ('', 'still thinking'))

    def test_v41_tool_block(self):
        ok, content, reasoning, calls, raw = p.parse_deepseek_generated('R</think>Running it.  ' + CALLS, True)
        self.assertTrue(ok);self.assertEqual((content, reasoning), ('Running it.', 'R'))
        self.assertEqual(raw, CALLS)
        self.assertEqual(calls[0]['name'], 'run_shell')
        args = json.loads(calls[0]['arguments'])
        self.assertEqual(args, {'command': 'ls &lt;/｜DSML｜ parameter> </｜DSML｜ parameter> && echo "x"', 'timeout_s': 10.5})
        self.assertIn('"timeout_s": 10.50', calls[0]['arguments'])   # raw JSON spelling is kept
        self.assertEqual(calls[1]['name'], 'read&file')
        self.assertEqual(json.loads(calls[1]['arguments']), {'opts': {'a': [1, 2], 'b': 'y'}})

    def test_tool_inside_unclosed_thinking_is_recovered(self):
        ok, content, reasoning, calls, raw = p.parse_deepseek_generated('<think>plan' + CALLS, True)
        self.assertTrue(ok);self.assertEqual((content, reasoning, len(calls)), ('', 'plan', 2))

    def test_dsml_inside_reasoning_is_text(self):
        ok, content, reasoning, calls, _ = p.parse_deepseek_generated('quote ' + r.DS41_TOOL_CALLS_START + '</think>ok', True)
        self.assertEqual((ok, content, calls), (True, 'ok', []))

    def test_malformed_block_becomes_text(self):
        broken = 'x</think>' + CALLS.replace(r.DS41_INVOKE_END, 'oops', 1)
        result = p.parse_for_response(broken, has_tools=True, saw_tool_start=True, require_thinking_closed=True, finish='stop')
        self.assertEqual((result['ok'], result['content'], result['calls'], result['finish'], result['recovered']), (False, broken, [], 'stop', True))
        result = p.parse_for_response(broken, has_tools=True, saw_tool_start=True, require_thinking_closed=True, finish='length')
        self.assertEqual(result['finish'], 'length')

    def test_repair_only_adds_the_outer_wrapper(self):
        truncated = 'a</think>' + CALLS[:-len(r.DS41_TOOL_CALLS_END)]
        self.assertEqual(p.try_repair_dsml(truncated), truncated + r.DS41_TOOL_CALLS_END)
        unfinished = 'a</think>' + CALLS[:CALLS.index(r.DS41_INVOKE_END)]
        self.assertIsNone(p.try_repair_dsml(unfinished))
        self.assertIsNone(p.try_repair_dsml('no tools here'))
        result = p.finish_message(truncated, has_tools=True, think=True, finish='stop')
        self.assertEqual((result['finish'], len(result['calls'])), ('tool_calls', 2))
        self.assertTrue(all(c['id'].startswith('call_') and len(c['id']) == 37 for c in result['calls']))

    def test_render_parse_round_trip(self):
        calls = [dict(id='i', name='f', arguments='{"path": "/a</｜DSML｜ parameter>&amp;lt;b", "n": 3, "o": {"k": [true, null]}}')]
        block = r.dsml_tool_calls_text(calls)
        ok, content, _, parsed, raw = p.parse_deepseek_generated('Doing it.' + block, False)
        self.assertTrue(ok);self.assertEqual((content, raw), ('Doing it.', block))
        self.assertEqual(json.loads(parsed[0]['arguments']), json.loads(calls[0]['arguments']))

    def test_openai_deltas(self):
        rows = p.tool_call_deltas([dict(id='call_1', name='f', arguments='{"a": "x\\ny", "b": 2}')])
        self.assertEqual(rows, [dict(index=0, id='call_1', type='function', function=dict(name='f', arguments='{"a":"x\\ny","b":2}'))])
        self.assertEqual(p.json_object_or_empty('nope'), '{}')


# What the model actually emitted in pi's session of 2026-11-11 (runtime/single-engine/
# ultra-*/stderr.log turns, and the `Malformed tool call` activity the router reported).
# The tags the stack renders are `<｜DSML｜ calls>` / `</｜DSML｜ parameter>`; the model put
# whitespace inside the opening tag and closed the parameter with the bare marker.
DRIFT = ('<｜DSML｜\n calls>\n' + r.DS41_INVOKE_START + ' name="bash">\n' +
         r.DS41_PARAM_START + ' name="command" string="true">grep -n x a.py | head -30; echo\n---; sed -n 1,120p b.py' +
         '</｜DSML｜>\n' + r.DS41_INVOKE_END + '\n' + r.DS41_TOOL_CALLS_END)


class DriftedTagTests(unittest.TestCase):
    def test_the_session_block_parses(self):
        result = p.finish_message('Looking.  ' + DRIFT, has_tools=True, think=False, finish='stop')
        self.assertTrue(result['ok'])
        self.assertEqual(len(result['calls']), 1)
        self.assertEqual(result['calls'][0]['name'], 'bash')
        self.assertEqual(json.loads(result['calls'][0]['arguments']),
                         {'command': 'grep -n x a.py | head -30; echo\n---; sed -n 1,120p b.py'})
        self.assertEqual(result['content'], 'Looking.')
        self.assertEqual(result['finish'], 'tool_calls')
        self.assertNotIn('tool_error', result)

    def test_whitespace_inside_a_tag_is_accepted(self):
        for gap in ('\n', '\r\n', '\t', '  ', ' \n '):
            text = DRIFT.replace('<｜DSML｜\n calls>', '<｜DSML｜' + gap + ' calls>', 1)
            ok, content, _, calls, _ = p.parse_deepseek_generated(text, False)
            self.assertTrue(ok, gap)
            self.assertEqual([c['name'] for c in calls], ['bash'], gap)

    def test_a_bare_marker_closes_any_open_element(self):
        bare_calls = ('Thinking</think>x' + r.DS41_TOOL_CALLS_START + '\n' + r.DS41_INVOKE_START + ' name="f">\n' +
                      r.DS41_PARAM_START + ' name="a" string="true">v' + '</｜DSML｜>\n' + '</｜DSML｜>\n' + '</｜DSML｜>')
        ok, content, reasoning, calls, _ = p.parse_deepseek_generated(bare_calls, True)
        self.assertTrue(ok)
        self.assertEqual((content, reasoning, json.loads(calls[0]['arguments'])), ('x', 'Thinking', {'a': 'v'}))

    def test_bare_close_inside_a_nested_object(self):
        block = ('Thinking</think>x' + r.DS41_TOOL_CALLS_START + '\n' + r.DS41_INVOKE_START + ' name="f">\n' +
                 r.DS41_PARAM_START + ' name="opts">\n' +
                 r.DS41_PARAM_START + ' name="a" string="false">[1, 2]' + '</｜DSML｜>\n' +
                 r.DS41_PARAM_START + ' name="b" string="true">y' + r.DS41_PARAM_END + '\n' +
                 '</｜DSML｜>\n' + r.DS41_INVOKE_END + '\n' + r.DS41_TOOL_CALLS_END)
        ok, _, _, calls, _ = p.parse_deepseek_generated(block, True)
        self.assertTrue(ok)
        self.assertEqual(json.loads(calls[0]['arguments']), {'opts': {'a': [1, 2], 'b': 'y'}})

    def test_a_bare_close_written_as_a_value_is_not_a_close(self):
        calls = [dict(id='i', name='f', arguments='{"command": "echo </｜DSML｜> done"}')]
        block = r.dsml_tool_calls_text(calls)
        self.assertNotIn('echo </｜DSML｜>', block)   # the render escapes it
        ok, _, _, parsed, raw = p.parse_deepseek_generated('run' + block, False)
        self.assertTrue(ok);self.assertEqual(raw, block)
        self.assertEqual(json.loads(parsed[0]['arguments']), {'command': 'echo </｜DSML｜> done'})

    def test_markup_no_grammar_can_read_is_still_refused(self):
        for broken in (r.DS41_TOOL_CALLS_START + '\n' + r.DS41_INVOKE_START + '>\n' + r.DS41_TOOL_CALLS_END,
                       r.DS41_TOOL_CALLS_START + '\n' + r.DS41_PARAM_START + ' name="a">v' + r.DS41_PARAM_END,
                       '<｜DSML｜ nonsense>'):
            result = p.finish_message(broken, has_tools=True, think=False, finish='stop')
            self.assertEqual(result['calls'], [])
            self.assertTrue(result.get('tool_error'))

    def test_the_refusal_says_why(self):
        result = p.finish_message(r.DS41_TOOL_CALLS_START + '\n' + r.DS41_INVOKE_START + '>\n' +
                                  r.DS41_TOOL_CALLS_END, has_tools=True, think=False, finish='stop')
        self.assertIn('invoke name=', result['tool_error']['reason'])


if __name__ == '__main__':unittest.main()
