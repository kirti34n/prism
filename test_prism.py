"""Tests for prism.py - pure functions plus mocked-LLM flow tests. No real network,
no real file I/O (state functions are patched or pure helpers are called directly)."""

import io
import json
import contextlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import prism


def _tty(value):
    fake = mock.MagicMock()
    fake.isatty.return_value = value
    return fake


class TestTokenize(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(prism._tokenize("Hello World"), ["hello", "world"])

    def test_punctuation(self):
        self.assertEqual(prism._tokenize("it's a test!"), ["it", "s", "a", "test"])

    def test_empty(self):
        self.assertEqual(prism._tokenize(""), [])

    def test_numbers(self):
        self.assertEqual(prism._tokenize("test123 foo"), ["test123", "foo"])


class TestBowDistance(unittest.TestCase):
    def test_identical(self):
        self.assertAlmostEqual(prism._bow_distance("hello world", "hello world"), 0.0)

    def test_completely_different(self):
        self.assertAlmostEqual(prism._bow_distance("alpha beta", "gamma delta"), 1.0)

    def test_partial_overlap(self):
        d = prism._bow_distance("hello world foo", "hello world bar")
        self.assertGreater(d, 0.0)
        self.assertLess(d, 1.0)

    def test_empty_strings(self):
        self.assertAlmostEqual(prism._bow_distance("", ""), 1.0)

    def test_one_empty(self):
        self.assertAlmostEqual(prism._bow_distance("hello", ""), 1.0)

    def test_symmetric(self):
        a, b = "the quick brown fox", "the lazy brown dog"
        self.assertAlmostEqual(prism._bow_distance(a, b), prism._bow_distance(b, a))

    def test_range_zero_to_one(self):
        d = prism._bow_distance("some words here", "some other words there")
        self.assertGreaterEqual(d, 0.0)
        self.assertLessEqual(d, 1.0)


class TestClassifySessionV3(unittest.TestCase):
    def c(self, cat, cb, ca, moved):
        return prism._classify_session(cat, cb, ca, moved)

    def test_reframing_trumps_everything(self):
        # different_question wins even over a sharp conviction drop
        self.assertEqual(self.c('different_question', 90, 50, 'pre_mortem'), 'reframing')

    def test_destabilization_at_threshold(self):
        self.assertEqual(self.c('same', 80, 60, None), 'destabilization')  # -20 fires

    def test_no_destabilization_below_threshold(self):
        self.assertEqual(self.c('same', 80, 61, None), 'unshaken')  # -19 does not

    def test_destabilization_without_category(self):
        self.assertEqual(self.c(None, 90, 65, None), 'destabilization')

    def test_destabilization_needs_both_convictions(self):
        self.assertEqual(self.c('shifted', None, 40, None), 'shift')

    def test_adoption_needs_moved_and_movement(self):
        self.assertEqual(self.c('shifted', 60, 55, 'blind_spot'), 'adoption')

    def test_moved_by_with_same_is_not_adoption(self):
        self.assertEqual(self.c('same', 60, 58, 'blind_spot'), 'unshaken')

    def test_switch(self):
        self.assertEqual(self.c('switched', 70, 65, None), 'switch')

    def test_shift(self):
        self.assertEqual(self.c('shifted', 70, 68, None), 'shift')

    def test_unshaken(self):
        self.assertEqual(self.c('same', 70, 70, None), 'unshaken')

    def test_unmeasured_when_nothing(self):
        self.assertEqual(self.c(None, None, None, None), 'unmeasured')

    def test_destabilization_beats_adoption(self):
        # both conditions hold: -25 drop AND moved_by+shifted. Precedence must be destabilization.
        self.assertEqual(self.c('shifted', 80, 55, 'blind_spot'), 'destabilization')


class TestMigration(unittest.TestCase):
    def test_migrate_preserves_and_tags(self):
        data = {'version': 2,
                'sessions': [{'id': 'a', 'question': 'q'}, {'id': 'b', 'schema': 'x'}],
                'strategy_weights': {'pre_mortem': 2.0}}
        out = prism._migrate_v2(data)
        self.assertEqual(out['version'], 3)
        self.assertNotIn('strategy_weights', out)
        self.assertEqual(len(out['sessions']), 2)
        self.assertEqual(out['sessions'][0]['schema'], 'v2-legacy')
        self.assertEqual(out['sessions'][1]['schema'], 'x')  # existing schema untouched

    def test_idempotent(self):
        data = {'version': 2, 'sessions': [{'id': 'a'}]}
        prism._migrate_v2(data)
        prism._migrate_v2(data)
        self.assertEqual(data['sessions'][0]['schema'], 'v2-legacy')
        self.assertEqual(data['version'], 3)

    def test_empty_sessions(self):
        out = prism._migrate_v2({'version': 2})
        self.assertEqual(out['version'], 3)


class TestSelectStrategies(unittest.TestCase):
    def test_explicit_list_honored(self):
        r = prism._select_strategies({'strategies': ['pre_mortem', 'falsification'],
                                      'num_perspectives': 4})
        self.assertEqual(r, ['pre_mortem', 'falsification'])

    def test_invalid_filtered(self):
        r = prism._select_strategies({'strategies': ['nonexistent', 'pre_mortem'],
                                      'num_perspectives': 4})
        self.assertEqual(r, ['pre_mortem'])

    def test_invalid_only_falls_to_random(self):
        r = prism._select_strategies({'strategies': ['nonexistent'], 'num_perspectives': 4})
        self.assertEqual(len(r), 4)
        self.assertTrue(all(k in prism.STRATEGY_KEYS for k in r))

    def test_auto_count_and_membership(self):
        r = prism._select_strategies({'strategies': 'auto', 'num_perspectives': 4})
        self.assertEqual(len(r), 4)
        self.assertTrue(all(k in prism.STRATEGY_KEYS for k in r))

    def test_default_never_selected(self):
        for _ in range(20):
            self.assertNotIn('default', prism._select_strategies({'num_perspectives': 5}))


class TestRevisitCandidate(unittest.TestCase):
    def test_skips_check_and_revisited(self):
        sessions = [
            {'session_type': 'check', 'question': 'c'},
            {'session_type': 'shift', 'position_after': 'a', 'revisit': {}},
            {'session_type': 'shift', 'position_after': 'later'},
        ]
        self.assertIs(prism._revisit_candidate(sessions), sessions[2])

    def test_skips_no_after(self):
        sessions = [{'session_type': 'unmeasured', 'position_after': None},
                    {'session_type': 'shift', 'position_after': 'x'}]
        self.assertIs(prism._revisit_candidate(sessions), sessions[1])

    def test_returns_oldest(self):
        sessions = [{'position_after': 'first'}, {'position_after': 'second'}]
        self.assertIs(prism._revisit_candidate(sessions), sessions[0])

    def test_legacy_eligible(self):
        sessions = [{'human_after': 'legacy answer'}]
        self.assertIs(prism._revisit_candidate(sessions), sessions[0])

    def test_none_when_empty(self):
        self.assertIsNone(prism._revisit_candidate([]))


class TestReadConviction(unittest.TestCase):
    def _conv(self, val):
        with mock.patch('sys.stdin', _tty(True)), mock.patch('builtins.input', return_value=val):
            return prism._read_conviction()

    def test_valid(self):
        self.assertEqual(self._conv('85'), 85)
        self.assertEqual(self._conv('0'), 0)
        self.assertEqual(self._conv('100'), 100)

    def test_percent_stripped(self):
        self.assertEqual(self._conv('85%'), 85)

    def test_out_of_range(self):
        self.assertIsNone(self._conv('101'))
        self.assertIsNone(self._conv('-3'))

    def test_non_numeric(self):
        self.assertIsNone(self._conv('abc'))
        self.assertIsNone(self._conv(''))

    def test_unicode_digit_like(self):
        # str.isdigit() is True for these but int() would raise - must not crash
        self.assertIsNone(self._conv('²'))
        self.assertIsNone(self._conv('¹²'))

    def test_non_tty_returns_none(self):
        with mock.patch('sys.stdin', _tty(False)):
            self.assertIsNone(prism._read_conviction())


class TestPromptContent(unittest.TestCase):
    def test_all_strategies_have_accuracy_guard(self):
        for key, s in prism.STRATEGIES.items():
            self.assertIn('verifiable', s['system'], f"{key} missing accuracy guard")

    def test_no_absolutist_tone(self):
        for key, s in prism.STRATEGIES.items():
            self.assertNotIn('commit fully', s['system'], key)
            self.assertNotIn('No hedging', s['system'], key)


class TestContrastiveScaffolding(unittest.TestCase):
    def test_default_passed_and_refute_instruction(self):
        calls = []
        def mock_llm(system, user, config):
            calls.append({'system': system, 'user': user})
            return 'mock response'
        original = prism._llm_call
        prism._llm_call = mock_llm
        try:
            prism._generate_perspectives('test question', ['devils_advocate'],
                                         {'provider': 'mock', 'max_tokens': 100}, quiet=True)
        finally:
            prism._llm_call = original
        self.assertGreaterEqual(len(calls), 2)
        strategy_call = calls[-1]
        self.assertIn('conventional AI answer', strategy_call['user'])
        self.assertIn('mock response', strategy_call['user'])
        self.assertIn('refute', strategy_call['user'])


class TestRebuttal(unittest.TestCase):
    def test_none_when_non_tty(self):
        with mock.patch('sys.stdin', _tty(False)):
            self.assertIsNone(prism._rebuttal_round('q', ['pre_mortem'], {'pre_mortem': 't'}, {}))

    def test_reply_uses_strategy_system_and_pushback(self):
        calls = []
        original = prism._llm_call
        prism._llm_call = lambda s, u, c: (calls.append((s, u)) or 'reply')
        try:
            with mock.patch('sys.stdin', _tty(True)), \
                 mock.patch('builtins.input', side_effect=['1', 'I disagree because X']), \
                 contextlib.redirect_stdout(io.StringIO()):
                r = prism._rebuttal_round('Q', ['pre_mortem'], {'pre_mortem': 'persp'}, {})
        finally:
            prism._llm_call = original
        self.assertEqual(r, {'strategy': 'pre_mortem', 'text': 'I disagree because X'})
        self.assertEqual(calls[0][0], prism.STRATEGIES['pre_mortem']['system'])
        self.assertIn('I disagree because X', calls[0][1])

    def test_skip_on_empty_choice(self):
        with mock.patch('sys.stdin', _tty(True)), \
             mock.patch('builtins.input', return_value=''), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertIsNone(prism._rebuttal_round('Q', ['pre_mortem'], {'pre_mortem': 'p'}, {}))


class TestNonTTYExplore(unittest.TestCase):
    def test_logs_unmeasured_session_no_prompts(self):
        saved = []
        tmp = Path(tempfile.mkdtemp())
        # Point the real _log at a temp dir instead of stubbing it - explore's final
        # log line contains a literal '→', which must not crash the run (cp1252 regression).
        patches = {
            '_llm_call': lambda s, u, c: 'response ' + s[:8],
            '_load_config': lambda: {'provider': 'mock', 'model': 'm',
                                     'num_perspectives': 3, 'num_shown': 3, 'max_tokens': 300},
            '_load_state': lambda: prism._new_state(),
            '_save_state': lambda st: saved.append(st),
            'CONFIG_DIR': tmp,
            'LOG_FILE': tmp / 'prism.log',
        }
        originals = {k: getattr(prism, k) for k in patches}
        for k, v in patches.items():
            setattr(prism, k, v)
        out = io.StringIO()
        try:
            with mock.patch('sys.stdin', _tty(False)), contextlib.redirect_stdout(out):
                prism.explore('should we ship on friday')
        finally:
            for k, v in originals.items():
                setattr(prism, k, v)
        self.assertTrue(saved, "no session saved")
        sess = saved[0]['sessions'][-1]
        self.assertEqual(sess['schema'], 'v3')
        self.assertEqual(sess['session_type'], 'unmeasured')
        self.assertIsNone(sess['position_before'])
        self.assertIsNone(sess['conviction_before'])
        self.assertNotIn('  > ', out.getvalue())  # no interactive prompt lines


class TestBuildOllama(unittest.TestCase):
    def _body(self, model):
        _, body, _, _ = prism._build_ollama('sys', 'usr', model, 0.7, 200, {})
        return json.loads(body)

    def test_thinking_model_disables_think(self):
        # qwen3+ would otherwise burn the budget inside <think> and truncate the answer
        b = self._body('qwen3.5:4b')
        self.assertIs(b['think'], False)
        self.assertEqual(b['options']['num_predict'], 200)

    def test_non_thinking_model_omits_think(self):
        b = self._body('llama3.1:8b')
        self.assertNotIn('think', b)

    def test_deepseek_r1_disables_think(self):
        self.assertIs(self._body('deepseek-r1:7b')['think'], False)


class TestBuildAnthropic(unittest.TestCase):
    def _body(self, model):
        _, body, _, _ = prism._build_anthropic('sys', 'usr', model, 0.9, 300, {})
        return json.loads(body)

    def test_omits_temperature_for_no_sampling_models(self):
        # Sonnet 5 / Opus 4.7+/ Fable 5 reject temperature with a 400
        for m in ('claude-sonnet-5', 'claude-opus-4-8', 'claude-fable-5'):
            self.assertNotIn('temperature', self._body(m), m)

    def test_keeps_temperature_for_older_models(self):
        for m in ('claude-haiku-4-5', 'claude-sonnet-4-6'):
            self.assertEqual(self._body(m)['temperature'], 0.9, m)


class TestForceUtf8(unittest.TestCase):
    def test_noop_when_stream_has_no_reconfigure(self):
        # StringIO has no .reconfigure, so the call must be swallowed, not raised
        with mock.patch('sys.stdout', io.StringIO()), mock.patch('sys.stderr', io.StringIO()):
            prism._force_utf8_output()  # must not raise


class TestLog(unittest.TestCase):
    def test_handles_non_ascii(self):
        tmp = Path(tempfile.mkdtemp())
        with mock.patch('prism.CONFIG_DIR', tmp), mock.patch('prism.LOG_FILE', tmp / 'l.log'):
            prism._log('type=shift conv=40→70 café \U0001f600')  # must not raise
        self.assertIn('40', (tmp / 'l.log').read_text(encoding='utf-8'))


class TestPrintWrapped(unittest.TestCase):
    def test_short_line_no_wrap(self):
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            prism._print_wrapped("short text", indent=4, width=76)
        self.assertIn("short text", captured.getvalue())

    def test_long_line_wraps(self):
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            prism._print_wrapped("word " * 30, indent=4, width=40)
        self.assertGreater(len(captured.getvalue().strip().split('\n')), 1)


if __name__ == '__main__':
    unittest.main()
