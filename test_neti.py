"""Tests for neti.py pure functions — no LLM calls, no file I/O."""

import math
import unittest

import neti


class TestTokenize(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(neti._tokenize("Hello World"), ["hello", "world"])

    def test_punctuation(self):
        self.assertEqual(neti._tokenize("it's a test!"), ["it", "s", "a", "test"])

    def test_empty(self):
        self.assertEqual(neti._tokenize(""), [])

    def test_numbers(self):
        self.assertEqual(neti._tokenize("test123 foo"), ["test123", "foo"])


class TestBowDistance(unittest.TestCase):
    def test_identical(self):
        self.assertAlmostEqual(neti._bow_distance("hello world", "hello world"), 0.0)

    def test_completely_different(self):
        self.assertAlmostEqual(neti._bow_distance("alpha beta", "gamma delta"), 1.0)

    def test_partial_overlap(self):
        d = neti._bow_distance("hello world foo", "hello world bar")
        self.assertGreater(d, 0.0)
        self.assertLess(d, 1.0)

    def test_empty_strings(self):
        self.assertAlmostEqual(neti._bow_distance("", ""), 1.0)

    def test_one_empty(self):
        self.assertAlmostEqual(neti._bow_distance("hello", ""), 1.0)

    def test_symmetric(self):
        a, b = "the quick brown fox", "the lazy brown dog"
        self.assertAlmostEqual(neti._bow_distance(a, b), neti._bow_distance(b, a))

    def test_range_zero_to_one(self):
        d = neti._bow_distance("some words here", "some other words there")
        self.assertGreaterEqual(d, 0.0)
        self.assertLessEqual(d, 1.0)


class TestCosineDistanceEmb(unittest.TestCase):
    def test_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        self.assertAlmostEqual(neti._cosine_distance_emb(v, v), 0.0)

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        self.assertAlmostEqual(neti._cosine_distance_emb(a, b), 1.0)

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        # cosine similarity = -1, distance = 1 - (-1) = 2, but clamped to max(0, ...)
        d = neti._cosine_distance_emb(a, b)
        self.assertGreaterEqual(d, 0.0)

    def test_similar_vectors(self):
        a = [1.0, 0.1, 0.0]
        b = [1.0, 0.2, 0.0]
        d = neti._cosine_distance_emb(a, b)
        self.assertGreaterEqual(d, 0.0)
        self.assertLess(d, 0.5)


class TestClassifySession(unittest.TestCase):
    def test_reframing(self):
        # after_text contains '?' and is far from question
        result = neti._classify_session(
            conf_before=8, conf_after=7, shift=0.4,
            direction='independent',
            after_text='What factors influence this outcome?',
            question='Does X cause Y?'
        )
        self.assertEqual(result, 'reframing')

    def test_destabilization(self):
        result = neti._classify_session(
            conf_before=8, conf_after=4, shift=0.05,
            direction='stable',
            after_text='Maybe not so sure anymore',
            question='Is X true?'
        )
        self.assertEqual(result, 'destabilization')

    def test_adoption(self):
        result = neti._classify_session(
            conf_before=5, conf_after=6, shift=0.2,
            direction='toward_pre_mortem',
            after_text='The pre-mortem view convinced me',
            question='Should we do X?'
        )
        self.assertEqual(result, 'adoption')

    def test_reconceptualization(self):
        result = neti._classify_session(
            conf_before=7, conf_after=6, shift=0.25,
            direction='independent',
            after_text='I think there is a third way entirely',
            question='Should we do X or Y?'
        )
        self.assertEqual(result, 'reconceptualization')

    def test_shift(self):
        result = neti._classify_session(
            conf_before=7, conf_after=6, shift=0.12,
            direction='stable',
            after_text='Slightly different view now',
            question='What about X?'
        )
        self.assertEqual(result, 'shift')

    def test_unshaken(self):
        result = neti._classify_session(
            conf_before=7, conf_after=7, shift=0.05,
            direction='stable',
            after_text='Same as before',
            question='What about X?'
        )
        self.assertEqual(result, 'unshaken')

    def test_destabilization_priority_over_adoption(self):
        # Confidence drop of 3+ should classify as destabilization even if direction is toward_X
        result = neti._classify_session(
            conf_before=9, conf_after=5, shift=0.2,
            direction='toward_blind_spot',
            after_text='Changed my mind completely',
            question='Is X true?'
        )
        self.assertEqual(result, 'destabilization')

    def test_none_confidence(self):
        result = neti._classify_session(
            conf_before=None, conf_after=None, shift=0.05,
            direction='stable',
            after_text='Same',
            question='What?'
        )
        self.assertEqual(result, 'unshaken')


class TestUpdateWeights(unittest.TestCase):
    def _make_state(self):
        return {'strategy_weights': {k: 1.0 for k in neti.STRATEGY_KEYS}}

    def test_shown_strategies_get_boost(self):
        state = self._make_state()
        neti._update_weights(state, ['pre_mortem', 'blind_spot'], None, 'shift', 'stable')
        self.assertGreater(state['strategy_weights']['pre_mortem'], 1.0)
        self.assertGreater(state['strategy_weights']['blind_spot'], 1.0)

    def test_rated_strategy_gets_larger_boost(self):
        state = self._make_state()
        neti._update_weights(state, ['pre_mortem', 'blind_spot'], 'pre_mortem', 'shift', 'stable')
        self.assertGreater(
            state['strategy_weights']['pre_mortem'],
            state['strategy_weights']['blind_spot']
        )

    def test_deep_shift_boosts_shown(self):
        state = self._make_state()
        neti._update_weights(state, ['pre_mortem'], None, 'reframing', 'stable')
        self.assertGreater(state['strategy_weights']['pre_mortem'], 1.05)

    def test_unshaken_reduces_shown(self):
        state = self._make_state()
        neti._update_weights(state, ['pre_mortem'], None, 'unshaken', 'stable')
        self.assertLess(state['strategy_weights']['pre_mortem'], 1.0)

    def test_weights_bounded_after_many_iterations(self):
        """Weights should not explode after many sessions of consistent rating."""
        state = self._make_state()
        for _ in range(100):
            neti._update_weights(
                state, ['pre_mortem', 'blind_spot'], 'pre_mortem',
                'reframing', 'toward_pre_mortem'
            )
        # After fix: weights should be reasonable (not thousands)
        for w in state['strategy_weights'].values():
            self.assertGreater(w, 0.01)
            # This will fail BEFORE the fix (weights explode to ~15000)
            # After the fix with normalization, all weights should be reasonable
            self.assertLess(w, 100.0)


class TestSelectStrategies(unittest.TestCase):
    def test_explicit_config(self):
        state = {'strategy_weights': {k: 1.0 for k in neti.STRATEGY_KEYS}}
        config = {'strategies': ['pre_mortem', 'falsification'], 'num_perspectives': 4}
        result = neti._select_strategies(state, config)
        self.assertEqual(result, ['pre_mortem', 'falsification'])

    def test_auto_returns_correct_count(self):
        state = {'strategy_weights': {k: 1.0 for k in neti.STRATEGY_KEYS}}
        config = {'strategies': 'auto', 'num_perspectives': 4}
        result = neti._select_strategies(state, config)
        self.assertEqual(len(result), 4)

    def test_auto_includes_top_weighted(self):
        weights = {k: 1.0 for k in neti.STRATEGY_KEYS}
        weights['pre_mortem'] = 5.0
        weights['blind_spot'] = 4.0
        state = {'strategy_weights': weights}
        config = {'strategies': 'auto', 'num_perspectives': 4}
        result = neti._select_strategies(state, config)
        self.assertIn('pre_mortem', result[:2])
        self.assertIn('blind_spot', result[:2])

    def test_invalid_strategy_filtered(self):
        state = {'strategy_weights': {k: 1.0 for k in neti.STRATEGY_KEYS}}
        config = {'strategies': ['nonexistent', 'pre_mortem'], 'num_perspectives': 4}
        result = neti._select_strategies(state, config)
        self.assertEqual(result, ['pre_mortem'])


class TestMeasureDirection(unittest.TestCase):
    def test_stable_when_no_change(self):
        result = neti._measure_direction(
            "I think X is true", "I think X is true",
            {'default': 'X is complicated', 'pre_mortem': 'X will fail'}
        )
        self.assertEqual(result, 'stable')

    def test_unknown_when_no_before(self):
        result = neti._measure_direction(
            None, "something", {'default': 'response'}
        )
        self.assertEqual(result, 'unknown')

    def test_unknown_when_no_after(self):
        result = neti._measure_direction(
            "something", None, {'default': 'response'}
        )
        self.assertEqual(result, 'unknown')


class TestMeasureIndependence(unittest.TestCase):
    def test_none_when_no_after(self):
        result = neti._measure_independence(None, {'default': 'response'})
        self.assertIsNone(result)

    def test_low_when_identical_to_response(self):
        text = "This is the exact response text"
        result = neti._measure_independence(text, {'default': text})
        self.assertIsNotNone(result)
        self.assertLess(result, 0.2)

    def test_range_zero_to_one(self):
        result = neti._measure_independence(
            "completely different text about bananas",
            {'default': 'analysis of quantum computing trends'}
        )
        if result is not None:
            self.assertGreaterEqual(result, 0.0)
            self.assertLessEqual(result, 1.0)


class TestPrintWrapped(unittest.TestCase):
    def test_short_line_no_wrap(self):
        """Should not raise."""
        import io, sys
        captured = io.StringIO()
        sys.stdout = captured
        try:
            neti._print_wrapped("short text", indent=4, width=76)
        finally:
            sys.stdout = sys.__stdout__
        self.assertIn("short text", captured.getvalue())

    def test_long_line_wraps(self):
        import io, sys
        captured = io.StringIO()
        sys.stdout = captured
        long = "word " * 30  # ~150 chars
        try:
            neti._print_wrapped(long, indent=4, width=40)
        finally:
            sys.stdout = sys.__stdout__
        lines = captured.getvalue().strip().split('\n')
        self.assertGreater(len(lines), 1)


if __name__ == '__main__':
    unittest.main()
