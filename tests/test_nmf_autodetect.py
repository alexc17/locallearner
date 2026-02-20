"""Tests for NMF auto-detection of the number of nonterminals.

Uses realistic grammars from the data directory (copied to fixtures) with
sampled corpora to verify that the auto-detection algorithm finds
the correct number of nonterminals.

The stopping criterion uses the ratio of a candidate's scaled chi-squared
statistic to the expected null maximum.  Real kernels have ratio >> 10;
spurious ones have ratio ~ 1.
"""

import pytest
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'locallearner'))

import locallearner
from locallearner import LocalLearner
import wcfg
import evaluation

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), 'fixtures')


@pytest.fixture
def grammar4_corpus_path():
    """Path to the grammar4 corpus (4 NTs, 1000 terminals, 100K sentences)."""
    return os.path.join(FIXTURES_DIR, 'grammar4_corpus.txt')


@pytest.fixture
def grammar4_pcfg_path():
    """Path to the grammar4 PCFG."""
    return os.path.join(FIXTURES_DIR, 'grammar4.pcfg')


@pytest.fixture
def grammar5_corpus_path():
    """Path to the grammar5 corpus (5 NTs, 172 terminals, 100K sentences)."""
    return os.path.join(FIXTURES_DIR, 'grammar5_corpus.txt')


@pytest.fixture
def grammar5_pcfg_path():
    """Path to the grammar5 PCFG."""
    return os.path.join(FIXTURES_DIR, 'grammar5.pcfg')


def _run_autodetect(corpus_path, min_count=10, seed=42, ratio_threshold=10.0,
                    max_nt=20, verbose=False):
    """Helper to run auto-detection with standard settings."""
    ll = LocalLearner(corpus_path)
    ll.nonterminals = 0  # auto-detect
    ll.number_clusters = 10
    ll.min_count_nmf = min_count
    ll.seed = seed
    ll.nmf_ratio_threshold = ratio_threshold
    ll.max_nonterminals = max_nt
    kernels = ll.find_kernels(verbose=verbose)
    return kernels, ll


class TestNMFAutoDetectGrammar5:
    """Auto-detection tests using grammar5 (5 NTs, 172 terminals)."""

    def test_auto_detects_correct_count(self, grammar5_corpus_path):
        """Auto-detection finds exactly 5 nonterminals for grammar5."""
        kernels, _ = _run_autodetect(grammar5_corpus_path)
        assert len(kernels) == 5, (
            f"Expected 5 NTs for grammar5, got {len(kernels)}: {kernels}"
        )

    def test_auto_detected_kernels_include_start(self, grammar5_corpus_path):
        """Auto-detected kernels always start with S."""
        kernels, _ = _run_autodetect(grammar5_corpus_path)
        assert kernels[0] == 'S'

    def test_auto_detected_kernels_are_good(self, grammar5_corpus_path, grammar5_pcfg_path):
        """Auto-detected kernels have high posteriors for grammar5."""
        kernels, _ = _run_autodetect(grammar5_corpus_path)

        target = wcfg.load_wcfg_from_file(grammar5_pcfg_path)
        result = evaluation.evaluate_kernels_hungarian(target, kernels)

        assert result['mean_posterior'] > 0.5, (
            f"Mean posterior {result['mean_posterior']:.3f} too low"
        )

    def test_fixed_count_works(self, grammar5_corpus_path, grammar5_pcfg_path):
        """With nonterminals=5 explicitly, find_kernels finds good kernels."""
        ll = LocalLearner(grammar5_corpus_path)
        ll.nonterminals = 5
        ll.number_clusters = 10
        ll.min_count_nmf = 10
        ll.seed = 42

        kernels = ll.find_kernels(verbose=False)
        assert len(kernels) == 5

        target = wcfg.load_wcfg_from_file(grammar5_pcfg_path)
        result = evaluation.evaluate_kernels_hungarian(target, kernels)
        assert result['mean_posterior'] > 0.5

    def test_auto_respects_min_nonterminals(self, grammar5_corpus_path):
        """Auto-detection never returns fewer than min_nonterminals."""
        ll = LocalLearner(grammar5_corpus_path)
        ll.nonterminals = 0
        ll.min_nonterminals = 3
        ll.number_clusters = 10
        ll.min_count_nmf = 10
        ll.seed = 42
        # Very strict threshold to encourage early stopping
        ll.nmf_ratio_threshold = 1e6

        kernels = ll.find_kernels(verbose=False)
        assert len(kernels) >= 3

    def test_auto_respects_max_nonterminals(self, grammar5_corpus_path):
        """Auto-detection never returns more than max_nonterminals."""
        ll = LocalLearner(grammar5_corpus_path)
        ll.nonterminals = 0
        ll.max_nonterminals = 3
        ll.number_clusters = 10
        ll.min_count_nmf = 10
        ll.seed = 42
        # Very lenient threshold
        ll.nmf_ratio_threshold = 0.01

        kernels = ll.find_kernels(verbose=False)
        assert len(kernels) <= 3


class TestNMFAutoDetectGrammar4:
    """Auto-detection tests using grammar4 (4 NTs, 1000 terminals)."""

    def test_auto_detects_correct_count(self, grammar4_corpus_path):
        """Auto-detection finds exactly 4 nonterminals for grammar4."""
        kernels, _ = _run_autodetect(grammar4_corpus_path, min_count=10)
        assert len(kernels) == 4, (
            f"Expected 4 NTs for grammar4, got {len(kernels)}: {kernels}"
        )

    def test_auto_detected_kernels_are_good(self, grammar4_corpus_path, grammar4_pcfg_path):
        """Auto-detected kernels have high posteriors for grammar4."""
        kernels, _ = _run_autodetect(grammar4_corpus_path, min_count=10)

        target = wcfg.load_wcfg_from_file(grammar4_pcfg_path)
        result = evaluation.evaluate_kernels_hungarian(target, kernels)

        assert result['mean_posterior'] > 0.5, (
            f"Mean posterior {result['mean_posterior']:.3f} too low"
        )

    def test_fixed_count_works(self, grammar4_corpus_path, grammar4_pcfg_path):
        """With nonterminals=4 explicitly, find_kernels finds good kernels."""
        ll = LocalLearner(grammar4_corpus_path)
        ll.nonterminals = 4
        ll.number_clusters = 10
        ll.min_count_nmf = 10
        ll.seed = 42

        kernels = ll.find_kernels(verbose=False)
        assert len(kernels) == 4

        target = wcfg.load_wcfg_from_file(grammar4_pcfg_path)
        result = evaluation.evaluate_kernels_hungarian(target, kernels)
        assert result['mean_posterior'] > 0.5


class TestNMFAutoDetectConsistency:
    """Cross-grammar consistency and structural invariant tests."""

    def test_different_grammars_different_counts(
        self, grammar4_corpus_path, grammar5_corpus_path
    ):
        """Auto-detection finds different counts for different grammars."""
        kernels4, _ = _run_autodetect(grammar4_corpus_path, min_count=10)
        kernels5, _ = _run_autodetect(grammar5_corpus_path, min_count=10)

        assert len(kernels4) == 4
        assert len(kernels5) == 5

    def test_ratio_threshold_monotonic(self, grammar5_corpus_path):
        """Higher ratio threshold gives fewer or equal nonterminals."""
        results = []
        for rt in [1.0, 5.0, 10.0, 50.0]:
            kernels, _ = _run_autodetect(
                grammar5_corpus_path, ratio_threshold=rt, max_nt=30
            )
            results.append(len(kernels))

        # Stricter threshold should find <= as many NTs
        for i in range(len(results) - 1):
            assert results[i] >= results[i + 1], (
                f"Ratio threshold monotonicity violated: "
                f"thresholds [1, 5, 10, 50] gave {results}"
            )

    def test_max_nonterminals_is_hard_limit(self, grammar5_corpus_path):
        """max_nonterminals is always respected regardless of threshold."""
        for max_nt in [2, 4, 6]:
            kernels, _ = _run_autodetect(
                grammar5_corpus_path, ratio_threshold=0.01, max_nt=max_nt
            )
            assert len(kernels) <= max_nt, (
                f"max_nonterminals={max_nt} violated: got {len(kernels)}"
            )

    def test_kernels_are_unique(self, grammar5_corpus_path):
        """All auto-detected kernels are distinct words."""
        kernels, _ = _run_autodetect(grammar5_corpus_path)
        assert len(kernels) == len(set(kernels))

    def test_first_kernel_is_start(self, grammar4_corpus_path, grammar5_corpus_path):
        """First kernel is always the start symbol for all grammars."""
        k4, _ = _run_autodetect(grammar4_corpus_path, min_count=10)
        k5, _ = _run_autodetect(grammar5_corpus_path)

        assert k4[0] == 'S'
        assert k5[0] == 'S'

    def test_auto_sets_nonterminals_attribute(self, grammar5_corpus_path):
        """After auto-detection, ll.nonterminals equals len(kernels)."""
        kernels, ll = _run_autodetect(grammar5_corpus_path)
        assert ll.nonterminals == len(kernels)

    def test_seed_reproducibility(self, grammar5_corpus_path):
        """Same seed produces same kernels."""
        k1, _ = _run_autodetect(grammar5_corpus_path, seed=123)
        k2, _ = _run_autodetect(grammar5_corpus_path, seed=123)
        assert k1 == k2

    def test_different_seeds_may_differ(self, grammar5_corpus_path):
        """Different seeds may produce different kernels (or same)."""
        # Just verify that different seeds don't crash
        k1, _ = _run_autodetect(grammar5_corpus_path, seed=1)
        k2, _ = _run_autodetect(grammar5_corpus_path, seed=99)
        assert len(k1) >= 2
        assert len(k2) >= 2
