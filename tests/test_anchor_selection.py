"""Integration tests for divergence-ordered anchor selection.

These tests train RNN cloze models on synthetic corpora and verify
that select_anchors_divergence_ordered produces correct results on
grammars with known structure.

Tests are marked 'slow' and excluded from the default pytest run.
Run explicitly with:

    pytest tests/test_anchor_selection.py
    pytest -m slow

Each test samples 100k sentences, trains 3 RNN models (full + two
split-half), and runs the anchor selection algorithm (~30-60s per
grammar depending on hardware).
"""

import os
import sys
import tempfile

import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'locallearner'))

import wcfg

# All tests in this file are slow integration tests
pytestmark = pytest.mark.slow

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _sample_corpus(pcfg_path, n_sentences=100_000, seed=42):
    """Sample a corpus from a grammar and write to a temp file."""
    g = wcfg.load_wcfg_from_file(pcfg_path)
    rng = np.random.RandomState(seed)
    sampler = wcfg.Sampler(g, random=rng)
    sents = [' '.join(sampler.sample_string())
             for _ in range(n_sentences)]
    fd, corpus_path = tempfile.mkstemp(suffix='.txt')
    with os.fdopen(fd, 'w') as f:
        for s in sents:
            f.write(s + '\n')
    return corpus_path


def _train_and_select(corpus_path, alpha=2.0, epochs=20):
    """Train models and run divergence-ordered anchor selection."""
    from neural_learner import NeuralLearner
    nl = NeuralLearner(corpus_path)
    nl.alpha = alpha
    nl.model_type = 'rnn'
    nl.n_epochs = epochs
    nl.train_single_model(verbose=False)
    nl.train_split_models(verbose=False)
    anchors = nl.select_anchors_divergence_ordered(verbose=False)
    return anchors, nl


def _anchors_by_prefix(anchors, prefixes):
    """Group anchors by their first-character prefix."""
    result = {}
    for p in prefixes:
        result[p] = [w for w in anchors if w.startswith(p)]
    return result


# ------------------------------------------------------------------
# Tests: grammars where selection should succeed
# ------------------------------------------------------------------

class TestAnchorSelectionSuccess:
    """Grammars where the algorithm should find the correct anchors."""

    def test_clean_3nt(self):
        """3 NTs, well-separated, minimal sharing.

        Expected: exactly 3 anchors, one per NT, no shared terminals.
        """
        corpus = _sample_corpus(os.path.join(DATA_DIR, 'test_3nt_clean.pcfg'))
        try:
            anchors, nl = _train_and_select(corpus)
            by = _anchors_by_prefix(anchors, ['a', 'b', 'c', 'x'])

            assert len(anchors) == 3, (
                f"Expected 3 anchors, got {len(anchors)}: {anchors}")
            assert len(by['a']) == 1, f"Expected 1 A-anchor, got {by['a']}"
            assert len(by['b']) == 1, f"Expected 1 B-anchor, got {by['b']}"
            assert len(by['c']) == 1, f"Expected 1 C-anchor, got {by['c']}"
            assert len(by['x']) == 0, (
                f"Shared terminals selected: {by['x']}")
        finally:
            os.unlink(corpus)

    def test_skewed_3nt(self):
        """3 NTs with frequency imbalance (A dominates).

        Expected: exactly 3 anchors, one per NT.
        """
        corpus = _sample_corpus(os.path.join(DATA_DIR, 'test_3nt_skewed.pcfg'))
        try:
            anchors, nl = _train_and_select(corpus)
            by = _anchors_by_prefix(anchors, ['a', 'b', 'c', 'x'])

            assert len(anchors) == 3, (
                f"Expected 3 anchors, got {len(anchors)}: {anchors}")
            assert len(by['a']) == 1, f"Expected 1 A-anchor, got {by['a']}"
            assert len(by['b']) == 1, f"Expected 1 B-anchor, got {by['b']}"
            assert len(by['c']) == 1, f"Expected 1 C-anchor, got {by['c']}"
            assert len(by['x']) == 0, (
                f"Shared terminals selected: {by['x']}")
        finally:
            os.unlink(corpus)

    def test_overlap_4nt(self):
        """4 NTs with heavy terminal overlap (12 shared terminals).

        Expected: exactly 4 anchors, one per NT, no shared terminals.
        """
        corpus = _sample_corpus(
            os.path.join(DATA_DIR, 'test_4nt_overlap.pcfg'))
        try:
            anchors, nl = _train_and_select(corpus)
            by = _anchors_by_prefix(anchors, ['a', 'b', 'c', 'd', 'x'])

            assert len(anchors) == 4, (
                f"Expected 4 anchors, got {len(anchors)}: {anchors}")
            assert len(by['a']) == 1, f"Expected 1 A-anchor, got {by['a']}"
            assert len(by['b']) == 1, f"Expected 1 B-anchor, got {by['b']}"
            assert len(by['c']) == 1, f"Expected 1 C-anchor, got {by['c']}"
            assert len(by['d']) == 1, f"Expected 1 D-anchor, got {by['d']}"
            assert len(by['x']) == 0, (
                f"Shared terminals selected: {by['x']}")
        finally:
            os.unlink(corpus)


# ------------------------------------------------------------------
# Tests: grammars where selection should fail (known limitations)
# ------------------------------------------------------------------

class TestAnchorSelectionExpectedFailures:
    """Grammars that expose known theoretical limitations.

    These test that the algorithm fails gracefully (finds a correct
    subset, doesn't hallucinate structure) rather than checking for
    exact outputs.
    """

    def test_identity_merges_identical_nts(self):
        """A and B have identical context distributions.

        All NTs appear in comparable context sets (no containment),
        but A and B are interchangeable in the grammar.

        Expected: algorithm finds exactly 2 anchors — one for
        {A,B} merged and one for C.  It must NOT find 3.
        """
        corpus = _sample_corpus(
            os.path.join(DATA_DIR, 'test_3nt_identity.pcfg'))
        try:
            anchors, nl = _train_and_select(corpus)
            by = _anchors_by_prefix(anchors, ['a', 'b', 'c'])

            # Must not find 3 separate anchors
            assert len(anchors) < 3, (
                f"Should not separate identical NTs, "
                f"got {len(anchors)}: {anchors}")

            # Should find exactly 2: one A-or-B plus one C
            assert len(anchors) == 2, (
                f"Expected 2 anchors (A/B merged + C), "
                f"got {len(anchors)}: {anchors}")
            assert len(by['c']) == 1, (
                f"Expected 1 C-anchor, got {by['c']}")
            assert len(by['a']) + len(by['b']) == 1, (
                f"Expected 1 A-or-B anchor, got A:{by['a']} B:{by['b']}")
        finally:
            os.unlink(corpus)

    def test_subset_drops_superset_nt(self):
        """A's context distribution is a proper subset of C's.

        No NTs are identical, but the containment A ⊂ C creates
        a false mixture asymmetry that causes C to be rejected.

        Expected: algorithm finds only 2 anchors (A and B),
        incorrectly dropping C.  It must NOT find 3.
        This documents the known limitation.
        """
        corpus = _sample_corpus(
            os.path.join(DATA_DIR, 'test_3nt_subset.pcfg'))
        try:
            anchors, nl = _train_and_select(corpus)
            by = _anchors_by_prefix(anchors, ['a', 'b', 'c'])

            # The subset containment prevents finding all 3
            assert len(anchors) < 3, (
                f"Should not find 3 anchors with subset containment, "
                f"got {len(anchors)}: {anchors}")

            # Should find A and B but not C
            assert len(anchors) == 2, (
                f"Expected 2 anchors (A + B, C dropped), "
                f"got {len(anchors)}: {anchors}")
            assert len(by['a']) == 1, (
                f"Expected 1 A-anchor, got {by['a']}")
            assert len(by['b']) == 1, (
                f"Expected 1 B-anchor, got {by['b']}")
            assert len(by['c']) == 0, (
                f"C should be dropped due to subset containment, "
                f"got {by['c']}")
        finally:
            os.unlink(corpus)


# ------------------------------------------------------------------
# Tests: divergence structure validation
# ------------------------------------------------------------------

class TestDivergenceStructure:
    """Verify that pairwise divergences have the expected structure."""

    def _compute_divergences(self, pcfg_path):
        """Train model and compute pairwise divergences."""
        from neural_learner import NeuralLearner
        corpus = _sample_corpus(pcfg_path)
        try:
            nl = NeuralLearner(corpus)
            nl.alpha = 2.0
            nl.model_type = 'rnn'
            nl.n_epochs = 20
            nl.train_single_model(verbose=False)

            terminals = sorted(nl.vocab,
                               key=lambda w: -nl.word_counts.get(w, 0))
            terminals = [w for w in terminals if w in nl.w2i]
            log_p = nl._compute_terminal_log_probs(terminals)
            div = nl._compute_pairwise_renyi(log_p, terminals)
            return div, nl
        finally:
            os.unlink(corpus)

    def test_identity_divergence_near_zero(self):
        """A-B divergence should be near zero when NTs are identical."""
        div, nl = self._compute_divergences(
            os.path.join(DATA_DIR, 'test_3nt_identity.pcfg'))

        # D(a1||b1) and D(b1||a1) should both be near zero
        d_ab = div.get(('a1', 'b1'), float('nan'))
        d_ba = div.get(('b1', 'a1'), float('nan'))
        assert abs(d_ab) < 0.5, (
            f"D(a1||b1)={d_ab:.3f}, expected near 0")
        assert abs(d_ba) < 0.5, (
            f"D(b1||a1)={d_ba:.3f}, expected near 0")

        # Should be roughly symmetric
        assert abs(d_ab - d_ba) < 0.5, (
            f"Asymmetry {abs(d_ab-d_ba):.3f} too large for same-NT")

    def test_identity_c_symmetric(self):
        """A-C and B-C divergences should be symmetric (no containment)."""
        div, nl = self._compute_divergences(
            os.path.join(DATA_DIR, 'test_3nt_identity.pcfg'))

        d_ac = div.get(('a1', 'c1'), 0)
        d_ca = div.get(('c1', 'a1'), 0)
        ratio = max(d_ac, d_ca) / max(min(d_ac, d_ca), 1e-6)

        assert ratio < 3.0, (
            f"A-C should be symmetric, "
            f"D(a1||c1)={d_ac:.3f}, D(c1||a1)={d_ca:.3f}, "
            f"ratio={ratio:.1f}")

    def test_subset_asymmetry(self):
        """A-C divergence should show strong asymmetry when A ⊂ C."""
        div, nl = self._compute_divergences(
            os.path.join(DATA_DIR, 'test_3nt_subset.pcfg'))

        d_ac = div.get(('a1', 'c1'), 0)
        d_ca = div.get(('c1', 'a1'), 0)

        # D(a1||c1) should be moderate (A's contexts ⊂ C's)
        # D(c1||a1) should be large (C has extra contexts)
        assert d_ac < d_ca, (
            f"Expected D(a1||c1) < D(c1||a1), "
            f"got {d_ac:.3f} vs {d_ca:.3f}")

        ratio = d_ca / max(d_ac, 1e-6)
        assert ratio > 3.0, (
            f"Expected strong asymmetry ratio > 3, "
            f"got {ratio:.1f}")

    def test_subset_bc_symmetric(self):
        """B-C divergence should be symmetric (no subset relation)."""
        div, nl = self._compute_divergences(
            os.path.join(DATA_DIR, 'test_3nt_subset.pcfg'))

        d_bc = div.get(('b1', 'c1'), 0)
        d_cb = div.get(('c1', 'b1'), 0)
        ratio = max(d_bc, d_cb) / max(min(d_bc, d_cb), 1e-6)

        assert ratio < 3.0, (
            f"B-C should be symmetric, "
            f"D(b1||c1)={d_bc:.3f}, D(c1||b1)={d_cb:.3f}, "
            f"ratio={ratio:.1f}")
