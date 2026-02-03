"""Tests for Neyessen clustering algorithm."""

import pytest
import os
import sys
import math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'locallearner'))

import neyessen
from neyessen import Clustering


class TestClusteringInit:
    """Tests for Clustering initialization."""

    def test_default_init(self):
        """Clustering initializes with default values."""
        c = Clustering()

        assert c.clusters == 32
        assert c.min_count == 10
        assert c.boundary == '<eos>'


class TestClusteringLoad:
    """Tests for loading corpus data."""

    def test_load_from_data(self):
        """Load statistics from list of sentences."""
        c = Clustering()
        c.min_count = 1  # Accept all words

        data = [
            ['a', 'b', 'c'],
            ['b', 'c', 'a'],
            ['a', 'a', 'b'],
        ]
        c.load(data=data)

        assert 'a' in c.word2idx
        assert 'b' in c.word2idx
        assert 'c' in c.word2idx

    def test_load_counts_unigrams(self):
        """Load correctly counts unigrams."""
        c = Clustering()
        c.min_count = 1

        data = [
            ['a', 'b'],
            ['a', 'a'],
        ]
        c.load(data=data)

        # Should count occurrences including boundary context
        assert c.unigrams is not None
        assert len(c.unigrams) == c.alphabetsize

    def test_load_counts_bigrams(self):
        """Load correctly counts bigrams."""
        c = Clustering()
        c.min_count = 1

        data = [
            ['a', 'b'],
            ['a', 'b'],
        ]
        c.load(data=data)

        # Bigram (a, b) should appear twice
        a_idx = c.lookup('a')
        b_idx = c.lookup('b')
        assert c.bigrams[(a_idx, b_idx)] == 2

    def test_load_handles_rare_words(self):
        """Load groups rare words as UNK."""
        c = Clustering()
        c.min_count = 3

        data = [
            ['common', 'common', 'common'],
            ['rare1', 'common', 'rare2'],
        ]
        c.load(data=data)

        # 'common' should be in vocabulary
        assert 'common' in c.word2idx
        # Rare words should be unknown
        assert 'rare1' in c.unknowns
        assert 'rare2' in c.unknowns


class TestRandomInitialization:
    """Tests for random cluster initialization."""

    def test_random_initialization_assigns_clusters(self):
        """random_initialization assigns all words to clusters."""
        c = Clustering()
        c.min_count = 1
        c.clusters = 5

        data = [['a', 'b', 'c', 'd', 'e']]
        c.load(data=data)
        c.random_initialization(seed=42)

        # All words should have cluster assignments
        assert len(c.cluster_indices) == c.alphabetsize
        # Boundary should be in cluster 0
        assert c.cluster_indices[0] == 0

    def test_random_initialization_reproducible(self):
        """Same seed produces same initialization."""
        c1 = Clustering()
        c1.min_count = 1
        c1.clusters = 5

        c2 = Clustering()
        c2.min_count = 1
        c2.clusters = 5

        data = [['a', 'b', 'c', 'd', 'e']]

        c1.load(data=data)
        c1.random_initialization(seed=42)

        c2.load(data=data)
        c2.random_initialization(seed=42)

        assert list(c1.cluster_indices) == list(c2.cluster_indices)


class TestObjectiveFunction:
    """Tests for mutual information objective."""

    def test_objective_function_computes(self):
        """objective_function returns a value."""
        c = Clustering()
        c.min_count = 1
        c.clusters = 3

        data = [
            ['a', 'b', 'a', 'b'],
            ['b', 'a', 'b', 'a'],
        ]
        c.load(data=data)
        c.random_initialization(seed=42)

        of = c.objective_function()
        assert isinstance(of, float)

    def test_objective_function_finite(self):
        """Objective function returns a finite value."""
        c = Clustering()
        c.min_count = 1
        c.clusters = 3

        data = [
            ['a', 'b', 'c'],
            ['b', 'c', 'a'],
        ]
        c.load(data=data)
        c.random_initialization(seed=42)

        of = c.objective_function()
        assert math.isfinite(of)


class TestRecluster:
    """Tests for reclustering (optimization step)."""

    def test_recluster_returns_change_count(self):
        """recluster returns number of changed assignments."""
        c = Clustering()
        c.min_count = 1
        c.clusters = 5

        data = [
            ['a', 'b', 'c', 'd'],
            ['b', 'c', 'd', 'a'],
            ['c', 'd', 'a', 'b'],
        ]
        c.load(data=data)
        c.random_initialization(seed=42)

        changed, delta = c.recluster()
        assert isinstance(changed, int)
        assert changed >= 0

    def test_recluster_improves_objective(self):
        """recluster should not decrease objective function."""
        c = Clustering()
        c.min_count = 1
        c.clusters = 5

        data = [
            ['a', 'b', 'a', 'b', 'a'],
            ['b', 'a', 'b', 'a', 'b'],
            ['a', 'a', 'a', 'b', 'b'],
        ]
        c.load(data=data)
        c.random_initialization(seed=42)

        before = c.objective_function()
        c.recluster()
        after = c.objective_function()

        # Objective should not decrease
        assert after >= before - 1e-10


class TestReclusterUntilDone:
    """Tests for full clustering convergence."""

    def test_recluster_until_done_converges(self):
        """recluster_until_done reaches fixed point."""
        c = Clustering()
        c.min_count = 1
        c.clusters = 3

        data = [
            ['a', 'b', 'a', 'b'],
            ['b', 'a', 'b', 'a'],
        ]
        c.load(data=data)
        c.random_initialization(seed=42)

        final_of = c.recluster_until_done(maxiters=100)

        # Should have converged (no more changes)
        changed, _ = c.recluster()
        assert changed == 0

    def test_recluster_until_done_returns_objective(self):
        """recluster_until_done returns final objective."""
        c = Clustering()
        c.min_count = 1
        c.clusters = 3

        data = [['a', 'b', 'c']]
        c.load(data=data)
        c.random_initialization(seed=42)

        of = c.recluster_until_done(maxiters=10)
        assert isinstance(of, (float, np.floating))
        assert math.isfinite(of)


class TestClusterMethod:
    """Tests for the high-level cluster method."""

    def test_cluster_returns_mapping(self):
        """cluster returns word-to-cluster mapping."""
        c = Clustering()
        c.min_count = 1
        c.clusters = 3

        data = [
            ['the', 'cat', 'sat'],
            ['the', 'dog', 'ran'],
        ]

        result = c.cluster(data, seed=42)

        assert isinstance(result, dict)
        assert 'the' in result
        assert 'cat' in result
        assert 'dog' in result

    def test_cluster_excludes_boundary(self):
        """cluster result excludes boundary symbol."""
        c = Clustering()
        c.min_count = 1
        c.clusters = 3

        data = [['a', 'b', 'c']]
        result = c.cluster(data, seed=42)

        assert '<eos>' not in result

    def test_cluster_assigns_integers(self):
        """cluster assigns integer cluster IDs."""
        c = Clustering()
        c.min_count = 1
        c.clusters = 3

        data = [['a', 'b', 'c']]
        result = c.cluster(data, seed=42)

        for word, cluster_id in result.items():
            assert isinstance(cluster_id, (int, type(result['a'])))
            assert 0 <= cluster_id < c.clusters


class TestBestCluster:
    """Tests for bestCluster method."""

    def test_best_cluster_finds_improvement(self):
        """bestCluster finds improving move if one exists."""
        c = Clustering()
        c.min_count = 1
        c.clusters = 5

        data = [
            ['a', 'b', 'a', 'b', 'a', 'b'],
            ['b', 'a', 'b', 'a', 'b', 'a'],
        ]
        c.load(data=data)
        c.random_initialization(seed=42)

        # Try to improve assignment of word 1 (not boundary)
        improvement = c.bestCluster(1)
        assert isinstance(improvement, float)


class TestCalculateChange:
    """Tests for calculateChange method."""

    def test_calculate_change_same_cluster(self):
        """Moving to same cluster has zero change."""
        c = Clustering()
        c.min_count = 1
        c.clusters = 3

        data = [['a', 'b', 'c']]
        c.load(data=data)
        c.random_initialization(seed=42)

        i = 1  # Some word
        old_cluster = c.cluster_indices[i]

        left = right = c.cluster_bigrams[old_cluster, :]
        doubles = 0

        change = c.calculateChange(i, old_cluster, old_cluster, left, right, doubles)
        assert change == 0.0


class TestMoveWord:
    """Tests for move_word method."""

    def test_move_word_updates_indices(self):
        """move_word updates cluster_indices."""
        c = Clustering()
        c.min_count = 1
        c.clusters = 5

        data = [['a', 'b', 'c']]
        c.load(data=data)
        c.random_initialization(seed=42)

        i = 1
        old_cluster = c.cluster_indices[i]
        new_cluster = (old_cluster + 1) % c.clusters
        if new_cluster == 0:
            new_cluster = 1  # Avoid boundary cluster

        left = right = c.cluster_bigrams[old_cluster, :]
        doubles = 0

        c.move_word(i, old_cluster, new_cluster, left, right, doubles)

        assert c.cluster_indices[i] == new_cluster


class TestMutualInformationHelper:
    """Tests for mi helper function."""

    def test_mi_positive_values(self):
        """mi with positive values returns finite result."""
        result = neyessen.mi(10, 100, 100)
        assert math.isfinite(result)

    def test_mi_zero_bigram(self):
        """mi with zero bigram count returns 0."""
        result = neyessen.mi(0, 100, 100)
        assert result == 0.0

    def test_mi_independence(self):
        """mi returns 0 for independent distribution."""
        # If p(a,b) = p(a) * p(b), MI term is 0
        # bg = u1 * u2 means bg * log(bg / (u1 * u2)) = bg * log(1) = 0
        result = neyessen.mi(100, 10, 10)  # bg = 100, u1*u2 = 100
        assert result == 0.0


class TestLookup:
    """Tests for lookup method."""

    def test_lookup_known_word(self):
        """lookup returns index for known word."""
        c = Clustering()
        c.min_count = 1

        data = [['hello', 'world']]
        c.load(data=data)

        idx = c.lookup('hello')
        assert isinstance(idx, int)
        assert idx > 0  # Not boundary

    def test_lookup_unknown_word(self):
        """lookup returns unk index for unknown word."""
        c = Clustering()
        c.min_count = 100  # High threshold

        data = [['a', 'b', 'c']]  # All rare
        c.load(data=data)

        idx = c.lookup('nonexistent')
        assert idx == c.unkidx
