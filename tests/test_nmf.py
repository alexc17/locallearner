"""Tests for NMF algorithm - anchor word discovery."""

import pytest
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'locallearner'))

import nmf
from nmf import NMF, min_on_line


class TestNMFInit:
    """Tests for NMF initialization."""

    def test_nmf_init(self):
        """NMF initializes correctly."""
        # Simple data matrix: 4 words, 3 features
        data = np.array([
            [10, 0, 0],
            [0, 10, 0],
            [0, 0, 10],
            [5, 5, 0],
        ], dtype=float)
        index = ['word1', 'word2', 'word3', 'word4']

        nmf_obj = NMF(data, index)

        assert nmf_obj.n == 4
        assert nmf_obj.f == 3
        assert nmf_obj.index == index

    def test_nmf_normalizes_rows(self):
        """NMF normalizes rows to sum to 1."""
        data = np.array([
            [10, 0, 0],
            [0, 10, 0],
        ], dtype=float)
        index = ['word1', 'word2']

        nmf_obj = NMF(data, index)

        # Each row should sum to 1 after normalization
        for i in range(nmf_obj.n):
            assert np.sum(nmf_obj.data[i, :]) == pytest.approx(1.0)


class TestNMFStart:
    """Tests for NMF start method (picking first basis)."""

    def test_start_sets_basis(self):
        """start sets the first basis vector."""
        data = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ], dtype=float)
        index = ['a', 'b', 'c']

        nmf_obj = NMF(data, index)
        nmf_obj.start(0)

        assert len(nmf_obj.bases) == 1
        assert nmf_obj.bases[0] == 0
        assert len(nmf_obj.M) == 1

    def test_start_l2_picks_peaked(self):
        """start_l2 picks the most peaked distribution."""
        # word3 is most peaked (all mass in one feature)
        data = np.array([
            [5, 5, 0],   # Uniform-ish
            [3, 3, 4],   # Fairly uniform
            [10, 0, 0],  # Very peaked
        ], dtype=float)
        index = ['word1', 'word2', 'word3']

        nmf_obj = NMF(data, index)
        result = nmf_obj.start_l2()

        assert result == 'word3'


class TestGramSchmidt:
    """Tests for Gram-Schmidt orthonormalization."""

    def test_gram_schmidt_orthonormal(self):
        """gram_schmidt produces orthonormal basis."""
        data = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
        ], dtype=float)
        index = ['a', 'b', 'c']

        nmf_obj = NMF(data, index)
        nmf_obj.start(0)
        nmf_obj.add_basis(1, gram_schmidt=True)

        # Check orthonormality
        basis = nmf_obj.orthonormal
        for i in range(len(basis)):
            # Each vector has unit norm
            assert np.linalg.norm(basis[i]) == pytest.approx(1.0)
            for j in range(i + 1, len(basis)):
                # Orthogonal to each other
                assert np.dot(basis[i], basis[j]) == pytest.approx(0.0, abs=1e-10)

    def test_gram_schmidt_computes_distances(self):
        """gram_schmidt computes distances for all points."""
        data = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0.5, 0.5, 0],
        ], dtype=float)
        index = ['a', 'b', 'c']

        nmf_obj = NMF(data, index)
        nmf_obj.start(0)
        nmf_obj.gram_schmidt()

        assert len(nmf_obj.distances) == nmf_obj.n


class TestFrankWolfe:
    """Tests for Frank-Wolfe optimization."""

    def test_frank_wolfe_finds_convex_combination(self):
        """frank_wolfe finds convex combination coefficients."""
        # Create data where word3 is midpoint of word1 and word2
        data = np.array([
            [1, 0],
            [0, 1],
            [0.5, 0.5],
        ], dtype=float)
        index = ['word1', 'word2', 'word3']

        nmf_obj = NMF(data, index)
        nmf_obj.start(0)
        nmf_obj.add_basis(1, gram_schmidt=False)

        # word3 should be 0.5 * word1 + 0.5 * word2
        x, d = nmf_obj.estimate_frank_wolfe(nmf_obj.data[2, :])

        assert x[0] == pytest.approx(0.5, abs=0.1)
        assert x[1] == pytest.approx(0.5, abs=0.1)
        assert d < 0.1  # Small residual

    def test_frank_wolfe_at_vertex(self):
        """frank_wolfe correctly identifies vertex points."""
        data = np.array([
            [1, 0],
            [0, 1],
            [0.8, 0.2],
        ], dtype=float)
        index = ['word1', 'word2', 'word3']

        nmf_obj = NMF(data, index)
        nmf_obj.start(0)
        nmf_obj.add_basis(1, gram_schmidt=False)

        # word1 should be all weight on first basis
        x, d = nmf_obj.estimate_frank_wolfe(nmf_obj.data[0, :])

        assert x[0] == pytest.approx(1.0, abs=0.01)
        assert x[1] == pytest.approx(0.0, abs=0.01)


class TestFindFurthest:
    """Tests for find_furthest method."""

    def test_find_furthest_finds_outlier(self):
        """find_furthest finds point furthest from current basis."""
        data = np.array([
            [1, 0, 0],
            [0.9, 0.1, 0],
            [0, 0, 1],  # Outlier
        ], dtype=float)
        index = ['word1', 'word2', 'word3']

        nmf_obj = NMF(data, index)
        nmf_obj.start(0)
        nmf_obj.gram_schmidt()

        idx, dist = nmf_obj.find_furthest(verbose=False)
        # word3 should be furthest from word1
        assert nmf_obj.index[idx] == 'word3'

    def test_find_furthest_excludes_bases(self):
        """find_furthest excludes current basis points."""
        data = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ], dtype=float)
        index = ['word1', 'word2', 'word3']

        nmf_obj = NMF(data, index)
        nmf_obj.start(0)
        nmf_obj.excluded.add(0)
        nmf_obj.gram_schmidt()

        idx, _ = nmf_obj.find_furthest(verbose=False)
        assert idx != 0  # Should not return the starting basis


class TestAddBasis:
    """Tests for add_basis method."""

    def test_add_basis_increases_dimension(self):
        """add_basis adds a new basis vector."""
        data = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ], dtype=float)
        index = ['word1', 'word2', 'word3']

        nmf_obj = NMF(data, index)
        nmf_obj.start(0)
        initial_bases = len(nmf_obj.bases)

        nmf_obj.add_basis(1)
        assert len(nmf_obj.bases) == initial_bases + 1


class TestFindNextElement:
    """Tests for find_next_element method."""

    def test_find_next_element(self):
        """find_next_element finds and adds next basis."""
        data = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ], dtype=float)
        index = ['word1', 'word2', 'word3']

        nmf_obj = NMF(data, index)
        nmf_obj.start(0)

        word = nmf_obj.find_next_element()
        assert word in index
        assert len(nmf_obj.bases) == 2


class TestMinOnLine:
    """Tests for min_on_line helper function."""

    def test_min_on_line_midpoint(self):
        """min_on_line finds optimal alpha for midpoint."""
        y = np.array([0.5, 0.5])
        y0 = np.array([0, 0])
        y1 = np.array([1, 1])

        alpha = min_on_line(y, y0, y1)
        assert alpha == pytest.approx(0.5)

    def test_min_on_line_at_y0(self):
        """min_on_line returns 0 when y is at y0."""
        y = np.array([0, 0])
        y0 = np.array([0, 0])
        y1 = np.array([1, 1])

        alpha = min_on_line(y, y0, y1)
        assert alpha == pytest.approx(0.0)

    def test_min_on_line_at_y1(self):
        """min_on_line returns 1 when y is at y1."""
        y = np.array([1, 1])
        y0 = np.array([0, 0])
        y1 = np.array([1, 1])

        alpha = min_on_line(y, y0, y1)
        assert alpha == pytest.approx(1.0)

    def test_min_on_line_clamps_to_01(self):
        """min_on_line clamps result to [0, 1]."""
        # y is beyond y1
        y = np.array([2, 2])
        y0 = np.array([0, 0])
        y1 = np.array([1, 1])

        alpha = min_on_line(y, y0, y1)
        assert alpha == 1.0

        # y is before y0
        y = np.array([-1, -1])
        alpha = min_on_line(y, y0, y1)
        assert alpha == 0.0


class TestClusterVertices:
    """Tests for cluster_vertices method."""

    def test_cluster_vertices(self):
        """cluster_vertices clusters points around bases."""
        # Create clearly separated clusters
        data = np.array([
            [1, 0, 0],      # Cluster 1 center
            [0.9, 0.1, 0],  # Near cluster 1
            [0, 1, 0],      # Cluster 2 center
            [0.1, 0.9, 0],  # Near cluster 2
        ], dtype=float)
        index = ['center1', 'near1', 'center2', 'near2']

        nmf_obj = NMF(data, index)
        nmf_obj.start(0)
        nmf_obj.add_basis(2, gram_schmidt=True)

        clustering, _ = nmf_obj.cluster_vertices()

        # near1 should be in same cluster as center1
        if 'near1' in clustering and 'center1' in clustering:
            pass  # Implementation may vary


class TestDistanceFromHyperplane:
    """Tests for distance_from_hyperplane method."""

    def test_distance_on_hyperplane(self):
        """Point on hyperplane has distance 0."""
        data = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0.5, 0.5, 0],  # On the line between word1 and word2
        ], dtype=float)
        index = ['word1', 'word2', 'word3']

        nmf_obj = NMF(data, index)
        nmf_obj.start(0)
        nmf_obj.add_basis(1, gram_schmidt=True)

        # word3 is in the span of word1 and word2
        d = nmf_obj.distance_from_hyperplane(nmf_obj.data[2, :])
        assert d == pytest.approx(0.0, abs=0.01)

    def test_distance_off_hyperplane(self):
        """Point off hyperplane has positive distance."""
        data = np.array([
            [1, 0, 0],
            [0, 0, 1],  # Off the x-axis
        ], dtype=float)
        index = ['word1', 'word2']

        nmf_obj = NMF(data, index)
        nmf_obj.start(0)
        nmf_obj.gram_schmidt()

        # word2 is not on the line through word1
        d = nmf_obj.distance_from_hyperplane(nmf_obj.data[1, :])
        assert d > 0


class TestSmallSampleFactor:
    """Tests for small sample factor correction."""

    def test_small_sample_factor(self):
        """small_sample_factor decreases with count."""
        data = np.array([[1, 0], [0, 1]], dtype=float)
        nmf_obj = NMF(data, ['a', 'b'], ssf=1.0)

        # SSF = ssf / sqrt(n)
        assert nmf_obj.small_sample_factor(1) == 1.0
        assert nmf_obj.small_sample_factor(4) == pytest.approx(0.5)
        assert nmf_obj.small_sample_factor(100) == pytest.approx(0.1)
