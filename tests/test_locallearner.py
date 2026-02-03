"""Tests for LocalLearner - main learning algorithm."""

import pytest
import os
import sys
import tempfile
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'locallearner'))

import locallearner
from locallearner import LocalLearner


class TestLocalLearnerInit:
    """Tests for LocalLearner initialization."""

    def test_init_from_file(self, sample_corpus_path):
        """LocalLearner initializes from corpus file."""
        ll = LocalLearner(sample_corpus_path)

        assert ll.number_samples > 0
        assert ll.number_tokens > 0
        assert ll.alphabet_size > 0

    def test_init_counts_correctly(self, sample_corpus_path):
        """LocalLearner counts sentences and tokens correctly."""
        ll = LocalLearner(sample_corpus_path)

        # sample_corpus.txt has 12 sentences, 4 words each
        assert ll.number_samples == 12
        assert ll.number_tokens == 48  # 12 * 4

    def test_init_builds_vocabulary(self, sample_corpus_path):
        """LocalLearner builds vocabulary from corpus."""
        ll = LocalLearner(sample_corpus_path)

        assert 'a' in ll.lexical_counts
        assert 'b' in ll.lexical_counts
        assert ll.lexical_counts['a'] > 0
        assert ll.lexical_counts['b'] > 0

    def test_init_default_hyperparameters(self, sample_corpus_path):
        """LocalLearner has sensible default hyperparameters."""
        ll = LocalLearner(sample_corpus_path)

        assert ll.nonterminals >= 2
        assert ll.number_clusters > 0
        assert ll.width >= 1


class TestLocalLearnerClustering:
    """Tests for clustering step."""

    def test_do_clustering(self, sample_corpus_path):
        """do_clustering creates word clusters."""
        ll = LocalLearner(sample_corpus_path)
        ll.number_clusters = 3

        ll.do_clustering()

        assert hasattr(ll, 'clusters')
        assert isinstance(ll.clusters, dict)
        # All vocabulary words should have cluster assignments
        for word in ll.idx2word:
            assert word in ll.clusters

    def test_clustering_assigns_integers(self, sample_corpus_path):
        """Clustering assigns integer cluster IDs."""
        ll = LocalLearner(sample_corpus_path)
        ll.number_clusters = 3

        ll.do_clustering()

        for word, cluster_id in ll.clusters.items():
            assert isinstance(cluster_id, (int, np.integer))


class TestLocalLearnerFeatures:
    """Tests for feature computation."""

    def test_compute_unigram_features(self, sample_corpus_path):
        """compute_unigram_features creates feature matrix."""
        ll = LocalLearner(sample_corpus_path)
        ll.number_clusters = 3
        ll.width = 1
        ll.stride = ll.number_clusters + 1
        ll.number_features = 2 * ll.width * ll.stride

        ll.do_clustering()
        ll.set_start_vector()
        ll.compute_unigram_features()

        assert hasattr(ll, 'unigram_features')
        # Shape: (alphabet_size + 1, number_features)
        assert ll.unigram_features.shape[0] == ll.alphabet_size + 1
        assert ll.unigram_features.shape[1] == ll.number_features

    def test_set_start_vector(self, sample_corpus_path):
        """set_start_vector creates correct start vector."""
        ll = LocalLearner(sample_corpus_path)
        ll.number_clusters = 3
        ll.width = 1
        ll.stride = ll.number_clusters + 1
        ll.number_features = 2 * ll.width * ll.stride

        ll.set_start_vector()

        assert hasattr(ll, 'start_vector')
        assert len(ll.start_vector) == ll.number_features


class TestLocalLearnerNMF:
    """Tests for NMF step."""

    def test_do_nmf(self, sample_corpus_path):
        """do_nmf finds kernel words."""
        ll = LocalLearner(sample_corpus_path)
        ll.nonterminals = 2
        ll.number_clusters = 3
        ll.min_count_nmf = 1  # Accept all words
        ll.width = 1
        ll.stride = ll.number_clusters + 1
        ll.number_features = 2 * ll.width * ll.stride

        ll.do_clustering()
        ll.set_start_vector()
        ll.compute_unigram_features()
        ll.do_nmf(verbose=False)

        assert hasattr(ll, 'kernels')
        assert len(ll.kernels) == ll.nonterminals
        # First kernel should be start symbol
        assert ll.kernels[0] == ll.start_symbol

    def test_init_nmf_with_kernels(self, sample_corpus_path):
        """init_nmf_with_kernels uses provided kernels."""
        ll = LocalLearner(sample_corpus_path)
        ll.nonterminals = 2
        ll.number_clusters = 3
        ll.min_count_nmf = 1
        ll.width = 1
        ll.stride = ll.number_clusters + 1
        ll.number_features = 2 * ll.width * ll.stride

        ll.do_clustering()
        ll.set_start_vector()
        ll.compute_unigram_features()

        # Use 'a' as the kernel
        kernels = ['S', 'a']
        ll.init_nmf_with_kernels(kernels)

        assert ll.kernels == kernels


class TestLocalLearnerParameters:
    """Tests for parameter estimation."""

    def test_compute_unary_parameters_fw(self, sample_corpus_path):
        """compute_unary_parameters_fw estimates lexical probabilities."""
        ll = LocalLearner(sample_corpus_path)
        ll.nonterminals = 2
        ll.number_clusters = 3
        ll.min_count_nmf = 1
        ll.width = 1
        ll.stride = ll.number_clusters + 1
        ll.number_features = 2 * ll.width * ll.stride

        ll.do_clustering()
        ll.set_start_vector()
        ll.compute_unigram_features()
        ll.do_nmf(verbose=False)
        ll.compute_unary_parameters_fw()

        assert hasattr(ll, 'unary_parameters')
        # Should have parameters for each (nonterminal, terminal) pair
        assert len(ll.unary_parameters) > 0

        # Parameters should be non-negative
        for prod, param in ll.unary_parameters.items():
            assert param >= 0

    def test_compute_binary_parameters_renyi(self, sample_corpus_path):
        """compute_binary_parameters_renyi estimates binary rule weights."""
        ll = LocalLearner(sample_corpus_path)
        ll.nonterminals = 2
        ll.number_clusters = 3
        ll.min_count_nmf = 1
        ll.width = 1
        ll.stride = ll.number_clusters + 1
        ll.number_features = 2 * ll.width * ll.stride
        ll.renyi = 2
        ll.posterior_threshold = 0.5

        ll.do_clustering()
        ll.set_start_vector()
        ll.compute_unigram_features()
        ll.do_nmf(verbose=False)
        ll.compute_unary_parameters_fw()
        ll.compute_clustered_bigram_features()
        ll.compute_binary_parameters_renyi()

        assert hasattr(ll, 'binary_parameters')
        assert len(ll.binary_parameters) > 0

        # Parameters should be non-negative
        for prod, param in ll.binary_parameters.items():
            assert param >= 0


class TestLocalLearnerGrammar:
    """Tests for grammar construction."""

    def test_set_nonterminal_labels(self, sample_corpus_path):
        """set_nonterminal_labels creates nonterminal names."""
        ll = LocalLearner(sample_corpus_path)
        ll.nonterminals = 2
        ll.number_clusters = 3
        ll.min_count_nmf = 1
        ll.width = 1
        ll.stride = ll.number_clusters + 1
        ll.number_features = 2 * ll.width * ll.stride

        ll.do_clustering()
        ll.set_start_vector()
        ll.compute_unigram_features()
        ll.do_nmf(verbose=False)
        ll.set_nonterminal_labels()

        assert hasattr(ll, 'kernel2nt')
        assert ll.kernel2nt[ll.kernels[0]] == 'S'
        # Other kernels should be NT_<word>
        for kernel in ll.kernels[1:]:
            assert ll.kernel2nt[kernel].startswith('NT_')

    def test_make_raw_wcfg(self, sample_corpus_path):
        """make_raw_wcfg creates a WCFG object."""
        ll = LocalLearner(sample_corpus_path)
        ll.nonterminals = 2
        ll.number_clusters = 3
        ll.min_count_nmf = 1
        ll.width = 1
        ll.stride = ll.number_clusters + 1
        ll.number_features = 2 * ll.width * ll.stride
        ll.binary_smoothing = 1e-10
        ll.unary_smoothing = 1e-10
        ll.posterior_threshold = 0.5

        ll.do_clustering()
        ll.set_start_vector()
        ll.compute_unigram_features()
        ll.do_nmf(verbose=False)
        ll.compute_unary_parameters_fw()
        ll.compute_clustered_bigram_features()
        ll.compute_binary_parameters_renyi()
        ll.set_nonterminal_labels()
        ll.make_raw_wcfg()

        assert hasattr(ll, 'output_grammar')
        assert ll.output_grammar.start == 'S'
        assert len(ll.output_grammar.productions) > 0


class TestLocalLearnerFindKernels:
    """Tests for find_kernels method."""

    def test_find_kernels(self, sample_corpus_path):
        """find_kernels discovers anchor words."""
        ll = LocalLearner(sample_corpus_path)
        ll.nonterminals = 2
        ll.number_clusters = 3
        ll.min_count_nmf = 1
        ll.seed = 42

        kernels = ll.find_kernels(verbose=False)

        assert len(kernels) == ll.nonterminals
        assert kernels[0] == 'S'  # Start symbol


class TestLocalLearnerHelpers:
    """Tests for helper methods."""

    def test_terminal_expectation(self, sample_corpus_path):
        """terminal_expectation computes expected frequency per sentence."""
        ll = LocalLearner(sample_corpus_path)

        # Check expectation for 'a'
        # terminal_expectation = count / number_samples
        # This can be > 1 if a word appears multiple times per sentence
        exp = ll.terminal_expectation('a')
        assert exp > 0
        # Expected value is count/sentences, which is meaningful for grammar learning

    def test_convert_production(self, sample_corpus_path):
        """convert_production maps kernels to nonterminal names."""
        ll = LocalLearner(sample_corpus_path)
        ll.nonterminals = 2
        ll.number_clusters = 3
        ll.min_count_nmf = 1
        ll.width = 1
        ll.stride = ll.number_clusters + 1
        ll.number_features = 2 * ll.width * ll.stride

        ll.do_clustering()
        ll.set_start_vector()
        ll.compute_unigram_features()
        ll.do_nmf(verbose=False)
        ll.set_nonterminal_labels()

        # Test lexical production
        lexical_prod = (ll.kernels[0], 'a')
        converted = ll.convert_production(lexical_prod)
        assert converted[0] == 'S'
        assert converted[1] == 'a'


class TestRenyiDivergence:
    """Tests for renyi_divergence helper function."""

    def test_renyi_divergence_identical(self):
        """Renyi divergence of identical distributions is 0."""
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([1.0, 2.0, 3.0])

        d = locallearner.renyi_divergence(v1, v2, alpha=2)
        assert d == pytest.approx(0.0, abs=1e-10)

    def test_renyi_divergence_nonnegative(self):
        """Renyi divergence is non-negative."""
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([3.0, 2.0, 1.0])

        d = locallearner.renyi_divergence(v1, v2, alpha=2)
        assert d >= 0

    def test_renyi_divergence_infinity_disjoint(self):
        """Renyi divergence is infinity for disjoint support."""
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 0.0, 1.0])

        d = locallearner.renyi_divergence(v1, v2, alpha=2)
        assert d == float('inf')


class TestLocalLearnerLearnWCFGFromKernels:
    """Tests for learn_wcfg_from_kernels_renyi method."""

    def test_learn_wcfg_from_kernels_renyi(self, sample_corpus_path):
        """learn_wcfg_from_kernels_renyi produces a grammar."""
        ll = LocalLearner(sample_corpus_path)
        ll.nonterminals = 2
        ll.number_clusters = 3
        ll.min_count_nmf = 1
        ll.seed = 42
        ll.posterior_threshold = 0.5

        # First find kernels
        kernels = ll.find_kernels(verbose=False)

        # Then learn grammar from those kernels
        grammar = ll.learn_wcfg_from_kernels_renyi(kernels, verbose=False)

        assert grammar is not None
        assert grammar.start == 'S'
        assert len(grammar.productions) > 0
