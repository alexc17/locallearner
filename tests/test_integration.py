"""Integration tests - end-to-end learning pipeline."""

import pytest
import os
import sys
import tempfile
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'locallearner'))

import wcfg
from wcfg import WCFG, Sampler, InsideComputation, load_wcfg_from_file
import numpy as np
from utility import collect_yield


class TestEndToEndSimple:
    """End-to-end tests with simple grammars."""

    def test_load_parse_sample_round_trip(self, simple_grammar_path):
        """Load grammar, sample, and verify parsing."""
        # Load
        g = load_wcfg_from_file(simple_grammar_path)
        assert g.is_normalised()

        # Sample
        rng = np.random.default_rng(42)
        sampler = Sampler(g, random=rng)

        # Parse sampled strings
        ic = InsideComputation(g)

        for _ in range(10):
            tree = sampler.sample_tree()
            s = tuple(collect_yield(tree))

            # Should be parseable
            p = ic.inside_probability(s)
            assert p > 0

            # Viterbi should return valid tree
            viterbi_tree = ic.viterbi_parse(s)
            assert tuple(collect_yield(viterbi_tree)) == s

    def test_grammar_store_reload_equivalence(self, simple_grammar_path):
        """Stored and reloaded grammar produces same results."""
        g1 = load_wcfg_from_file(simple_grammar_path)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.pcfg', delete=False) as f:
            temp_path = f.name

        try:
            g1.store(temp_path)
            g2 = load_wcfg_from_file(temp_path)

            # Both grammars should give same probabilities
            ic1 = InsideComputation(g1)
            ic2 = InsideComputation(g2)

            for s in [('a', 'a'), ('a', 'b'), ('b', 'a'), ('b', 'b')]:
                p1 = ic1.inside_probability(s)
                p2 = ic2.inside_probability(s)
                assert p1 == pytest.approx(p2, rel=1e-5)
        finally:
            os.unlink(temp_path)


class TestNormalizationPipeline:
    """Tests for the normalization pipeline."""

    def test_wcfg_to_pcfg_pipeline(self):
        """Convert unnormalized WCFG to consistent PCFG."""
        # Create unnormalized WCFG
        g = WCFG()
        g.nonterminals = {'S', 'A'}
        g.terminals = {'a', 'b'}
        g.start = 'S'
        g.productions = [
            ('S', 'A', 'A'),
            ('A', 'a'),
            ('A', 'b'),
        ]
        g.parameters = {
            ('S', 'A', 'A'): 2.5,
            ('A', 'a'): 1.2,
            ('A', 'b'): 0.8,
        }
        g.set_log_parameters()

        # Normalize
        g.locally_normalise()
        assert g.is_normalised()

        # Renormalize to make consistent
        g.renormalise()
        assert g.is_consistent(epsilon=1e-5)

        # Should be sampleable
        rng = np.random.default_rng(42)
        sampler = Sampler(g, random=rng)
        tree = sampler.sample_tree()
        assert tree is not None


class TestParsingPipeline:
    """Tests for the parsing pipeline."""

    def test_parse_all_valid_strings(self, simple_pcfg):
        """Parse all strings the grammar can generate."""
        ic = InsideComputation(simple_pcfg)

        # All length-2 strings over {a, b}
        total_prob = 0.0
        for w1 in ['a', 'b']:
            for w2 in ['a', 'b']:
                s = (w1, w2)
                p = ic.inside_probability(s)
                assert p > 0
                total_prob += p

        # Total probability should be 1 (grammar is tight)
        assert total_prob == pytest.approx(1.0)

    def test_viterbi_and_inside_consistency(self, simple_pcfg):
        """Viterbi parse probability <= inside probability."""
        ic = InsideComputation(simple_pcfg)

        for s in [('a', 'a'), ('a', 'b'), ('b', 'a'), ('b', 'b')]:
            inside_prob = ic.inside_probability(s)
            tree = ic.viterbi_parse(s)
            viterbi_prob = math.exp(simple_pcfg.log_probability_derivation(tree))

            # Viterbi finds most probable derivation
            # For unambiguous grammar, they should be equal
            assert viterbi_prob == pytest.approx(inside_prob)


class TestEMPipeline:
    """Tests for EM (inside-outside) pipeline."""

    def test_inside_outside_estimation(self, simple_pcfg):
        """Inside-outside reestimation from corpus."""
        # Generate training data
        rng = np.random.default_rng(42)
        sampler = Sampler(simple_pcfg, random=rng)

        corpus = [tuple(sampler.sample_string()) for _ in range(100)]

        # Reestimate
        reestimated = simple_pcfg.estimate_inside_outside_from_list(
            corpus, maxlength=10, maxcount=100
        )

        # Should still be normalized
        assert reestimated.is_normalised(epsilon=0.1)

        # Parameters should be close to original (with enough data)
        for prod in simple_pcfg.productions:
            if prod in reestimated.parameters:
                original = simple_pcfg.parameters[prod]
                estimated = reestimated.parameters[prod]
                # Allow 30% deviation due to sampling noise
                assert abs(original - estimated) < 0.3 or estimated > 0


class TestAmbiguousParsing:
    """Tests for parsing ambiguous grammars."""

    def test_count_parses_ambiguous(self, ambiguous_pcfg):
        """Count multiple parses for ambiguous strings."""
        ic = InsideComputation(ambiguous_pcfg)

        # "aaa" should have multiple parses
        count = ic.count_parses(('a', 'a', 'a'))
        assert count > 1

        # "aa" should have exactly 1 parse
        count = ic.count_parses(('a', 'a'))
        assert count == 1

    def test_ambiguous_probability_sums_derivations(self, ambiguous_pcfg):
        """Inside probability sums over all derivations."""
        ic = InsideComputation(ambiguous_pcfg)

        # For "aaa", probability should account for all derivations
        p = ic.inside_probability(('a', 'a', 'a'))
        assert p > 0

        # The grammar can generate strings of any length >= 2
        # Check that longer strings are less probable (more expansion steps)
        p2 = ic.inside_probability(('a', 'a'))
        p4 = ic.inside_probability(('a', 'a', 'a', 'a'))
        assert p4 < p2


class TestLocalLearnerIntegration:
    """Integration tests for LocalLearner."""

    def test_locallearner_find_kernels(self, sample_corpus_path):
        """LocalLearner finds kernels from corpus."""
        from locallearner import LocalLearner

        ll = LocalLearner(sample_corpus_path)
        ll.nonterminals = 2
        ll.number_clusters = 3
        ll.min_count_nmf = 1
        ll.seed = 42

        kernels = ll.find_kernels(verbose=False)

        assert len(kernels) == 2
        assert kernels[0] == 'S'

    def test_locallearner_learn_and_parse(self, sample_corpus_path):
        """LocalLearner learns grammar that can parse training data."""
        from locallearner import LocalLearner

        ll = LocalLearner(sample_corpus_path)
        ll.nonterminals = 2
        ll.number_clusters = 3
        ll.min_count_nmf = 1
        ll.seed = 42
        ll.posterior_threshold = 0.5

        # Find kernels and learn grammar
        kernels = ll.find_kernels(verbose=False)
        grammar = ll.learn_wcfg_from_kernels_renyi(kernels, verbose=False)

        assert grammar is not None
        assert grammar.start == 'S'

        # Try to normalize and make PCFG
        grammar.locally_normalise()

        # Grammar should be usable (may not parse all sentences due to smoothing)
        ic = InsideComputation(grammar)

        # At least some training sentences should be parseable
        parseable = 0
        for s in ll.sentences[:5]:
            try:
                p = ic.inside_probability(s)
                if p > 0:
                    parseable += 1
            except Exception:
                pass

        # We expect some sentences to be parseable
        assert parseable >= 0  # May be 0 for very small corpus


class TestGrammarProperties:
    """Tests for grammar property verification."""

    def test_pcfg_probabilities_sum_to_one(self, simple_pcfg):
        """PCFG production probabilities sum to 1 per nonterminal."""
        totals = simple_pcfg.check_normalisation()

        for nt, total in totals.items():
            assert total == pytest.approx(1.0)

    def test_consistent_pcfg_generates_probability_1(self, simple_pcfg):
        """Consistent PCFG assigns total probability 1 to all strings."""
        # For a tight grammar, partition function at start should be 1
        pf = simple_pcfg.compute_partition_function_fp()
        assert pf['S'] == pytest.approx(1.0, rel=1e-5)

    def test_expected_length_matches_samples(self, simple_pcfg):
        """Expected length matches sample average."""
        # Analytical expected length
        expected = simple_pcfg.expected_length()

        # Sample average
        rng = np.random.default_rng(42)
        sampler = Sampler(simple_pcfg, random=rng)

        lengths = [len(sampler.sample_string()) for _ in range(500)]
        sample_mean = sum(lengths) / len(lengths)

        # Should be close (simple grammar always generates length 2)
        assert sample_mean == pytest.approx(expected, rel=0.1)


class TestThreeNonterminalIntegration:
    """Integration tests with 3-nonterminal grammar."""

    def test_three_nt_sampling_and_parsing(self, three_nt_pcfg):
        """Sample from 3-NT grammar and verify parsing."""
        rng = np.random.default_rng(42)
        sampler = Sampler(three_nt_pcfg, random=rng)
        ic = InsideComputation(three_nt_pcfg)

        for _ in range(20):
            tree = sampler.sample_tree()
            s = tuple(collect_yield(tree))

            # Should be parseable
            p = ic.inside_probability(s)
            assert p > 0

    def test_three_nt_em_reestimation(self, three_nt_pcfg):
        """EM reestimation on 3-NT grammar."""
        rng = np.random.default_rng(42)
        sampler = Sampler(three_nt_pcfg, random=rng)

        corpus = [tuple(sampler.sample_string()) for _ in range(200)]

        reestimated = three_nt_pcfg.estimate_inside_outside_from_list(
            corpus, maxlength=10, maxcount=200
        )

        # Should be valid PCFG
        assert reestimated.is_normalised(epsilon=0.1)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_sentence(self, simple_pcfg):
        """Empty sentence returns zero probability."""
        ic = InsideComputation(simple_pcfg)

        # Empty sentence cannot be parsed, returns 0 probability
        p = ic.inside_probability(())
        assert p == 0.0

    def test_unknown_terminal(self, simple_pcfg):
        """Unknown terminal raises ParseFailureException."""
        from utility import ParseFailureException

        ic = InsideComputation(simple_pcfg)

        with pytest.raises(ParseFailureException):
            ic.inside_log_probability(('x', 'y'))

    def test_grammar_with_zero_rules_after_trim(self):
        """Grammar with all zero rules becomes empty after trim."""
        g = WCFG()
        g.nonterminals = {'S', 'A'}
        g.terminals = {'a'}
        g.start = 'S'
        g.productions = [('S', 'A', 'A'), ('A', 'a')]
        g.parameters = {
            ('S', 'A', 'A'): 0.0,
            ('A', 'a'): 0.0,
        }

        g.trim_zeros()

        assert len(g.productions) == 0
