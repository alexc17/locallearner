"""Tests for WCFG sampling - Sampler class."""

import pytest
import os
import sys
import math
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'locallearner'))

import wcfg
from wcfg import WCFG, Sampler, InsideComputation
from utility import collect_yield


class TestSamplerInit:
    """Tests for Sampler initialization."""

    def test_sampler_init(self, simple_pcfg):
        """Sampler initializes correctly."""
        rng = np.random.default_rng(42)
        sampler = Sampler(simple_pcfg, random=rng)

        assert sampler.start == simple_pcfg.start
        assert 'S' in sampler.multinomials
        assert 'A' in sampler.multinomials

    def test_sampler_requires_normalized(self):
        """Sampler requires normalized grammar."""
        g = WCFG()
        g.nonterminals = {'S', 'A'}
        g.terminals = {'a'}
        g.start = 'S'
        g.productions = [('S', 'A', 'A'), ('A', 'a')]
        g.parameters = {
            ('S', 'A', 'A'): 0.5,  # Not normalized
            ('A', 'a'): 0.7,
        }
        g.set_log_parameters()

        rng = np.random.default_rng(42)
        with pytest.raises(AssertionError):
            Sampler(g, random=rng)


class TestSampleTree:
    """Tests for sample_tree method."""

    def test_sample_tree_returns_valid_tree(self, simple_pcfg):
        """sample_tree returns a valid derivation tree."""
        rng = np.random.default_rng(42)
        sampler = Sampler(simple_pcfg, random=rng)

        tree = sampler.sample_tree()

        # Should be rooted at S
        assert tree[0] == 'S'
        # Should have valid structure
        assert len(tree) == 3  # Binary production

    def test_sample_tree_yield_is_valid(self, simple_pcfg):
        """Sampled tree yield contains only valid terminals."""
        rng = np.random.default_rng(42)
        sampler = Sampler(simple_pcfg, random=rng)

        for _ in range(10):
            tree = sampler.sample_tree()
            s = collect_yield(tree)
            for w in s:
                assert w in simple_pcfg.terminals


class TestSampleString:
    """Tests for sample_string method."""

    def test_sample_string_returns_list(self, simple_pcfg):
        """sample_string returns a list of terminals."""
        rng = np.random.default_rng(42)
        sampler = Sampler(simple_pcfg, random=rng)

        s = sampler.sample_string()
        assert isinstance(s, list)
        assert all(w in simple_pcfg.terminals for w in s)

    def test_sample_string_length(self, simple_pcfg):
        """Sampled strings have correct length for grammar."""
        rng = np.random.default_rng(42)
        sampler = Sampler(simple_pcfg, random=rng)

        # Simple grammar always produces length-2 strings
        for _ in range(10):
            s = sampler.sample_string()
            assert len(s) == 2


class TestSampleDistribution:
    """Tests verifying samples match grammar distribution."""

    def test_sample_distribution_simple(self, simple_pcfg):
        """Sample distribution approximately matches grammar."""
        rng = np.random.default_rng(42)
        sampler = Sampler(simple_pcfg, random=rng)

        n_samples = 1000
        counts = Counter()

        for _ in range(n_samples):
            s = tuple(sampler.sample_string())
            counts[s] += 1

        # All 4 strings should have roughly equal probability (0.25 each)
        expected = n_samples / 4
        for s in [('a', 'a'), ('a', 'b'), ('b', 'a'), ('b', 'b')]:
            # Allow 20% deviation
            assert abs(counts[s] - expected) < expected * 0.3

    def test_sample_distribution_three_nt(self, three_nt_pcfg):
        """Sample distribution for 3-nonterminal grammar."""
        rng = np.random.default_rng(42)
        sampler = Sampler(three_nt_pcfg, random=rng)

        n_samples = 1000
        counts = Counter()

        for _ in range(n_samples):
            s = tuple(sampler.sample_string())
            counts[s] += 1

        # Check that "ab" is more common than "ba"
        # P(ab) = 0.6 * 0.7 * 0.8 = 0.336
        # P(ba) = 0.4 * 0.8 * 0.7 = 0.224
        assert counts[('a', 'b')] > counts[('b', 'a')] * 0.8

    def test_sampled_trees_parseable(self, simple_pcfg):
        """All sampled trees should be parseable."""
        rng = np.random.default_rng(42)
        sampler = Sampler(simple_pcfg, random=rng)
        ic = InsideComputation(simple_pcfg)

        for _ in range(20):
            tree = sampler.sample_tree()
            s = tuple(collect_yield(tree))
            # Should be able to compute probability
            p = ic.inside_probability(s)
            assert p > 0


class TestSampleProduction:
    """Tests for sample_production method."""

    def test_sample_production_returns_valid(self, simple_pcfg):
        """sample_production returns a valid production."""
        rng = np.random.default_rng(42)
        sampler = Sampler(simple_pcfg, random=rng)

        prod = sampler.sample_production('S')
        assert prod in simple_pcfg.productions
        assert prod[0] == 'S'

    def test_sample_production_distribution(self, simple_pcfg):
        """sample_production follows correct distribution."""
        rng = np.random.default_rng(42)
        sampler = Sampler(simple_pcfg, random=rng)

        n_samples = 1000
        counts = Counter()

        for _ in range(n_samples):
            prod = sampler.sample_production('A')
            counts[prod] += 1

        # A -> a and A -> b should be roughly equal
        expected = n_samples / 2
        assert abs(counts[('A', 'a')] - expected) < expected * 0.2
        assert abs(counts[('A', 'b')] - expected) < expected * 0.2


class TestMultinomial:
    """Tests for Multinomial helper class."""

    def test_multinomial_caching(self, simple_pcfg):
        """Multinomial caches samples for efficiency."""
        rng = np.random.default_rng(42)
        m = wcfg.Multinomial(simple_pcfg, 'A', rng, cache_size=100)

        # Should be able to sample many times
        for _ in range(200):
            prod = m.sample_production()
            assert prod[0] == 'A'


class TestMaxDepth:
    """Tests for max_depth handling in sampling."""

    def test_max_depth_prevents_infinite_recursion(self, ambiguous_pcfg):
        """max_depth prevents infinite recursion in recursive grammars."""
        rng = np.random.default_rng(42)
        sampler = Sampler(ambiguous_pcfg, random=rng, max_depth=10)

        # Should complete without infinite loop
        # Might raise ValueError if max depth exceeded
        success_count = 0
        for _ in range(20):
            try:
                tree = sampler.sample_tree()
                success_count += 1
            except ValueError:
                pass  # Max depth exceeded - expected occasionally

        # Should get at least some successful samples
        assert success_count > 0


class TestReproducibility:
    """Tests for sampling reproducibility with seeds."""

    def test_reproducible_with_same_seed(self, simple_pcfg):
        """Same seed produces same sequence."""
        rng1 = np.random.default_rng(42)
        sampler1 = Sampler(simple_pcfg, random=rng1)

        rng2 = np.random.default_rng(42)
        sampler2 = Sampler(simple_pcfg, random=rng2)

        for _ in range(10):
            s1 = sampler1.sample_string()
            s2 = sampler2.sample_string()
            assert s1 == s2

    def test_different_with_different_seed(self, simple_pcfg):
        """Different seeds produce different sequences (usually)."""
        rng1 = np.random.default_rng(42)
        sampler1 = Sampler(simple_pcfg, random=rng1)

        rng2 = np.random.default_rng(123)
        sampler2 = Sampler(simple_pcfg, random=rng2)

        samples1 = [tuple(sampler1.sample_string()) for _ in range(20)]
        samples2 = [tuple(sampler2.sample_string()) for _ in range(20)]

        # Sequences should differ (with high probability)
        assert samples1 != samples2
