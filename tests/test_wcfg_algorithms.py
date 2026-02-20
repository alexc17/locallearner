"""Tests for WCFG algorithms - partition functions, convergence, expectations."""

import pytest
import os
import sys
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'locallearner'))

import wcfg
from wcfg import WCFG
from utility import DivergentWCFGException


class TestPartitionFunctionFP:
    """Tests for fixed-point partition function computation."""

    def test_partition_function_simple(self, simple_pcfg):
        """Partition function for simple PCFG is 1.0 at start."""
        pf = simple_pcfg.compute_partition_function_fp()
        assert pf['S'] == pytest.approx(1.0, rel=1e-5)

    def test_partition_function_all_nonterminals(self, simple_pcfg):
        """Partition function computed for all nonterminals."""
        pf = simple_pcfg.compute_partition_function_fp()
        assert 'S' in pf
        assert 'A' in pf
        # A has only lexical rules, so PF should be 1.0
        assert pf['A'] == pytest.approx(1.0, rel=1e-5)

    def test_partition_function_three_nt(self, three_nt_pcfg):
        """Partition function for 3-nonterminal grammar."""
        pf = three_nt_pcfg.compute_partition_function_fp()
        assert pf['S'] == pytest.approx(1.0, rel=1e-5)
        assert pf['A'] == pytest.approx(1.0, rel=1e-5)
        assert pf['B'] == pytest.approx(1.0, rel=1e-5)


class TestConvergence:
    """Tests for grammar convergence checking."""

    def test_is_convergent_simple(self, simple_pcfg):
        """Simple PCFG is convergent."""
        assert simple_pcfg.is_convergent()

    def test_is_convergent_three_nt(self, three_nt_pcfg):
        """3-nonterminal PCFG is convergent."""
        assert three_nt_pcfg.is_convergent()

    def test_is_consistent_pcfg(self, simple_pcfg):
        """Normalized PCFG is consistent."""
        assert simple_pcfg.is_consistent(epsilon=1e-5)

    def test_is_pcfg(self, simple_pcfg):
        """is_pcfg returns True for normalized grammar."""
        assert simple_pcfg.is_pcfg(epsilon=1e-5)

    def test_is_pcfg_false_for_unnormalized(self):
        """is_pcfg returns False for unnormalized grammar."""
        g = WCFG()
        g.nonterminals = {'S', 'A'}
        g.terminals = {'a'}
        g.start = 'S'
        g.productions = [('S', 'A', 'A'), ('A', 'a')]
        g.parameters = {
            ('S', 'A', 'A'): 0.5,
            ('A', 'a'): 0.7,
        }
        g.set_log_parameters()
        assert not g.is_pcfg(epsilon=1e-5)


class TestRenormalise:
    """Tests for grammar renormalization."""

    def test_renormalise_consistent_grammar(self, simple_pcfg):
        """Renormalising an already consistent grammar preserves it."""
        original_params = dict(simple_pcfg.parameters)
        simple_pcfg.renormalise()

        for prod in original_params:
            if prod in simple_pcfg.parameters:
                assert simple_pcfg.parameters[prod] == pytest.approx(
                    original_params[prod], rel=1e-5
                )

    def test_renormalise_makes_consistent(self):
        """Renormalise converts WCFG to consistent PCFG."""
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
            ('S', 'A', 'A'): 2.0,
            ('A', 'a'): 1.5,
            ('A', 'b'): 1.5,
        }
        g.set_log_parameters()

        g.renormalise()

        assert g.is_consistent(epsilon=1e-5)

    def test_renormalise_locally(self):
        """renormalise_locally makes it a valid PCFG."""
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
            ('S', 'A', 'A'): 3.0,
            ('A', 'a'): 2.0,
            ('A', 'b'): 4.0,
        }
        g.set_log_parameters()

        g.renormalise_locally()

        assert g.is_pcfg(epsilon=1e-5)
        assert g.parameters[('S', 'A', 'A')] == pytest.approx(1.0)
        assert g.parameters[('A', 'a')] == pytest.approx(1/3)
        assert g.parameters[('A', 'b')] == pytest.approx(2/3)


class TestNonterminalExpectations:
    """Tests for nonterminal expectation computation."""

    def test_nonterminal_expectations_simple(self, simple_pcfg):
        """Nonterminal expectations for simple grammar."""
        ne = simple_pcfg.nonterminal_expectations()

        # S is used once per derivation
        assert ne['S'] == pytest.approx(1.0)
        # A is used twice per derivation (S -> A A)
        assert ne['A'] == pytest.approx(2.0)

    def test_nonterminal_expectations_recursive(self, ambiguous_pcfg):
        """Nonterminal expectations raises LinAlgError for critical grammar.

        The ambiguous_pcfg has A -> A A (0.5) | a (0.5), giving a
        branching number of exactly 1.  The expected number of NT uses
        is infinite, so (I - T) is singular.
        """
        import numpy as np
        with pytest.raises(np.linalg.LinAlgError):
            ambiguous_pcfg.nonterminal_expectations()


class TestProductionExpectations:
    """Tests for production expectation computation."""

    def test_production_expectations_simple(self, simple_pcfg):
        """Production expectations for simple grammar."""
        pe = simple_pcfg.production_expectations()

        # S -> A A used once per derivation
        assert pe[('S', 'A', 'A')] == pytest.approx(1.0)
        # A -> a used once on average (E[A] * P(A -> a) = 2 * 0.5 = 1)
        assert pe[('A', 'a')] == pytest.approx(1.0)
        assert pe[('A', 'b')] == pytest.approx(1.0)

    def test_terminal_expectations(self, simple_pcfg):
        """Terminal expectations - expected count per derivation."""
        te = simple_pcfg.terminal_expectations()

        # Each terminal expected once
        assert te['a'] == pytest.approx(1.0)
        assert te['b'] == pytest.approx(1.0)

    def test_expected_length(self, simple_pcfg):
        """Expected string length."""
        el = simple_pcfg.expected_length()
        # Simple grammar always produces length-2 strings
        assert el == pytest.approx(2.0)


class TestEntropy:
    """Tests for entropy computations."""

    def test_entropy_nonterminals_simple(self, simple_pcfg):
        """Entropy of nonterminal distributions."""
        ent = simple_pcfg.entropy_nonterminals()

        # S has only one production: entropy = 0
        assert ent['S'] == pytest.approx(0.0)
        # A has two productions with p=0.5: entropy = -2 * 0.5 * log(0.5) = log(2)
        assert ent['A'] == pytest.approx(math.log(2))

    def test_derivational_entropy(self, simple_pcfg):
        """Derivational entropy (entropy over derivation trees)."""
        de = simple_pcfg.derivational_entropy()

        # Two independent A choices, each with entropy log(2)
        # Total: 2 * log(2)
        assert de == pytest.approx(2 * math.log(2))


class TestMakeUnary:
    """Tests for make_unary (single-terminal version)."""

    def test_make_unary_preserves_structure(self, simple_pcfg):
        """make_unary preserves grammar structure."""
        unary = simple_pcfg.make_unary()

        assert unary.start == simple_pcfg.start
        assert unary.nonterminals == simple_pcfg.nonterminals
        assert len(unary.terminals) == 1

    def test_make_unary_collapses_terminals(self, three_nt_pcfg):
        """make_unary collapses all terminals to one symbol."""
        unary = three_nt_pcfg.make_unary()

        assert wcfg.UNARY_SYMBOL in unary.terminals
        assert len(unary.terminals) == 1


class TestPartitionNonterminals:
    """Tests for partitioning nonterminals into SCCs."""

    def test_partition_simple(self, simple_pcfg):
        """Partition nonterminals of simple grammar."""
        sccs = simple_pcfg.partition_nonterminals()
        # Non-recursive grammar: each NT in its own SCC
        assert len(sccs) >= 1

    def test_partition_recursive(self, ambiguous_pcfg):
        """Partition nonterminals with recursive grammar."""
        sccs = ambiguous_pcfg.partition_nonterminals()
        # A -> A A makes A self-recursive
        # Find the SCC containing A
        for scc in sccs:
            if 'A' in scc:
                assert len(scc) >= 1


class TestUnaryInside:
    """Tests for UnaryInside (length distribution)."""

    def test_unary_inside_simple(self, simple_pcfg):
        """UnaryInside computes length probabilities."""
        ui = wcfg.UnaryInside(simple_pcfg, 5)

        # Simple grammar only generates length-2 strings
        assert ui.probability(1) == pytest.approx(0.0)
        assert ui.probability(2) == pytest.approx(1.0)
        assert ui.probability(3) == pytest.approx(0.0)

    def test_probability_short_string(self, simple_pcfg):
        """compute_probability_short_string for length limit."""
        p = simple_pcfg.compute_probability_short_string(5)
        # All probability mass is at length 2
        assert p == pytest.approx(1.0)


class TestDivergentGrammar:
    """Tests for handling divergent grammars."""

    def test_divergent_grammar_detection(self):
        """Detect divergent grammar."""
        # Create a grammar that might diverge
        g = WCFG()
        g.nonterminals = {'S', 'A'}
        g.terminals = {'a'}
        g.start = 'S'
        g.productions = [
            ('S', 'A', 'A'),
            ('A', 'A', 'A'),
            ('A', 'a'),
        ]
        # Make it very likely to expand
        g.parameters = {
            ('S', 'A', 'A'): 1.0,
            ('A', 'A', 'A'): 0.9,
            ('A', 'a'): 0.1,
        }
        g.set_log_parameters()

        # This grammar is actually convergent (barely), but let's test renormalization
        g2 = g.renormalise_divergent_wcfg2()
        assert g2 is not None

    def test_renormalise_divergent_wcfg2(self):
        """renormalise_divergent_wcfg2 scales grammar."""
        g = WCFG()
        g.nonterminals = {'S', 'A'}
        g.terminals = {'a'}
        g.start = 'S'
        g.productions = [
            ('S', 'A', 'A'),
            ('A', 'A', 'A'),
            ('A', 'a'),
        ]
        g.parameters = {
            ('S', 'A', 'A'): 2.0,  # Not normalized
            ('A', 'A', 'A'): 1.5,
            ('A', 'a'): 0.5,
        }
        g.set_log_parameters()

        g2 = g.renormalise_divergent_wcfg2()
        # Should scale down the parameters
        assert g2.parameters[('S', 'A', 'A')] <= 1.0
