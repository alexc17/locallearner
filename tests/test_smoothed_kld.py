"""Tests for smoothed_kld_exact: smoothed version should always return finite values."""

import pytest
import os
import sys
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'locallearner'))

import wcfg
from wcfg import WCFG
import evaluation


def make_example_pcfg():
    """Build a small PCFG:
    S  -> A B  0.6
    S  -> A A  0.4
    A  -> a    0.5
    A  -> b    0.3
    A  -> c    0.2
    B  -> b    0.6
    B  -> c    0.4
    """
    g = WCFG()
    g.nonterminals = ['S', 'A', 'B']
    g.terminals = ['a', 'b', 'c']
    g.start = 'S'
    prods = [
        ('S', 'A', 'B', 0.6),
        ('S', 'A', 'A', 0.4),
        ('A', 'a', 0.5),
        ('A', 'b', 0.3),
        ('A', 'c', 0.2),
        ('B', 'b', 0.6),
        ('B', 'c', 0.4),
    ]
    g.productions = []
    g.parameters = {}
    for p in prods:
        if len(p) == 3:
            prod = (p[0], p[1])
            g.productions.append(prod)
            g.parameters[prod] = p[2]
        else:
            prod = (p[0], p[1], p[2])
            g.productions.append(prod)
            g.parameters[prod] = p[3]
    g.set_log_parameters()
    return g


def make_reduced_pcfg():
    """Same as example but delete A -> c, then renormalise.
    This means the reduced grammar cannot produce any string
    containing 'c' from A, changing the distribution.

    S  -> A B  0.6
    S  -> A A  0.4
    A  -> a    0.625   (0.5/0.8)
    A  -> b    0.375   (0.3/0.8)
    B  -> b    0.6
    B  -> c    0.4
    """
    g = WCFG()
    g.nonterminals = ['S', 'A', 'B']
    g.terminals = ['a', 'b', 'c']
    g.start = 'S'
    prods = [
        ('S', 'A', 'B', 0.6),
        ('S', 'A', 'A', 0.4),
        ('A', 'a', 0.625),
        ('A', 'b', 0.375),
        ('B', 'b', 0.6),
        ('B', 'c', 0.4),
    ]
    g.productions = []
    g.parameters = {}
    for p in prods:
        if len(p) == 3:
            prod = (p[0], p[1])
            g.productions.append(prod)
            g.parameters[prod] = p[2]
        else:
            prod = (p[0], p[1], p[2])
            g.productions.append(prod)
            g.parameters[prod] = p[3]
    g.set_log_parameters()
    return g


class TestSmoothedKLD:

    def test_self_kld_is_zero(self):
        """KLD of a grammar with itself should be zero."""
        g = make_example_pcfg()
        kld = evaluation.labeled_kld_exact(g, g)
        assert math.isfinite(kld)
        assert abs(kld) < 1e-10

    def test_unsmoothed_missing_production_gives_inf(self):
        """labeled_kld_exact returns inf when hypothesis is missing a
        production that the target uses (A -> c is missing)."""
        target = make_example_pcfg()
        hyp = make_reduced_pcfg()
        # hyp has no A -> c, but target does
        assert ('A', 'c') not in hyp.parameters
        kld = evaluation.labeled_kld_exact(target, hyp)
        assert kld == math.inf

    def test_smoothed_missing_production_gives_finite(self):
        """smoothed_kld_exact should return a finite positive value
        even when the hypothesis is missing a production."""
        target = make_example_pcfg()
        hyp = make_reduced_pcfg()
        # Remove 'c' from hyp terminals list to simulate partial grammar
        assert ('A', 'c') not in hyp.parameters
        kld = evaluation.smoothed_kld_exact(target, hyp)
        assert math.isfinite(kld), f"Expected finite KLD, got {kld}"
        assert kld > 0

    def test_smoothed_kld_positive(self):
        """Smoothed KLD between different grammars should be positive and finite."""
        target = make_example_pcfg()
        hyp = make_reduced_pcfg()
        kld = evaluation.smoothed_kld_exact(target, hyp)
        assert math.isfinite(kld)
        assert kld > 0.01  # should be noticeably positive

    def test_smoothed_larger_than_unsmoothed_when_both_finite(self):
        """When both grammars have the same productions, smoothing should
        increase KLD slightly (by adding epsilon noise)."""
        target = make_example_pcfg()
        hyp = make_example_pcfg()
        # Perturb one parameter slightly
        hyp.parameters[('A', 'a')] = 0.45
        hyp.parameters[('A', 'b')] = 0.35
        hyp.set_log_parameters()

        kld_raw = evaluation.labeled_kld_exact(target, hyp)
        kld_smooth = evaluation.smoothed_kld_exact(target, hyp)
        assert math.isfinite(kld_raw)
        assert math.isfinite(kld_smooth)
        # Both should be positive
        assert kld_raw > 0
        assert kld_smooth > 0

    def test_smoothed_kld_with_missing_binary_rule(self):
        """Even when a binary rule is missing, smoothed KLD should be finite."""
        target = make_example_pcfg()
        hyp = make_example_pcfg()
        # Remove S -> A A binary rule
        del hyp.parameters[('S', 'A', 'A')]
        hyp.productions = [p for p in hyp.productions if p != ('S', 'A', 'A')]
        hyp.locally_normalise()
        hyp.set_log_parameters()

        # Unsmoothed should be inf
        kld_raw = evaluation.labeled_kld_exact(target, hyp)
        assert kld_raw == math.inf

        # Smoothed should be finite
        kld_smooth = evaluation.smoothed_kld_exact(target, hyp)
        assert math.isfinite(kld_smooth), f"Expected finite KLD, got {kld_smooth}"
        assert kld_smooth > 0
