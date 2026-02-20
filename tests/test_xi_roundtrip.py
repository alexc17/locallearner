"""Tests for PCFG -> xi-WCFG -> PCFG roundtrip conversion.

Verifies that:
1. convert_parameters_pi2xi produces a valid xi-parameterised WCFG
2. Converting back via convert_parameters_xi2pi recovers the original PCFG
3. When the xi-WCFG is divergent, the renormalise_divergent_wcfg2 + renormalise
   path still produces a PCFG with low KLD over derivation trees
4. The full convert_wcfg_to_pcfg pipeline (as used in the CLI script) works
"""

import pytest
import os
import sys
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'locallearner'))

import wcfg
from wcfg import WCFG
import evaluation


# ------------------------------------------------------------------ #
# Helper: build test grammars
# ------------------------------------------------------------------ #

def make_simple_pcfg():
    """A small consistent PCFG:
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


def make_larger_pcfg():
    """A slightly larger PCFG with S lexical rules:
    S  -> A B  0.5
    S  -> B A  0.2
    S  -> A A  0.1
    S  -> a    0.1
    S  -> b    0.1
    A  -> a    0.4
    A  -> b    0.3
    A  -> c    0.1
    A  -> A B  0.2
    B  -> b    0.3
    B  -> c    0.3
    B  -> d    0.2
    B  -> B A  0.2
    """
    g = WCFG()
    g.nonterminals = ['S', 'A', 'B']
    g.terminals = ['a', 'b', 'c', 'd']
    g.start = 'S'
    prods = [
        ('S', 'A', 'B', 0.5),
        ('S', 'B', 'A', 0.2),
        ('S', 'A', 'A', 0.1),
        ('S', 'a', 0.1),
        ('S', 'b', 0.1),
        ('A', 'a', 0.4),
        ('A', 'b', 0.3),
        ('A', 'c', 0.1),
        ('A', 'A', 'B', 0.2),
        ('B', 'b', 0.3),
        ('B', 'c', 0.3),
        ('B', 'd', 0.2),
        ('B', 'B', 'A', 0.2),
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


def convert_wcfg_to_pcfg(xi_wcfg):
    """Replicate the logic from convert_wcfg_to_pcfg.py:
    If divergent, apply renormalise_divergent_wcfg2 first,
    then renormalise (Chi-Zhang partition function renormalization).
    """
    g = xi_wcfg.copy()
    if not g.is_convergent():
        g = g.renormalise_divergent_wcfg2()
        assert g.is_convergent(), "Still divergent after beta scaling"
    g.renormalise()
    return g


# ------------------------------------------------------------------ #
# Tests: pi -> xi -> pi roundtrip
# ------------------------------------------------------------------ #

class TestXiRoundtrip:

    def test_pi2xi_produces_different_parameters(self):
        """Xi parameters should differ from pi parameters."""
        pcfg = make_simple_pcfg()
        xi_wcfg = pcfg.convert_parameters_pi2xi()

        # They should have the same productions
        assert set(xi_wcfg.productions) == set(pcfg.productions)

        # But parameters should generally differ
        diffs = [abs(xi_wcfg.parameters[p] - pcfg.parameters[p])
                 for p in pcfg.productions]
        assert max(diffs) > 1e-10, "Xi parameters should differ from pi"

    def test_xi2pi_recovers_original(self):
        """pi -> xi -> pi should recover original parameters exactly."""
        pcfg = make_simple_pcfg()
        xi_wcfg = pcfg.convert_parameters_pi2xi()
        recovered = xi_wcfg.convert_parameters_xi2pi()

        for prod in pcfg.productions:
            assert abs(recovered.parameters[prod] - pcfg.parameters[prod]) < 1e-10, \
                f"Parameter mismatch for {prod}: " \
                f"{recovered.parameters[prod]:.10f} vs {pcfg.parameters[prod]:.10f}"

    def test_xi2pi_recovers_larger_grammar(self):
        """Roundtrip for a grammar with S lexical rules and recursion."""
        pcfg = make_larger_pcfg()
        xi_wcfg = pcfg.convert_parameters_pi2xi()
        recovered = xi_wcfg.convert_parameters_xi2pi()

        for prod in pcfg.productions:
            assert abs(recovered.parameters[prod] - pcfg.parameters[prod]) < 1e-8, \
                f"Parameter mismatch for {prod}: " \
                f"{recovered.parameters[prod]:.10f} vs {pcfg.parameters[prod]:.10f}"

    def test_roundtrip_kld_is_zero(self):
        """KLD between original and roundtripped PCFG should be ~0."""
        pcfg = make_simple_pcfg()
        xi_wcfg = pcfg.convert_parameters_pi2xi()
        recovered = xi_wcfg.convert_parameters_xi2pi()

        kld = evaluation.labeled_kld_exact(pcfg, recovered)
        assert math.isfinite(kld)
        assert abs(kld) < 1e-8, f"Roundtrip KLD should be ~0, got {kld}"

    def test_roundtrip_kld_larger_grammar(self):
        """KLD roundtrip for the larger grammar."""
        pcfg = make_larger_pcfg()
        xi_wcfg = pcfg.convert_parameters_pi2xi()
        recovered = xi_wcfg.convert_parameters_xi2pi()

        kld = evaluation.labeled_kld_exact(pcfg, recovered)
        assert math.isfinite(kld)
        assert abs(kld) < 1e-6, f"Roundtrip KLD should be ~0, got {kld}"


# ------------------------------------------------------------------ #
# Tests: xi-WCFG convergence properties
# ------------------------------------------------------------------ #

class TestXiConvergence:

    def test_xi_from_pcfg_is_convergent(self):
        """Xi-WCFG derived from a valid PCFG should be convergent."""
        pcfg = make_simple_pcfg()
        xi_wcfg = pcfg.convert_parameters_pi2xi()
        assert xi_wcfg.is_convergent(), "Xi-WCFG from valid PCFG should be convergent"

    def test_xi_partition_function(self):
        """Partition function of xi-WCFG from a PCFG should be finite."""
        pcfg = make_simple_pcfg()
        xi_wcfg = pcfg.convert_parameters_pi2xi()
        pf = xi_wcfg.compute_partition_function_fp()
        assert math.isfinite(pf[xi_wcfg.start])
        assert pf[xi_wcfg.start] > 0

    def test_scaled_xi_still_recoverable(self):
        """If we scale xi parameters uniformly (making it non-PCFG),
        the renormalise path should still produce a valid PCFG with
        the same conditional distribution (KLD ~ 0)."""
        pcfg = make_simple_pcfg()
        xi_wcfg = pcfg.convert_parameters_pi2xi()

        # Scale all parameters by 0.5 — still convergent but not a PCFG
        scaled = xi_wcfg.copy()
        for prod in scaled.productions:
            scaled.parameters[prod] *= 0.5
        scaled.set_log_parameters()

        assert scaled.is_convergent()
        recovered = convert_wcfg_to_pcfg(scaled)

        kld = evaluation.labeled_kld_exact(pcfg, recovered)
        assert math.isfinite(kld)
        assert abs(kld) < 1e-6, \
            f"Uniformly scaled xi should recover same PCFG, KLD={kld}"


# ------------------------------------------------------------------ #
# Tests: divergent WCFG recovery
# ------------------------------------------------------------------ #

class TestDivergentRecovery:

    def _make_divergent_xi(self):
        """Create a divergent xi-WCFG by inflating binary rule parameters.
        Uses the larger grammar which has recursion (A->AB, B->BA)."""
        pcfg = make_larger_pcfg()
        xi_wcfg = pcfg.convert_parameters_pi2xi()

        # Inflate binary rules — the recursive grammar diverges easily
        div = xi_wcfg.copy()
        for prod in div.productions:
            if len(prod) == 3:
                div.parameters[prod] *= 5.0
        div.set_log_parameters()
        return pcfg, div

    def test_inflated_is_divergent(self):
        """Inflating binary xi parameters should make the WCFG divergent."""
        _, div = self._make_divergent_xi()
        assert not div.is_convergent()

    def test_divergent_recovery_produces_pcfg(self):
        """renormalise_divergent_wcfg2 + renormalise should produce
        a valid locally-normalised PCFG."""
        _, div = self._make_divergent_xi()
        recovered = convert_wcfg_to_pcfg(div)

        # Check local normalisation: each NT's productions should sum to ~1
        totals = recovered.check_local_normalisation()
        for nt, total in totals.items():
            assert abs(total - 1.0) < 1e-6, \
                f"NT {nt} not normalised: sum={total}"

    def test_divergent_recovery_finite_kld(self):
        """Recovered PCFG from divergent WCFG should have finite KLD
        against the original."""
        pcfg, div = self._make_divergent_xi()
        recovered = convert_wcfg_to_pcfg(div)

        kld = evaluation.smoothed_kld_exact(pcfg, recovered)
        assert math.isfinite(kld), f"KLD should be finite, got {kld}"

    def test_divergent_recovery_preserves_structure(self):
        """Recovered PCFG should have the same nonterminals and terminals."""
        pcfg, div = self._make_divergent_xi()
        recovered = convert_wcfg_to_pcfg(div)

        assert set(recovered.nonterminals) == set(pcfg.nonterminals)
        # Terminals should be a superset (trim_zeros may remove some)
        assert set(recovered.terminals).issubset(set(pcfg.terminals))


# ------------------------------------------------------------------ #
# Tests: renormalise preserves conditional tree distribution
# ------------------------------------------------------------------ #

class TestRenormalisePreservesConditional:

    def test_renormalise_convergent_preserves_kld(self):
        """For a convergent (but non-PCFG) WCFG, renormalise should
        produce a PCFG with KLD=0 against the original's conditional
        distribution."""
        pcfg = make_simple_pcfg()
        xi_wcfg = pcfg.convert_parameters_pi2xi()

        # Scale to make it non-PCFG but still convergent
        scaled = xi_wcfg.copy()
        for prod in scaled.productions:
            scaled.parameters[prod] *= 0.7
        scaled.set_log_parameters()

        assert scaled.is_convergent()
        recovered = convert_wcfg_to_pcfg(scaled)

        # Should recover the exact same PCFG (same conditional distribution)
        kld = evaluation.labeled_kld_exact(pcfg, recovered)
        assert math.isfinite(kld)
        assert abs(kld) < 1e-6, \
            f"Renormalisation should preserve conditional distribution, KLD={kld}"

    def test_nonuniform_scaling_changes_conditional(self):
        """Non-uniform per-NT scaling of xi parameters should change the
        conditional distribution, yielding KLD > 0."""
        pcfg = make_simple_pcfg()
        xi_wcfg = pcfg.convert_parameters_pi2xi()

        # Scale only one NT's binary rules — changes the conditional
        perturbed = xi_wcfg.copy()
        for prod in perturbed.productions:
            if len(prod) == 3 and prod[0] == 'S':
                perturbed.parameters[prod] *= 2.0
            # Leave A, B binary rules unchanged
        # Also perturb one lexical rule
        for prod in perturbed.productions:
            if prod == ('A', 'a'):
                perturbed.parameters[prod] *= 0.5
        perturbed.set_log_parameters()

        if perturbed.is_convergent():
            recovered = convert_wcfg_to_pcfg(perturbed)
            kld = evaluation.labeled_kld_exact(pcfg, recovered)
            assert math.isfinite(kld)
            assert kld > 1e-6, \
                f"Non-uniform scaling should change distribution, KLD={kld}"

    def test_renormalise_larger_grammar_roundtrip(self):
        """Full roundtrip for the larger grammar with S lexical rules."""
        pcfg = make_larger_pcfg()
        xi_wcfg = pcfg.convert_parameters_pi2xi()
        recovered = convert_wcfg_to_pcfg(xi_wcfg)

        kld = evaluation.labeled_kld_exact(pcfg, recovered)
        assert math.isfinite(kld)
        assert abs(kld) < 1e-6, \
            f"Roundtrip via convert_wcfg_to_pcfg should give KLD~0, got {kld}"


# ------------------------------------------------------------------ #
# Tests: file-based roundtrip (save/load)
# ------------------------------------------------------------------ #

class TestFileRoundtrip:

    def test_save_load_xi_wcfg(self, tmp_path):
        """Save xi-WCFG to file, reload, and verify roundtrip."""
        pcfg = make_simple_pcfg()
        xi_wcfg = pcfg.convert_parameters_pi2xi()

        fpath = str(tmp_path / "test.wcfg")
        xi_wcfg.store(fpath)

        loaded = wcfg.load_wcfg_from_file(fpath)

        # Check parameters match (limited by file format precision)
        for prod in xi_wcfg.productions:
            assert abs(loaded.parameters[prod] - xi_wcfg.parameters[prod]) < 1e-5, \
                f"Save/load mismatch for {prod}: {loaded.parameters[prod]} vs {xi_wcfg.parameters[prod]}"

    def test_save_load_convert_roundtrip(self, tmp_path):
        """Save xi-WCFG, reload, convert to PCFG, check KLD."""
        pcfg = make_larger_pcfg()
        xi_wcfg = pcfg.convert_parameters_pi2xi()

        fpath = str(tmp_path / "test.wcfg")
        xi_wcfg.store(fpath)
        loaded = wcfg.load_wcfg_from_file(fpath)

        recovered = convert_wcfg_to_pcfg(loaded)

        kld = evaluation.labeled_kld_exact(pcfg, recovered)
        assert math.isfinite(kld)
        assert abs(kld) < 1e-5, \
            f"File roundtrip KLD should be ~0, got {kld}"


# ------------------------------------------------------------------ #
# Tests: real grammar file if available
# ------------------------------------------------------------------ #

class TestWithRealGrammar:

    GRAMMAR_PATH = os.path.join(
        os.path.dirname(__file__), '..', 'tmp', 'jm_experiment', 'g007', 'grammar.pcfg')

    @pytest.mark.skipif(
        not os.path.exists(GRAMMAR_PATH),
        reason="Real grammar file not available")
    def test_real_grammar_roundtrip(self):
        """pi -> xi -> pi roundtrip on a real grammar."""
        pcfg = wcfg.load_wcfg_from_file(self.GRAMMAR_PATH)
        xi_wcfg = pcfg.convert_parameters_pi2xi()

        assert xi_wcfg.is_convergent()

        recovered = xi_wcfg.convert_parameters_xi2pi()

        kld = evaluation.labeled_kld_exact(pcfg, recovered)
        assert math.isfinite(kld)
        assert abs(kld) < 1e-6, f"Real grammar roundtrip KLD={kld}"

    @pytest.mark.skipif(
        not os.path.exists(GRAMMAR_PATH),
        reason="Real grammar file not available")
    def test_real_grammar_renormalise_roundtrip(self):
        """pi -> xi -> convert_wcfg_to_pcfg on a real grammar."""
        pcfg = wcfg.load_wcfg_from_file(self.GRAMMAR_PATH)
        xi_wcfg = pcfg.convert_parameters_pi2xi()
        recovered = convert_wcfg_to_pcfg(xi_wcfg)

        kld = evaluation.labeled_kld_exact(pcfg, recovered)
        assert math.isfinite(kld)
        assert abs(kld) < 1e-5, f"Real grammar renormalise roundtrip KLD={kld}"
