"""Tests for WCFG basic operations - loading, storing, normalization."""

import pytest
import os
import sys
import math
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'locallearner'))

import wcfg
from wcfg import WCFG, load_wcfg_from_file


class TestWCFGLoading:
    """Tests for loading WCFG from files."""

    def test_load_simple_grammar(self, simple_grammar_path):
        """Load simple grammar from file."""
        g = load_wcfg_from_file(simple_grammar_path)
        assert g is not None
        assert g.start == 'S'
        assert 'S' in g.nonterminals
        assert 'A' in g.nonterminals
        assert 'a' in g.terminals
        assert 'b' in g.terminals

    def test_load_ambiguous_grammar(self, ambiguous_grammar_path):
        """Load ambiguous grammar from file."""
        g = load_wcfg_from_file(ambiguous_grammar_path)
        assert g is not None
        assert g.start == 'S'
        assert ('A', 'A', 'A') in g.productions

    def test_loaded_grammar_has_parameters(self, simple_grammar):
        """Loaded grammar has correct parameters."""
        assert simple_grammar.parameters[('S', 'A', 'A')] == 1.0
        assert simple_grammar.parameters[('A', 'a')] == 0.5
        assert simple_grammar.parameters[('A', 'b')] == 0.5

    def test_loaded_grammar_has_log_parameters(self, simple_grammar):
        """Loaded grammar has log parameters set."""
        assert ('S', 'A', 'A') in simple_grammar.log_parameters
        assert simple_grammar.log_parameters[('A', 'a')] == pytest.approx(math.log(0.5))


class TestWCFGStoring:
    """Tests for storing WCFG to files."""

    def test_store_and_reload(self, simple_pcfg):
        """Store grammar and reload - round trip test."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pcfg', delete=False) as f:
            temp_path = f.name

        try:
            simple_pcfg.store(temp_path)
            reloaded = load_wcfg_from_file(temp_path)

            # Check structure is preserved
            assert reloaded.start == simple_pcfg.start
            assert reloaded.nonterminals == simple_pcfg.nonterminals
            assert reloaded.terminals == simple_pcfg.terminals

            # Check productions are preserved
            for prod in simple_pcfg.productions:
                assert prod in reloaded.productions
                assert reloaded.parameters[prod] == pytest.approx(
                    simple_pcfg.parameters[prod], rel=1e-5
                )
        finally:
            os.unlink(temp_path)

    def test_store_with_header(self, simple_pcfg):
        """Store grammar with header comments."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pcfg', delete=False) as f:
            temp_path = f.name

        try:
            simple_pcfg.store(temp_path, header=['Test header', 'Line 2'])
            with open(temp_path) as f:
                content = f.read()
            assert '#Test header' in content
            assert '#Line 2' in content
        finally:
            os.unlink(temp_path)


class TestWCFGNormalization:
    """Tests for WCFG normalization."""

    def test_is_normalised_true(self, simple_pcfg):
        """simple_pcfg should already be normalized."""
        assert simple_pcfg.is_normalised()

    def test_is_normalised_false(self):
        """Unnormalized grammar returns False."""
        g = WCFG()
        g.nonterminals = {'S', 'A'}
        g.terminals = {'a'}
        g.start = 'S'
        g.productions = [('S', 'A', 'A'), ('A', 'a')]
        g.parameters = {
            ('S', 'A', 'A'): 0.5,  # Not normalized
            ('A', 'a'): 0.7,  # Not normalized
        }
        g.set_log_parameters()
        assert not g.is_normalised()

    def test_locally_normalise(self):
        """locally_normalise converts WCFG to PCFG."""
        g = WCFG()
        g.nonterminals = {'S', 'A'}
        g.terminals = {'a', 'b'}
        g.start = 'S'
        g.productions = [('S', 'A', 'A'), ('A', 'a'), ('A', 'b')]
        g.parameters = {
            ('S', 'A', 'A'): 2.0,
            ('A', 'a'): 3.0,
            ('A', 'b'): 2.0,
        }
        g.set_log_parameters()

        g.locally_normalise()

        assert g.is_normalised()
        assert g.parameters[('S', 'A', 'A')] == pytest.approx(1.0)
        assert g.parameters[('A', 'a')] == pytest.approx(0.6)
        assert g.parameters[('A', 'b')] == pytest.approx(0.4)

    def test_check_normalisation(self, simple_pcfg):
        """check_normalisation returns totals per nonterminal."""
        totals = simple_pcfg.check_normalisation()
        assert totals['S'] == pytest.approx(1.0)
        assert totals['A'] == pytest.approx(1.0)


class TestWCFGCopy:
    """Tests for WCFG copy method."""

    def test_copy_creates_new_object(self, simple_pcfg):
        """Copy creates a distinct object."""
        copy = simple_pcfg.copy()
        assert copy is not simple_pcfg
        assert copy.parameters is not simple_pcfg.parameters
        assert copy.productions is not simple_pcfg.productions

    def test_copy_preserves_data(self, simple_pcfg):
        """Copy preserves all data."""
        copy = simple_pcfg.copy()
        assert copy.start == simple_pcfg.start
        assert copy.nonterminals == simple_pcfg.nonterminals
        assert copy.terminals == simple_pcfg.terminals
        assert copy.productions == simple_pcfg.productions
        assert copy.parameters == simple_pcfg.parameters

    def test_copy_is_independent(self, simple_pcfg):
        """Modifying copy doesn't affect original."""
        copy = simple_pcfg.copy()
        copy.parameters[('A', 'a')] = 0.99
        assert simple_pcfg.parameters[('A', 'a')] == 0.5


class TestWCFGRelabel:
    """Tests for relabel method."""

    def test_relabel_nonterminals(self, simple_pcfg):
        """Relabel nonterminals with a mapping."""
        mapping = {'S': 'X', 'A': 'Y'}
        relabeled = simple_pcfg.relabel(mapping)

        assert 'X' in relabeled.nonterminals
        assert 'Y' in relabeled.nonterminals
        assert relabeled.start == 'X'
        assert ('X', 'Y', 'Y') in relabeled.productions
        assert ('Y', 'a') in relabeled.productions

    def test_relabel_preserves_terminals(self, simple_pcfg):
        """Relabeling doesn't change terminals."""
        mapping = {'S': 'X', 'A': 'Y'}
        relabeled = simple_pcfg.relabel(mapping)
        assert relabeled.terminals == simple_pcfg.terminals

    def test_relabel_preserves_parameters(self, simple_pcfg):
        """Relabeling preserves parameter values."""
        mapping = {'S': 'X', 'A': 'Y'}
        relabeled = simple_pcfg.relabel(mapping)
        assert relabeled.parameters[('Y', 'a')] == simple_pcfg.parameters[('A', 'a')]


class TestWCFGTrimZeros:
    """Tests for trim_zeros method."""

    def test_trim_zeros_removes_zero_productions(self):
        """trim_zeros removes productions with zero weight."""
        g = WCFG()
        g.nonterminals = {'S', 'A', 'B'}
        g.terminals = {'a', 'b'}
        g.start = 'S'
        g.productions = [
            ('S', 'A', 'A'),
            ('S', 'B', 'B'),
            ('A', 'a'),
            ('B', 'b'),
        ]
        g.parameters = {
            ('S', 'A', 'A'): 1.0,
            ('S', 'B', 'B'): 0.0,  # Zero weight
            ('A', 'a'): 1.0,
            ('B', 'b'): 0.0,  # Zero weight
        }

        g.trim_zeros()

        assert ('S', 'B', 'B') not in g.productions
        assert ('B', 'b') not in g.productions
        assert 'B' not in g.nonterminals
        assert 'b' not in g.terminals

    def test_trim_zeros_with_threshold(self):
        """trim_zeros with custom threshold."""
        g = WCFG()
        g.nonterminals = {'S', 'A'}
        g.terminals = {'a', 'b'}
        g.start = 'S'
        g.productions = [('S', 'A', 'A'), ('A', 'a'), ('A', 'b')]
        g.parameters = {
            ('S', 'A', 'A'): 1.0,
            ('A', 'a'): 0.1,
            ('A', 'b'): 0.01,  # Below threshold
        }

        g.trim_zeros(threshold=0.05)

        assert ('A', 'b') not in g.productions


class TestWCFGUnkify:
    """Tests for unkify method."""

    def test_unkify_creates_unk_rules(self, three_nt_pcfg):
        """unkify groups rare terminals under UNK."""
        frequent = {'a', 'b'}
        unk_grammar = three_nt_pcfg.unkify(frequent, '<UNK>')

        assert '<UNK>' in unk_grammar.terminals
        assert 'a' in unk_grammar.terminals
        assert 'b' in unk_grammar.terminals
        assert 'c' not in unk_grammar.terminals

    def test_unkify_preserves_frequent(self, three_nt_pcfg):
        """unkify preserves frequent terminals."""
        frequent = {'a', 'b'}
        unk_grammar = three_nt_pcfg.unkify(frequent, '<UNK>')

        # Should still have rules for frequent terminals
        assert ('A', 'a') in unk_grammar.productions
        assert ('B', 'b') in unk_grammar.productions


class TestWCFGCounts:
    """Tests for counting methods."""

    def test_count_lexical(self, simple_pcfg):
        """Count lexical productions."""
        assert simple_pcfg.count_lexical() == 2  # A -> a, A -> b

    def test_count_binary(self, simple_pcfg):
        """Count binary productions."""
        assert simple_pcfg.count_binary() == 1  # S -> A A

    def test_count_three_nt(self, three_nt_pcfg):
        """Counts for three-nonterminal grammar."""
        assert three_nt_pcfg.count_lexical() == 4
        assert three_nt_pcfg.count_binary() == 2
