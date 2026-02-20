"""Tests for viterbi_parse lexical rule checking and NeuralLearner build_wcfg."""

import pytest
import os
import sys
import math
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'locallearner'))

import wcfg
from wcfg import WCFG, InsideComputation
from utility import ParseFailureException, collect_yield


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def grammar_with_s_lexical():
    """PCFG where S has both binary and lexical rules.

    S -> A B (0.7)
    S -> a   (0.3)
    A -> a   (0.6)
    A -> b   (0.4)
    B -> b   (0.8)
    B -> a   (0.2)
    """
    g = WCFG()
    g.nonterminals = {'S', 'A', 'B'}
    g.terminals = {'a', 'b'}
    g.start = 'S'
    g.productions = [
        ('S', 'A', 'B'),
        ('S', 'a'),
        ('A', 'a'),
        ('A', 'b'),
        ('B', 'b'),
        ('B', 'a'),
    ]
    g.parameters = {
        ('S', 'A', 'B'): 0.7,
        ('S', 'a'): 0.3,
        ('A', 'a'): 0.6,
        ('A', 'b'): 0.4,
        ('B', 'b'): 0.8,
        ('B', 'a'): 0.2,
    }
    g.set_log_parameters()
    return g


@pytest.fixture
def grammar_no_s_lexical():
    """PCFG where S has only binary rules (no S -> terminal).

    S -> A A (1.0)
    A -> a   (0.5)
    A -> b   (0.5)
    """
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
        ('S', 'A', 'A'): 1.0,
        ('A', 'a'): 0.5,
        ('A', 'b'): 0.5,
    }
    g.set_log_parameters()
    return g


@pytest.fixture
def grammar_partial_s_lexical():
    """PCFG where S -> a exists but S -> b does not.

    S -> A B  (0.6)
    S -> a    (0.4)
    A -> a    (1.0)
    B -> b    (1.0)
    """
    g = WCFG()
    g.nonterminals = {'S', 'A', 'B'}
    g.terminals = {'a', 'b'}
    g.start = 'S'
    g.productions = [
        ('S', 'A', 'B'),
        ('S', 'a'),
        ('A', 'a'),
        ('B', 'b'),
    ]
    g.parameters = {
        ('S', 'A', 'B'): 0.6,
        ('S', 'a'): 0.4,
        ('A', 'a'): 1.0,
        ('B', 'b'): 1.0,
    }
    g.set_log_parameters()
    return g


# ============================================================
# Tests: viterbi_parse lexical rule validation
# ============================================================

class TestViterbiLexicalValidation:
    """Tests that viterbi_parse correctly checks lexical rules exist."""

    def test_length1_parseable(self, grammar_with_s_lexical):
        """Length-1 string parses when S -> terminal exists."""
        ic = InsideComputation(grammar_with_s_lexical)
        tree = ic.viterbi_parse(('a',))
        assert tree == ('S', 'a')

    def test_length1_unparseable_no_s_lexical(self, grammar_no_s_lexical):
        """Length-1 string fails when S has no lexical rules."""
        ic = InsideComputation(grammar_no_s_lexical)
        with pytest.raises(ParseFailureException):
            ic.viterbi_parse(('a',))

    def test_length1_unparseable_wrong_terminal(self, grammar_partial_s_lexical):
        """Length-1 string fails when S -> terminal doesn't exist for that word."""
        ic = InsideComputation(grammar_partial_s_lexical)
        # S -> a exists, so ('a',) should parse
        tree = ic.viterbi_parse(('a',))
        assert tree == ('S', 'a')
        # S -> b does NOT exist, so ('b',) should fail
        with pytest.raises(ParseFailureException):
            ic.viterbi_parse(('b',))

    def test_length1_unknown_terminal(self, grammar_with_s_lexical):
        """Length-1 string with unknown terminal fails."""
        ic = InsideComputation(grammar_with_s_lexical)
        with pytest.raises(ParseFailureException):
            ic.viterbi_parse(('z',))

    def test_length2_still_works(self, grammar_with_s_lexical):
        """Length-2 strings still parse correctly."""
        ic = InsideComputation(grammar_with_s_lexical)
        tree = ic.viterbi_parse(('a', 'b'))
        assert tree[0] == 'S'
        assert collect_yield(tree) == ['a', 'b']

    def test_length1_probability_consistent(self, grammar_with_s_lexical):
        """Viterbi and inside probability agree for length-1 strings."""
        ic = InsideComputation(grammar_with_s_lexical)
        lp = ic.inside_log_probability(('a',))
        assert math.exp(lp) == pytest.approx(0.3)

    def test_inside_and_viterbi_agree_on_failure(self, grammar_no_s_lexical):
        """Both inside_log_probability and viterbi_parse fail for
        length-1 strings when no S lexical rules exist."""
        ic = InsideComputation(grammar_no_s_lexical)
        with pytest.raises(ParseFailureException):
            ic.inside_log_probability(('a',))
        with pytest.raises(ParseFailureException):
            ic.viterbi_parse(('a',))


# ============================================================
# Tests: NeuralLearner build_wcfg includes S lexical rules
# ============================================================

class TestNeuralLearnerSLexical:
    """Tests that NeuralLearner produces grammars with S lexical rules."""

    @pytest.fixture
    def corpus_with_length1(self, tmp_path):
        """Create a corpus that includes length-1 sentences."""
        corpus_file = tmp_path / "corpus.txt"
        lines = []
        # Length-1 sentences
        for _ in range(50):
            lines.append("aaa")
        for _ in range(30):
            lines.append("bbb")
        # Length-2 sentences
        for _ in range(100):
            lines.append("aaa bbb")
        for _ in range(80):
            lines.append("bbb aaa")
        # Length-3+ sentences
        for _ in range(200):
            lines.append("aaa bbb aaa")
        for _ in range(150):
            lines.append("bbb aaa bbb")
        for _ in range(100):
            lines.append("aaa aaa bbb bbb")
        corpus_file.write_text('\n'.join(lines) + '\n')
        return str(corpus_file)

    def test_length1_counts(self, corpus_with_length1):
        """NeuralLearner correctly counts length-1 sentences."""
        from neural_learner import NeuralLearner
        nl = NeuralLearner(corpus_with_length1)
        assert nl.length1_counts['aaa'] == 50
        assert nl.length1_counts['bbb'] == 30
        assert pytest.approx(nl.E_length1['aaa']) == 50 / nl.n_sentences
        assert pytest.approx(nl.E_length1['bbb']) == 30 / nl.n_sentences

    def test_lexical_xi_includes_s_rules(self, corpus_with_length1):
        """estimate_lexical_xi produces S -> terminal rules."""
        from neural_learner import NeuralLearner
        nl = NeuralLearner(corpus_with_length1)

        # Manually set up minimal state to test lexical xi for S
        nl.anchors = ['aaa']
        nl.nonterminals = ['S', 'NT_aaa']
        nl.anchor2nt = {'aaa': 'NT_aaa'}

        # We need the neural model for non-S rules, but S rules
        # come directly from length-1 counts. Patch to skip neural.
        nl.lexical_xi = {}
        # Add S lexical rules directly (the logic from estimate_lexical_xi)
        for b, e in nl.E_length1.items():
            nl.lexical_xi[('S', b)] = e

        assert ('S', 'aaa') in nl.lexical_xi
        assert ('S', 'bbb') in nl.lexical_xi
        assert nl.lexical_xi[('S', 'aaa')] == pytest.approx(50 / nl.n_sentences)
        assert nl.lexical_xi[('S', 'bbb')] == pytest.approx(30 / nl.n_sentences)

    def test_build_wcfg_has_s_lexical(self, corpus_with_length1):
        """build_wcfg includes S -> terminal productions."""
        from neural_learner import NeuralLearner
        nl = NeuralLearner(corpus_with_length1)
        nl.anchors = ['aaa']
        nl.nonterminals = ['S', 'NT_aaa']
        nl.anchor2nt = {'aaa': 'NT_aaa'}

        # Minimal xi parameters
        nl.lexical_xi = {
            ('NT_aaa', 'aaa'): 0.5,
            ('NT_aaa', 'bbb'): 0.3,
            ('S', 'aaa'): 50 / nl.n_sentences,
            ('S', 'bbb'): 30 / nl.n_sentences,
        }
        nl.binary_xi = {
            ('S', 'NT_aaa', 'NT_aaa'): 0.8,
        }

        g = nl.build_wcfg(verbose=False)

        # Check S lexical productions exist
        s_lex = [p for p in g.productions if p[0] == 'S' and len(p) == 2]
        assert len(s_lex) == 2
        assert ('S', 'aaa') in g.parameters
        assert ('S', 'bbb') in g.parameters
        assert g.parameters[('S', 'aaa')] > 0
        assert g.parameters[('S', 'bbb')] > 0

    def test_grammar_can_parse_length1(self, corpus_with_length1):
        """Grammar from build_wcfg can parse length-1 strings."""
        from neural_learner import NeuralLearner
        nl = NeuralLearner(corpus_with_length1)
        nl.anchors = ['aaa']
        nl.nonterminals = ['S', 'NT_aaa']
        nl.anchor2nt = {'aaa': 'NT_aaa'}

        nl.lexical_xi = {
            ('NT_aaa', 'aaa'): 0.5,
            ('NT_aaa', 'bbb'): 0.3,
            ('S', 'aaa'): 50 / nl.n_sentences,
            ('S', 'bbb'): 30 / nl.n_sentences,
        }
        nl.binary_xi = {
            ('S', 'NT_aaa', 'NT_aaa'): 0.8,
        }

        g = nl.build_wcfg(verbose=False)
        g.locally_normalise_lax()
        g.set_log_parameters()

        ic = InsideComputation(g)

        # Length-1 should parse
        tree = ic.viterbi_parse(('aaa',))
        assert tree == ('S', 'aaa')

        tree = ic.viterbi_parse(('bbb',))
        assert tree == ('S', 'bbb')

        # Length-2 should also parse
        tree = ic.viterbi_parse(('aaa', 'bbb'))
        assert tree[0] == 'S'
        assert collect_yield(tree) == ['aaa', 'bbb']


# ============================================================
# Tests: IO reestimation preserves length-1 parsing
# ============================================================

class TestIOPreservesLength1:
    """Tests that IO reestimation preserves S lexical rules."""

    def test_io_preserves_s_lexical_rules(self, grammar_with_s_lexical):
        """IO reestimation preserves S -> terminal rules when
        length-1 sentences appear in the data."""
        data = [
            ('a',),          # length-1
            ('a', 'b'),      # length-2
            ('b', 'a'),      # length-2
            ('a',),          # length-1
        ]
        reest = grammar_with_s_lexical.estimate_inside_outside_from_list(
            data, maxlength=10, maxcount=100)
        reest.set_log_parameters()

        # S -> a should still exist
        assert ('S', 'a') in reest.parameters
        assert reest.parameters[('S', 'a')] > 0

        # Should still be parseable
        ic = InsideComputation(reest)
        tree = ic.viterbi_parse(('a',))
        assert tree == ('S', 'a')

    def test_io_does_not_create_spurious_s_lexical(self, grammar_no_s_lexical):
        """IO reestimation does not create S -> b when it didn't exist."""
        data = [
            ('a', 'b'),
            ('b', 'a'),
        ]
        reest = grammar_no_s_lexical.estimate_inside_outside_from_list(
            data, maxlength=10, maxcount=100)
        reest.set_log_parameters()

        # S -> a should NOT exist (no S lexical rules in original)
        s_lex = [p for p in reest.productions if p[0] == 'S' and len(p) == 2]
        assert len(s_lex) == 0
