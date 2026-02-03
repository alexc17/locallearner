"""Tests for WCFG parsing - inside-outside, CKY, Viterbi."""

import pytest
import os
import sys
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'locallearner'))

import wcfg
from wcfg import WCFG, InsideComputation
from utility import ParseFailureException, collect_yield


class TestInsideComputation:
    """Tests for InsideComputation class."""

    def test_inside_computation_init(self, simple_pcfg):
        """InsideComputation initializes correctly."""
        ic = InsideComputation(simple_pcfg)
        assert ic.start == simple_pcfg.start
        assert 'a' in ic.terminals
        assert 'b' in ic.terminals

    def test_inside_log_probability_simple(self, simple_pcfg):
        """Compute inside log probability for simple sentences."""
        ic = InsideComputation(simple_pcfg)

        # P("aa") = P(S -> A A) * P(A -> a) * P(A -> a) = 1.0 * 0.5 * 0.5 = 0.25
        lp = ic.inside_log_probability(('a', 'a'))
        assert lp == pytest.approx(math.log(0.25))

    def test_inside_log_probability_all_strings(self, simple_pcfg):
        """All strings of length 2 have equal probability."""
        ic = InsideComputation(simple_pcfg)

        strings = [('a', 'a'), ('a', 'b'), ('b', 'a'), ('b', 'b')]
        probs = [math.exp(ic.inside_log_probability(s)) for s in strings]

        # All should be 0.25
        for p in probs:
            assert p == pytest.approx(0.25)

        # Sum should be 1.0
        assert sum(probs) == pytest.approx(1.0)

    def test_inside_probability_method(self, simple_pcfg):
        """inside_probability returns probability (not log)."""
        ic = InsideComputation(simple_pcfg)
        p = ic.inside_probability(('a', 'b'))
        assert p == pytest.approx(0.25)

    def test_unparseable_sentence_raises(self, simple_pcfg):
        """Unparseable sentence raises ParseFailureException."""
        ic = InsideComputation(simple_pcfg)

        # Single word can't be parsed with this grammar (needs 2 words)
        with pytest.raises(ParseFailureException):
            ic.inside_log_probability(('a',))

        # Unknown terminal
        with pytest.raises(ParseFailureException):
            ic.inside_log_probability(('x', 'y'))


class TestViterbiParsing:
    """Tests for Viterbi parsing."""

    def test_viterbi_parse_simple(self, simple_pcfg):
        """Viterbi parse returns a valid tree."""
        ic = InsideComputation(simple_pcfg)
        tree = ic.viterbi_parse(('a', 'b'))

        # Check tree structure
        assert tree[0] == 'S'
        assert len(tree) == 3
        assert collect_yield(tree) == ['a', 'b']

    def test_viterbi_parse_matches_sentence(self, simple_pcfg):
        """Viterbi tree yield matches input sentence."""
        ic = InsideComputation(simple_pcfg)

        for s in [('a', 'a'), ('a', 'b'), ('b', 'a'), ('b', 'b')]:
            tree = ic.viterbi_parse(s)
            assert tuple(collect_yield(tree)) == s

    def test_viterbi_parse_unparseable_raises(self, simple_pcfg):
        """Viterbi raises ParseFailureException for unparseable input."""
        ic = InsideComputation(simple_pcfg)

        # Unknown terminals should raise ParseFailureException
        with pytest.raises(ParseFailureException):
            ic.viterbi_parse(('x', 'y'))


class TestAmbiguousParsing:
    """Tests for parsing with ambiguous grammars."""

    def test_ambiguous_grammar_multiple_parses(self, ambiguous_pcfg):
        """Ambiguous grammar can parse same string multiple ways."""
        ic = InsideComputation(ambiguous_pcfg)

        # "aaa" has multiple derivations
        count = ic.count_parses(('a', 'a', 'a'))
        assert count > 1

    def test_ambiguous_grammar_probability(self, ambiguous_pcfg):
        """Probability accounts for all derivations."""
        ic = InsideComputation(ambiguous_pcfg)

        # P("aa") - two A's, each with P(A -> a) = 0.5
        # S -> A A (1.0), each A -> a (0.5)
        p_aa = ic.inside_probability(('a', 'a'))
        assert p_aa == pytest.approx(0.25)

    def test_count_parses_unambiguous(self, simple_pcfg):
        """Unambiguous grammar has exactly 1 parse."""
        ic = InsideComputation(simple_pcfg)
        count = ic.count_parses(('a', 'b'))
        assert count == 1


class TestBracketedParsing:
    """Tests for bracketed parsing (constrained by tree structure)."""

    def test_bracketed_log_probability(self, simple_pcfg):
        """Bracketed log probability matches unconstrained for unambiguous."""
        ic = InsideComputation(simple_pcfg)

        # For unambiguous grammar, bracketed should equal total
        tree = ('S', ('A', 'a'), ('B', 'b'))  # Wrong labels but right structure
        # The bracketed parse considers all labelings

        # With correct structure
        correct_tree = ('X', ('Y', 'a'), ('Z', 'b'))
        lp = ic.inside_bracketed_log_probability(correct_tree)
        # Should match inside probability since there's only one bracketing
        assert math.exp(lp) == pytest.approx(0.25)

    def test_bracketed_viterbi_parse(self, simple_pcfg):
        """Bracketed Viterbi returns tree with correct structure."""
        ic = InsideComputation(simple_pcfg)

        # Give a tree structure, get back labeled version
        unlabeled_tree = ('X', ('Y', 'a'), ('Z', 'b'))
        tree = ic.bracketed_viterbi_parse(unlabeled_tree)

        assert tree[0] == 'S'
        assert collect_yield(tree) == ['a', 'b']


class TestPosteriors:
    """Tests for posterior computation (E-step of EM)."""

    def test_add_posteriors_simple(self, simple_pcfg):
        """add_posteriors accumulates rule posteriors."""
        from collections import defaultdict
        ic = InsideComputation(simple_pcfg)
        posteriors = defaultdict(float)

        lp = ic.add_posteriors(('a', 'b'), posteriors)

        # Should have posteriors for all rules used
        assert ('S', 'A', 'A') in posteriors
        assert ('A', 'a') in posteriors
        assert ('A', 'b') in posteriors

        # For unambiguous parse, posteriors should be 1.0
        assert posteriors[('S', 'A', 'A')] == pytest.approx(1.0)
        assert posteriors[('A', 'a')] == pytest.approx(1.0)
        assert posteriors[('A', 'b')] == pytest.approx(1.0)

    def test_add_posteriors_multiple_sentences(self, simple_pcfg):
        """add_posteriors accumulates over multiple sentences."""
        from collections import defaultdict
        ic = InsideComputation(simple_pcfg)
        posteriors = defaultdict(float)

        ic.add_posteriors(('a', 'a'), posteriors)
        ic.add_posteriors(('b', 'b'), posteriors)

        # S -> A A used twice
        assert posteriors[('S', 'A', 'A')] == pytest.approx(2.0)
        # A -> a used twice in first sentence
        assert posteriors[('A', 'a')] == pytest.approx(2.0)
        # A -> b used twice in second sentence
        assert posteriors[('A', 'b')] == pytest.approx(2.0)

    def test_add_posteriors_weight(self, simple_pcfg):
        """add_posteriors respects weight parameter."""
        from collections import defaultdict
        ic = InsideComputation(simple_pcfg)
        posteriors = defaultdict(float)

        ic.add_posteriors(('a', 'b'), posteriors, weight=2.0)

        assert posteriors[('S', 'A', 'A')] == pytest.approx(2.0)

    def test_add_posteriors_returns_log_prob(self, simple_pcfg):
        """add_posteriors returns log probability of sentence."""
        from collections import defaultdict
        ic = InsideComputation(simple_pcfg)
        posteriors = defaultdict(float)

        lp = ic.add_posteriors(('a', 'b'), posteriors)
        assert lp == pytest.approx(math.log(0.25))


class TestThreeNonterminalGrammar:
    """Tests with a more complex 3-nonterminal grammar."""

    def test_parse_three_nt(self, three_nt_pcfg):
        """Parse with 3-nonterminal grammar."""
        ic = InsideComputation(three_nt_pcfg)

        # "ab" can be S -> A B or S -> B A
        tree = ic.viterbi_parse(('a', 'b'))
        assert collect_yield(tree) == ['a', 'b']

    def test_probability_three_nt(self, three_nt_pcfg):
        """Probability computation with 3-nonterminal grammar."""
        ic = InsideComputation(three_nt_pcfg)

        # P("ab") = P(S -> A B) * P(A -> a) * P(B -> b)
        #         + P(S -> B A) * P(B -> a)? - no B -> a
        # So only S -> A B path: 0.6 * 0.7 * 0.8 = 0.336
        p = ic.inside_probability(('a', 'b'))
        assert p == pytest.approx(0.336)

    def test_viterbi_parse_unk(self, three_nt_pcfg):
        """Viterbi with unknown word replacement."""
        # Add UNK handling
        unk_grammar = three_nt_pcfg.unkify({'a', 'b'}, '<UNK>')
        ic = InsideComputation(unk_grammar)

        # 'c' was grouped into UNK
        tree = ic.viterbi_parse_unk(('a', 'c'), '<UNK>')
        assert collect_yield(tree) == ['a', '<UNK>']


class TestLogDerivationScoring:
    """Tests for scoring derivation trees."""

    def test_log_probability_derivation(self, simple_pcfg):
        """Score a derivation tree."""
        tree = ('S', ('A', 'a'), ('A', 'b'))
        lp = simple_pcfg.log_probability_derivation(tree)

        # P = P(S -> A A) * P(A -> a) * P(A -> b) = 1.0 * 0.5 * 0.5 = 0.25
        assert lp == pytest.approx(math.log(0.25))

    def test_log_score_derivation(self, simple_pcfg):
        """Alternative derivation scoring method."""
        tree = ('S', ('A', 'a'), ('A', 'b'))
        lp = simple_pcfg.log_score_derivation(tree)
        assert lp == pytest.approx(math.log(0.25))

    def test_weight_derivation(self, simple_pcfg):
        """Weight derivation returns probability (not log)."""
        tree = ('S', ('A', 'a'), ('A', 'b'))
        p = simple_pcfg.weight_derivation(tree)
        assert p == pytest.approx(0.25)
