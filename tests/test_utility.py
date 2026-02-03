"""Tests for utility.py - tree operations, spans, and helper functions."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'locallearner'))

import utility
from utility import (
    collect_yield,
    string_to_tree,
    tree_to_string,
    collect_labeled_spans,
    collect_unlabeled_spans,
    strongly_connected_components,
    tree_depth,
    tree_to_unlabeled_tree,
    unlabeled_tree_equal,
    count_productions,
    ParseFailureException,
    DivergentWCFGException,
)
from collections import Counter


class TestCollectYield:
    """Tests for collect_yield function."""

    def test_lexical_tree(self):
        """A simple lexical tree (A, 'word')."""
        tree = ('A', 'word')
        assert collect_yield(tree) == ['word']

    def test_binary_tree(self):
        """A binary tree with two leaves."""
        tree = ('S', ('A', 'hello'), ('B', 'world'))
        assert collect_yield(tree) == ['hello', 'world']

    def test_nested_tree(self):
        """A more deeply nested tree."""
        tree = ('S',
                ('A', ('A', 'a'), ('B', 'b')),
                ('C', 'c'))
        assert collect_yield(tree) == ['a', 'b', 'c']

    def test_left_branching(self):
        """Left-branching tree."""
        tree = ('S',
                ('A',
                 ('A', ('A', 'a'), ('B', 'b')),
                 ('C', 'c')),
                ('D', 'd'))
        assert collect_yield(tree) == ['a', 'b', 'c', 'd']


class TestTreeStringConversion:
    """Tests for string_to_tree and tree_to_string."""

    def test_simple_tree(self):
        """Simple lexical production."""
        s = "(A word)"
        tree = string_to_tree(s)
        assert tree == ('A', 'word')

    def test_binary_tree(self):
        """Binary tree."""
        s = "(S (A a) (B b))"
        tree = string_to_tree(s)
        assert tree == ('S', ('A', 'a'), ('B', 'b'))

    def test_nested_tree(self):
        """Nested tree structure."""
        s = "(S (A (A a) (B b)) (C c))"
        tree = string_to_tree(s)
        expected = ('S', ('A', ('A', 'a'), ('B', 'b')), ('C', 'c'))
        assert tree == expected

    def test_tree_to_string_lexical(self):
        """tree_to_string on lexical production."""
        tree = ('A', 'word')
        assert tree_to_string(tree) == "(A word)"

    def test_tree_to_string_binary(self):
        """tree_to_string on binary tree."""
        tree = ('S', ('A', 'a'), ('B', 'b'))
        assert tree_to_string(tree) == "(S (A a) (B b))"

    def test_round_trip(self):
        """Round-trip: string -> tree -> string."""
        original = "(S (A (A a) (B b)) (C c))"
        tree = string_to_tree(original)
        result = tree_to_string(tree)
        # Re-parse to compare structure
        assert string_to_tree(result) == tree


class TestSpans:
    """Tests for span collection functions."""

    def test_labeled_spans_lexical(self):
        """Labeled spans for a lexical tree."""
        tree = ('A', 'word')
        spans = collect_labeled_spans(tree)
        # Single span: (label, start, end)
        assert ('A', 0, 1) in spans

    def test_labeled_spans_binary(self):
        """Labeled spans for a binary tree."""
        tree = ('S', ('A', 'a'), ('B', 'b'))
        spans = collect_labeled_spans(tree)
        assert ('A', 0, 1) in spans
        assert ('B', 1, 2) in spans
        assert ('S', 0, 2) in spans

    def test_unlabeled_spans_binary(self):
        """Unlabeled spans for a binary tree (excludes root span)."""
        tree = ('S', ('A', 'a'), ('B', 'b'))
        spans = collect_unlabeled_spans(tree)
        # Unlabeled spans don't include the full span
        assert len(spans) == 0  # Only binary spans, and (0,2) is excluded

    def test_unlabeled_spans_nested(self):
        """Unlabeled spans for a nested tree."""
        tree = ('S', ('A', ('A', 'a'), ('B', 'b')), ('C', 'c'))
        spans = collect_unlabeled_spans(tree)
        # Should include (0, 2) for the left subtree
        assert (0, 2) in spans

    def test_span_count(self):
        """Count of spans matches expected."""
        tree = ('S', ('A', ('A', 'a'), ('B', 'b')), ('C', 'c'))
        labeled = collect_labeled_spans(tree)
        # 5 nodes total
        assert len(labeled) == 5


class TestUnlabeledTree:
    """Tests for unlabeled tree operations."""

    def test_tree_to_unlabeled_lexical(self):
        """Convert lexical tree to unlabeled."""
        tree = ('A', 'word')
        assert tree_to_unlabeled_tree(tree) == 'word'

    def test_tree_to_unlabeled_binary(self):
        """Convert binary tree to unlabeled."""
        tree = ('S', ('A', 'a'), ('B', 'b'))
        assert tree_to_unlabeled_tree(tree) == ('a', 'b')

    def test_unlabeled_tree_equal_same(self):
        """Two trees with same structure are equal when unlabeled."""
        tree1 = ('S', ('A', 'a'), ('B', 'b'))
        tree2 = ('X', ('Y', 'a'), ('Z', 'b'))
        assert unlabeled_tree_equal(tree1, tree2)

    def test_unlabeled_tree_equal_different(self):
        """Trees with different structure are not equal."""
        tree1 = ('S', ('A', 'a'), ('B', 'b'))
        tree2 = ('S', ('A', ('A', 'a'), ('B', 'b')), ('C', 'c'))
        assert not unlabeled_tree_equal(tree1, tree2)


class TestTreeDepth:
    """Tests for tree_depth function."""

    def test_depth_lexical(self):
        """Lexical tree has depth 1."""
        tree = ('A', 'word')
        assert tree_depth(tree) == 1

    def test_depth_binary(self):
        """Simple binary tree has depth 2."""
        tree = ('S', ('A', 'a'), ('B', 'b'))
        assert tree_depth(tree) == 2

    def test_depth_nested(self):
        """Nested tree depth."""
        tree = ('S', ('A', ('A', 'a'), ('B', 'b')), ('C', 'c'))
        assert tree_depth(tree) == 3


class TestCountProductions:
    """Tests for count_productions function."""

    def test_lexical_only(self):
        """Count productions in a lexical tree."""
        tree = ('A', 'word')
        counter = Counter()
        count_productions(tree, counter)
        assert counter[('A', 'word')] == 1

    def test_binary_tree(self):
        """Count productions in a binary tree."""
        tree = ('S', ('A', 'a'), ('B', 'b'))
        counter = Counter()
        count_productions(tree, counter)
        assert counter[('S', 'A', 'B')] == 1
        assert counter[('A', 'a')] == 1
        assert counter[('B', 'b')] == 1


class TestStronglyConnectedComponents:
    """Tests for Tarjan's SCC algorithm."""

    def test_no_edges(self):
        """Graph with no edges - each node is its own SCC."""
        graph = {'A': [], 'B': [], 'C': []}
        sccs = strongly_connected_components(graph)
        assert len(sccs) == 3
        for scc in sccs:
            assert len(scc) == 1

    def test_simple_cycle(self):
        """Simple cycle A -> B -> A."""
        graph = {'A': ['B'], 'B': ['A']}
        sccs = strongly_connected_components(graph)
        assert len(sccs) == 1
        assert set(sccs[0]) == {'A', 'B'}

    def test_two_sccs(self):
        """Two separate SCCs."""
        graph = {
            'A': ['B'],
            'B': ['A'],
            'C': ['D'],
            'D': ['C'],
        }
        sccs = strongly_connected_components(graph)
        assert len(sccs) == 2
        scc_sets = [set(scc) for scc in sccs]
        assert {'A', 'B'} in scc_sets
        assert {'C', 'D'} in scc_sets

    def test_dag(self):
        """DAG - no cycles, each node is its own SCC."""
        graph = {'A': ['B', 'C'], 'B': ['C'], 'C': []}
        sccs = strongly_connected_components(graph)
        assert len(sccs) == 3

    def test_self_loop(self):
        """Node with self-loop."""
        graph = {'A': ['A']}
        sccs = strongly_connected_components(graph)
        assert len(sccs) == 1
        assert 'A' in sccs[0]


class TestExceptions:
    """Tests for custom exceptions."""

    def test_parse_failure_exception(self):
        """ParseFailureException can be raised and caught."""
        with pytest.raises(ParseFailureException):
            raise ParseFailureException("test")

    def test_divergent_wcfg_exception(self):
        """DivergentWCFGException can be raised and caught."""
        with pytest.raises(DivergentWCFGException):
            raise DivergentWCFGException()


class TestCatalanNumbers:
    """Tests for catalan_numbers function."""

    def test_catalan_0(self):
        """C(0) = 1."""
        assert utility.catalan_numbers(0) == 1

    def test_catalan_1(self):
        """C(1) = 1."""
        assert utility.catalan_numbers(1) == 1

    def test_catalan_2(self):
        """C(2) = 2."""
        assert utility.catalan_numbers(2) == 2

    def test_catalan_3(self):
        """C(3) = 5."""
        assert utility.catalan_numbers(3) == 5

    def test_catalan_4(self):
        """C(4) = 14."""
        assert utility.catalan_numbers(4) == 14

    def test_catalan_5(self):
        """C(5) = 42."""
        assert utility.catalan_numbers(5) == 42


class TestRandomBinaryTree:
    """Tests for random binary tree generation."""

    def test_random_binary_tree_sentence_length_1(self):
        """Random tree with 1-word sentence."""
        sentence = ('a',)
        tree = utility.random_binary_tree_sentence(sentence, 'X')
        assert collect_yield(tree) == ['a']
        assert tree[0] == 'X'

    def test_random_binary_tree_sentence_length_2(self):
        """Random tree with 2-word sentence."""
        sentence = ('a', 'b')
        tree = utility.random_binary_tree_sentence(sentence, 'X')
        assert collect_yield(tree) == ['a', 'b']

    def test_random_binary_tree_sentence_preserves_order(self):
        """Random tree preserves word order."""
        sentence = ('the', 'cat', 'sat')
        tree = utility.random_binary_tree_sentence(sentence, 'X')
        assert collect_yield(tree) == list(sentence)
