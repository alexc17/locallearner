"""Shared fixtures and test utilities for LocalLearner test suite."""

import os
import sys
import tempfile
import pytest
import numpy as np

# Add locallearner to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'locallearner'))

import wcfg
import utility


# Fixture paths
FIXTURES_DIR = os.path.join(os.path.dirname(__file__), 'fixtures')


@pytest.fixture
def simple_grammar_path():
    """Path to the simple grammar fixture."""
    return os.path.join(FIXTURES_DIR, 'simple_grammar.pcfg')


@pytest.fixture
def ambiguous_grammar_path():
    """Path to the ambiguous grammar fixture."""
    return os.path.join(FIXTURES_DIR, 'ambiguous_grammar.pcfg')


@pytest.fixture
def sample_corpus_path():
    """Path to the sample corpus fixture."""
    return os.path.join(FIXTURES_DIR, 'sample_corpus.txt')


@pytest.fixture
def simple_grammar(simple_grammar_path):
    """Load the simple grammar as a WCFG object."""
    return wcfg.load_wcfg_from_file(simple_grammar_path)


@pytest.fixture
def ambiguous_grammar(ambiguous_grammar_path):
    """Load the ambiguous grammar as a WCFG object."""
    return wcfg.load_wcfg_from_file(ambiguous_grammar_path)


@pytest.fixture
def simple_pcfg():
    """
    Create a simple normalized PCFG programmatically.

    S -> A A (1.0)
    A -> a (0.5)
    A -> b (0.5)

    Expected: P("aa") = P("ab") = P("ba") = P("bb") = 0.25
    """
    g = wcfg.WCFG()
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
def ambiguous_pcfg():
    """
    Create an ambiguous PCFG programmatically.

    S -> A A (1.0)
    A -> A A (0.5)
    A -> a (0.5)

    This grammar has multiple derivations for strings like "aaa".
    """
    g = wcfg.WCFG()
    g.nonterminals = {'S', 'A'}
    g.terminals = {'a'}
    g.start = 'S'

    g.productions = [
        ('S', 'A', 'A'),
        ('A', 'A', 'A'),
        ('A', 'a'),
    ]
    g.parameters = {
        ('S', 'A', 'A'): 1.0,
        ('A', 'A', 'A'): 0.5,
        ('A', 'a'): 0.5,
    }
    g.set_log_parameters()
    return g


@pytest.fixture
def three_nt_pcfg():
    """
    A PCFG with 3 nonterminals for more complex testing.

    S -> A B (0.6)
    S -> B A (0.4)
    A -> a (0.7)
    A -> c (0.3)
    B -> b (0.8)
    B -> c (0.2)
    """
    g = wcfg.WCFG()
    g.nonterminals = {'S', 'A', 'B'}
    g.terminals = {'a', 'b', 'c'}
    g.start = 'S'

    g.productions = [
        ('S', 'A', 'B'),
        ('S', 'B', 'A'),
        ('A', 'a'),
        ('A', 'c'),
        ('B', 'b'),
        ('B', 'c'),
    ]
    g.parameters = {
        ('S', 'A', 'B'): 0.6,
        ('S', 'B', 'A'): 0.4,
        ('A', 'a'): 0.7,
        ('A', 'c'): 0.3,
        ('B', 'b'): 0.8,
        ('B', 'c'): 0.2,
    }
    g.set_log_parameters()
    return g


@pytest.fixture
def tmp_grammar_file(simple_pcfg):
    """Create a temporary grammar file and return its path."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pcfg', delete=False) as f:
        simple_pcfg.store(f.name)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def tmp_corpus_file():
    """Create a temporary corpus file and return its path."""
    sentences = [
        'a b a b',
        'b a b a',
        'a a b b',
        'b b a a',
        'a b b a',
        'b a a b',
    ]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write('\n'.join(sentences))
        f.write('\n')
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def rng():
    """Provide a seeded random number generator for reproducibility."""
    return np.random.default_rng(42)
