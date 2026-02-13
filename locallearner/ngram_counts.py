"""Count, save, and load bigram and trigram counts from a corpus.

Sentence boundaries are represented by the token '<s>'.

File format (tab-separated, one ngram per line):
  Bigrams:   word1 \\t word2 \\t count
  Trigrams:  word1 \\t word2 \\t word3 \\t count

First line is a header comment with metadata.
Files are plain text, gzip-compressed for compactness (.gz suffix).

Usage:
    # From a list of sentences (each a tuple of words):
    bigrams = count_bigrams(sentences)
    save_bigrams(bigrams, 'corpus.bigrams.gz')

    bigrams = load_bigrams('corpus.bigrams.gz')

    trigrams = count_trigrams(sentences)
    save_trigrams(trigrams, 'corpus.trigrams.gz')

    trigrams = load_trigrams('corpus.trigrams.gz')
"""

import gzip
from collections import Counter

BOUNDARY = '<s>'


def count_bigrams(sentences):
    """Count all bigrams in a corpus, including sentence boundaries.

    Each sentence is padded with <s> on both sides:
      <s> w1 w2 ... wn <s>
    producing bigrams: (<s>,w1), (w1,w2), ..., (wn,<s>).

    Args:
        sentences: iterable of tuples/lists of word strings.

    Returns:
        Counter mapping (word1, word2) -> count.
    """
    counts = Counter()
    for s in sentences:
        prev = BOUNDARY
        for w in s:
            counts[(prev, w)] += 1
            prev = w
        counts[(prev, BOUNDARY)] += 1
    return counts


def count_trigrams(sentences):
    """Count all trigrams in a corpus, including sentence boundaries.

    Each sentence is padded with <s> on both sides:
      <s> w1 w2 ... wn <s>
    producing trigrams: (<s>,w1,w2), (w1,w2,w3), ..., (wn-1,wn,<s>).
    For length-1 sentences: (<s>,w1,<s>).
    For length-2 sentences: (<s>,w1,w2), (w1,w2,<s>).

    Args:
        sentences: iterable of tuples/lists of word strings.

    Returns:
        Counter mapping (word1, word2, word3) -> count.
    """
    counts = Counter()
    for s in sentences:
        n = len(s)
        if n == 0:
            continue
        if n == 1:
            counts[(BOUNDARY, s[0], BOUNDARY)] += 1
            continue
        # First trigram
        counts[(BOUNDARY, s[0], s[1])] += 1
        # Internal trigrams
        for i in range(1, n - 1):
            counts[(s[i - 1], s[i], s[i + 1])] += 1
        # Last trigram
        counts[(s[n - 2], s[n - 1], BOUNDARY)] += 1
    return counts


def save_bigrams(counts, filename):
    """Save bigram counts to a gzip-compressed tab-separated file."""
    opener = gzip.open if filename.endswith('.gz') else open
    with opener(filename, 'wt') as f:
        f.write(f'# bigrams n_types={len(counts)} '
                f'n_tokens={sum(counts.values())}\n')
        for (w1, w2), c in sorted(counts.items()):
            f.write(f'{w1}\t{w2}\t{c}\n')


def save_trigrams(counts, filename):
    """Save trigram counts to a gzip-compressed tab-separated file."""
    opener = gzip.open if filename.endswith('.gz') else open
    with opener(filename, 'wt') as f:
        f.write(f'# trigrams n_types={len(counts)} '
                f'n_tokens={sum(counts.values())}\n')
        for (w1, w2, w3), c in sorted(counts.items()):
            f.write(f'{w1}\t{w2}\t{w3}\t{c}\n')


def load_bigrams(filename):
    """Load bigram counts from a tab-separated file (plain or gzipped).

    Returns:
        Counter mapping (word1, word2) -> int count.
    """
    counts = Counter()
    opener = gzip.open if filename.endswith('.gz') else open
    with opener(filename, 'rt') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.rstrip('\n').split('\t')
            counts[(parts[0], parts[1])] = int(parts[2])
    return counts


def load_trigrams(filename):
    """Load trigram counts from a tab-separated file (plain or gzipped).

    Returns:
        Counter mapping (word1, word2, word3) -> int count.
    """
    counts = Counter()
    opener = gzip.open if filename.endswith('.gz') else open
    with opener(filename, 'rt') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.rstrip('\n').split('\t')
            counts[(parts[0], parts[1], parts[2])] = int(parts[3])
    return counts


def count_and_save(sentences, bigram_file, trigram_file=None):
    """Count bigrams (and optionally trigrams) and save them.

    Convenience function for the common case of saving both at once.
    """
    bigrams = count_bigrams(sentences)
    save_bigrams(bigrams, bigram_file)

    if trigram_file is not None:
        trigrams = count_trigrams(sentences)
        save_trigrams(trigrams, trigram_file)
        return bigrams, trigrams
    return bigrams, None
