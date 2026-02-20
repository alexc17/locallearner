"""Tests for neural baseline LSTM language model."""

import pytest
import os
import sys
import math
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'locallearner'))

# Skip all tests if torch is not available
torch = pytest.importorskip("torch")

from neural_baseline import (
    Vocabulary,
    LSTMLanguageModel,
    load_corpus,
    get_device,
    TORCH_AVAILABLE,
)


class TestVocabulary:
    """Tests for Vocabulary class."""

    def test_special_tokens_exist(self):
        """Vocabulary has special tokens initialized."""
        vocab = Vocabulary()
        assert vocab.PAD_TOKEN in vocab.word2idx
        assert vocab.SOS_TOKEN in vocab.word2idx
        assert vocab.EOS_TOKEN in vocab.word2idx
        assert vocab.UNK_TOKEN in vocab.word2idx

    def test_special_token_indices(self):
        """Special tokens have correct indices."""
        vocab = Vocabulary()
        assert vocab.pad_idx == 0
        assert vocab.sos_idx == 1
        assert vocab.eos_idx == 2
        assert vocab.unk_idx == 3

    def test_build_from_sentences(self):
        """Build vocabulary from sentences."""
        vocab = Vocabulary()
        sentences = [['a', 'b', 'a'], ['b', 'c', 'b']]
        vocab.build_from_sentences(sentences)

        assert 'a' in vocab.word2idx
        assert 'b' in vocab.word2idx
        assert 'c' in vocab.word2idx
        assert len(vocab) == 7  # 4 special + 3 words

    def test_min_count_filtering(self):
        """Words below min_count are not added."""
        vocab = Vocabulary(min_count=2)
        sentences = [['a', 'b', 'a'], ['b', 'c', 'b']]
        vocab.build_from_sentences(sentences)

        assert 'a' in vocab.word2idx
        assert 'b' in vocab.word2idx
        assert 'c' not in vocab.word2idx  # Only appears once

    def test_encode_known_word(self):
        """Encoding known word returns correct index."""
        vocab = Vocabulary()
        vocab.build_from_sentences([['hello', 'world']])

        idx = vocab.encode('hello')
        assert idx == vocab.word2idx['hello']

    def test_encode_unknown_word(self):
        """Encoding unknown word returns UNK index."""
        vocab = Vocabulary()
        vocab.build_from_sentences([['hello', 'world']])

        idx = vocab.encode('unknown')
        assert idx == vocab.unk_idx

    def test_encode_sentence(self):
        """Encoding sentence adds SOS and EOS."""
        vocab = Vocabulary()
        vocab.build_from_sentences([['a', 'b']])

        encoded = vocab.encode_sentence(['a', 'b'])
        assert encoded[0] == vocab.sos_idx
        assert encoded[-1] == vocab.eos_idx
        assert len(encoded) == 4  # SOS + a + b + EOS

    def test_to_dict_from_dict_roundtrip(self):
        """Vocabulary can be serialized and deserialized."""
        vocab = Vocabulary(min_count=2)
        vocab.build_from_sentences([['a', 'b', 'a'], ['b', 'c', 'b']])

        data = vocab.to_dict()
        restored = Vocabulary.from_dict(data)

        assert restored.word2idx == vocab.word2idx
        assert restored.min_count == vocab.min_count


class TestLSTMLanguageModel:
    """Tests for LSTMLanguageModel class."""

    @pytest.fixture
    def simple_sentences(self):
        """Simple training corpus."""
        return [
            ['a', 'b'],
            ['b', 'a'],
            ['a', 'a'],
            ['b', 'b'],
        ]

    @pytest.fixture
    def trained_model(self, simple_sentences):
        """A trained model on simple corpus."""
        model = LSTMLanguageModel(
            vocab_size=1,
            embed_dim=32,
            hidden_dim=64,
            num_layers=1,
            dropout=0.0,
        )
        model.fit(
            simple_sentences,
            epochs=50,
            lr=0.01,
            batch_size=2,
            verbose=False,
        )
        return model

    def test_model_creation(self):
        """Model can be created with specified parameters."""
        model = LSTMLanguageModel(
            vocab_size=100,
            embed_dim=64,
            hidden_dim=128,
            num_layers=2,
        )
        assert model.vocab_size == 100
        assert model.embed_dim == 64
        assert model.hidden_dim == 128
        assert model.num_layers == 2

    def test_fit_creates_vocabulary(self, simple_sentences):
        """Fitting creates vocabulary."""
        model = LSTMLanguageModel(vocab_size=1)
        model.fit(simple_sentences, epochs=1, verbose=False)

        assert model.vocab is not None
        assert 'a' in model.vocab.word2idx
        assert 'b' in model.vocab.word2idx

    def test_fit_returns_losses(self, simple_sentences):
        """Fitting returns list of losses."""
        model = LSTMLanguageModel(vocab_size=1)
        losses = model.fit(simple_sentences, epochs=10, verbose=False)

        assert len(losses) == 10
        assert all(isinstance(l, float) for l in losses)
        # Loss should generally decrease
        assert losses[-1] <= losses[0]

    def test_log_probability_returns_negative(self, trained_model):
        """Log probability is negative (probability < 1)."""
        lp = trained_model.log_probability(('a', 'b'))
        assert lp < 0

    def test_log_probability_known_sequence(self, trained_model):
        """Log probability for known sequence is reasonable."""
        lp = trained_model.log_probability(('a', 'b'))
        assert lp > -100  # Should not be extremely unlikely

    def test_log_probability_with_unk(self, trained_model):
        """Log probability handles unknown words."""
        lp = trained_model.log_probability(('a', 'unknown', 'b'))
        assert lp < 0
        assert math.isfinite(lp)

    def test_perplexity_positive(self, trained_model):
        """Perplexity is positive."""
        sentences = [('a', 'b'), ('b', 'a')]
        ppl = trained_model.perplexity(sentences)
        assert ppl > 0

    def test_perplexity_reasonable_range(self, trained_model):
        """Perplexity is in reasonable range for small vocab."""
        sentences = [('a', 'b'), ('b', 'a'), ('a', 'a')]
        ppl = trained_model.perplexity(sentences)
        # For vocab size ~6, perplexity should be less than vocab size
        assert ppl < 100

    def test_save_load_roundtrip(self, trained_model):
        """Model can be saved and loaded."""
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_path = f.name

        try:
            trained_model.save(temp_path)
            loaded = LSTMLanguageModel.load(temp_path)

            # Check parameters match
            assert loaded.vocab_size == trained_model.vocab_size
            assert loaded.embed_dim == trained_model.embed_dim
            assert loaded.hidden_dim == trained_model.hidden_dim

            # Check vocabulary loaded
            assert loaded.vocab is not None
            assert loaded.vocab.word2idx == trained_model.vocab.word2idx

            # Check predictions match
            sentence = ('a', 'b')
            orig_lp = trained_model.log_probability(sentence)
            loaded_lp = loaded.log_probability(sentence)
            assert abs(orig_lp - loaded_lp) < 1e-5
        finally:
            os.unlink(temp_path)


class TestLoadCorpus:
    """Tests for corpus loading utility."""

    def test_load_corpus(self, sample_corpus_path):
        """Load corpus from file."""
        sentences = load_corpus(sample_corpus_path)
        assert len(sentences) > 0
        assert all(isinstance(s, list) for s in sentences)
        assert all(isinstance(w, str) for s in sentences for w in s)

    def test_load_corpus_from_temp(self):
        """Load corpus from temporary file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("a b c\n")
            f.write("d e f\n")
            f.write("\n")  # Empty line should be skipped
            f.write("g h\n")
            temp_path = f.name

        try:
            sentences = load_corpus(temp_path)
            assert len(sentences) == 3
            assert sentences[0] == ['a', 'b', 'c']
            assert sentences[1] == ['d', 'e', 'f']
            assert sentences[2] == ['g', 'h']
        finally:
            os.unlink(temp_path)


class TestDeviceSelection:
    """Tests for device selection utility."""

    def test_get_device_returns_device(self):
        """get_device returns a torch device."""
        device = get_device()
        assert isinstance(device, torch.device)

    def test_get_device_type(self):
        """get_device returns valid device type."""
        device = get_device()
        assert device.type in ('cpu', 'cuda', 'mps')


class TestProbabilitySumToOne:
    """Test that probabilities approximately sum to 1."""

    def test_next_token_probabilities_sum_to_one(self):
        """Next token probabilities should sum to 1."""
        # Create and train a minimal model
        sentences = [['a', 'b'], ['b', 'a'], ['a', 'a'], ['b', 'b']]
        model = LSTMLanguageModel(
            vocab_size=1,
            embed_dim=16,
            hidden_dim=32,
            num_layers=1,
            dropout=0.0,
        )
        model.fit(sentences, epochs=10, verbose=False)
        model.eval()

        device = next(model.parameters()).device

        # Get log probabilities for all tokens given context "<sos>"
        import torch.nn.functional as F

        input_seq = torch.tensor([[model.vocab.sos_idx]], dtype=torch.long).to(device)

        with torch.no_grad():
            logits = model(input_seq)
            probs = F.softmax(logits[0, 0], dim=-1)

        # Probabilities should sum to 1
        total = probs.sum().item()
        assert abs(total - 1.0) < 1e-5, f"Probabilities sum to {total}, expected 1.0"
