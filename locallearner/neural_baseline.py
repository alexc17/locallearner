"""
Neural baseline LSTM language model for comparison with PCFG learners.

Provides exact string log-likelihoods via autoregressive chain rule:
P(x) = ∏ P(x_t | x_{<t})
"""

import math
from collections import Counter
from typing import List, Tuple, Optional, Dict, Any

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def get_device() -> 'torch.device':
    """Select best available device (MPS > CUDA > CPU)."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for neural baseline. Install with: pip install torch")

    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class Vocabulary:
    """Vocabulary for mapping words to indices with special tokens."""

    PAD_TOKEN = "<pad>"
    SOS_TOKEN = "<sos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"

    def __init__(self, min_count: int = 1):
        self.min_count = min_count
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.word_counts: Counter = Counter()

        # Initialize special tokens
        self._add_special_tokens()

    def _add_special_tokens(self):
        """Add special tokens to vocabulary."""
        for token in [self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]:
            idx = len(self.word2idx)
            self.word2idx[token] = idx
            self.idx2word[idx] = token

    @property
    def pad_idx(self) -> int:
        return self.word2idx[self.PAD_TOKEN]

    @property
    def sos_idx(self) -> int:
        return self.word2idx[self.SOS_TOKEN]

    @property
    def eos_idx(self) -> int:
        return self.word2idx[self.EOS_TOKEN]

    @property
    def unk_idx(self) -> int:
        return self.word2idx[self.UNK_TOKEN]

    def __len__(self) -> int:
        return len(self.word2idx)

    def build_from_sentences(self, sentences: List[List[str]]):
        """Build vocabulary from list of tokenized sentences."""
        # Count word frequencies
        for sentence in sentences:
            self.word_counts.update(sentence)

        # Add words meeting minimum count threshold
        for word, count in self.word_counts.items():
            if count >= self.min_count and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def encode(self, word: str) -> int:
        """Encode a word to its index."""
        return self.word2idx.get(word, self.unk_idx)

    def decode(self, idx: int) -> str:
        """Decode an index to its word."""
        return self.idx2word.get(idx, self.UNK_TOKEN)

    def encode_sentence(self, sentence: List[str], add_special: bool = True) -> List[int]:
        """Encode a sentence to indices, optionally adding SOS/EOS."""
        indices = [self.encode(w) for w in sentence]
        if add_special:
            indices = [self.sos_idx] + indices + [self.eos_idx]
        return indices

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'word2idx': self.word2idx,
            'idx2word': {str(k): v for k, v in self.idx2word.items()},
            'min_count': self.min_count,
            'word_counts': dict(self.word_counts),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Vocabulary':
        """Create from dictionary."""
        vocab = cls(min_count=data['min_count'])
        vocab.word2idx = data['word2idx']
        vocab.idx2word = {int(k): v for k, v in data['idx2word'].items()}
        vocab.word_counts = Counter(data['word_counts'])
        return vocab


class SentenceDataset(Dataset):
    """PyTorch dataset for sentences."""

    def __init__(self, sentences: List[List[str]], vocab: Vocabulary):
        self.sentences = sentences
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> Tuple['torch.Tensor', 'torch.Tensor']:
        sentence = self.sentences[idx]
        encoded = self.vocab.encode_sentence(sentence, add_special=True)

        # Input: <sos> w1 w2 ... wn
        # Target: w1 w2 ... wn <eos>
        input_seq = torch.tensor(encoded[:-1], dtype=torch.long)
        target_seq = torch.tensor(encoded[1:], dtype=torch.long)

        return input_seq, target_seq


def collate_fn(batch: List[Tuple['torch.Tensor', 'torch.Tensor']], pad_idx: int) -> Tuple['torch.Tensor', 'torch.Tensor', 'torch.Tensor']:
    """Collate function for batching with padding."""
    inputs, targets = zip(*batch)

    # Get lengths for masking
    lengths = torch.tensor([len(seq) for seq in inputs])

    # Pad sequences
    inputs_padded = nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=pad_idx)
    targets_padded = nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=pad_idx)

    return inputs_padded, targets_padded, lengths


class LSTMLanguageModel(nn.Module):
    """LSTM Language Model for computing exact string probabilities."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        vocab: Optional[Vocabulary] = None,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for neural baseline. Install with: pip install torch")

        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_p = dropout
        self.vocab = vocab

        # Model layers
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: 'torch.Tensor', lengths: Optional['torch.Tensor'] = None) -> 'torch.Tensor':
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len)
            lengths: Optional lengths for packing

        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        embedded = self.embedding(x)

        if lengths is not None:
            # Pack for variable length sequences
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            outputs, _ = self.lstm(packed)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        else:
            outputs, _ = self.lstm(embedded)

        outputs = self.dropout(outputs)
        logits = self.output(outputs)

        return logits

    def fit(
        self,
        sentences: List[List[str]],
        epochs: int = 100,
        lr: float = 0.001,
        batch_size: int = 32,
        device: Optional['torch.device'] = None,
        verbose: bool = True,
        min_count: int = 1,
    ) -> List[float]:
        """
        Train the language model.

        Args:
            sentences: List of tokenized sentences
            epochs: Number of training epochs
            lr: Learning rate
            batch_size: Batch size
            device: Device to train on
            verbose: Print training progress
            min_count: Minimum word count for vocabulary

        Returns:
            List of training losses per epoch
        """
        if device is None:
            device = get_device()

        # Build vocabulary if not provided
        if self.vocab is None:
            self.vocab = Vocabulary(min_count=min_count)
            self.vocab.build_from_sentences(sentences)

            # Reinitialize model with correct vocab size
            self.vocab_size = len(self.vocab)
            self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
            self.output = nn.Linear(self.hidden_dim, self.vocab_size)

        self.to(device)

        # Create dataset and dataloader
        dataset = SentenceDataset(sentences, self.vocab)

        def collate_with_pad(batch):
            return collate_fn(batch, self.vocab.pad_idx)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_with_pad,
        )

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss(ignore_index=self.vocab.pad_idx)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        losses = []

        for epoch in range(epochs):
            self.train()
            total_loss = 0.0
            num_batches = 0

            for inputs, targets, lengths in dataloader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                lengths = lengths.to(device)

                optimizer.zero_grad()

                logits = self(inputs, lengths)

                # Reshape for loss computation
                loss = criterion(
                    logits.view(-1, self.vocab_size),
                    targets.view(-1)
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            losses.append(avg_loss)

            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        return losses

    def log_probability(self, sentence: Tuple[str, ...], device: Optional['torch.device'] = None) -> float:
        """
        Compute exact log P(sentence) using the chain rule.

        Args:
            sentence: Tuple of words (matches WCFG interface)

        Returns:
            Log probability of the sentence
        """
        if self.vocab is None:
            raise ValueError("Model has not been trained - no vocabulary available")

        if device is None:
            device = next(self.parameters()).device

        self.eval()

        # Encode sentence with special tokens
        encoded = self.vocab.encode_sentence(list(sentence), add_special=True)

        # Input: <sos> w1 w2 ... wn
        # Target: w1 w2 ... wn <eos>
        input_seq = torch.tensor(encoded[:-1], dtype=torch.long).unsqueeze(0).to(device)
        target_seq = encoded[1:]

        with torch.no_grad():
            logits = self(input_seq)  # (1, seq_len, vocab_size)
            log_probs = F.log_softmax(logits, dim=-1)  # (1, seq_len, vocab_size)

            # Sum log P(w_t | context) for each target token
            total_lp = 0.0
            for t, target_idx in enumerate(target_seq):
                total_lp += log_probs[0, t, target_idx].item()

        return total_lp

    def perplexity(self, sentences: List[Tuple[str, ...]], device: Optional['torch.device'] = None) -> float:
        """
        Compute perplexity on a corpus.

        Args:
            sentences: List of sentences (as tuples of words)

        Returns:
            Perplexity value
        """
        total_log_prob = 0.0
        total_tokens = 0

        for sentence in sentences:
            lp = self.log_probability(sentence, device)
            total_log_prob += lp
            total_tokens += len(sentence) + 1  # +1 for EOS

        # Perplexity = exp(-1/N * sum(log P))
        avg_log_prob = total_log_prob / total_tokens
        return math.exp(-avg_log_prob)

    def save(self, path: str):
        """Save model to file."""
        state = {
            'model_state_dict': self.state_dict(),
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'dropout': self.dropout_p,
            'vocab': self.vocab.to_dict() if self.vocab else None,
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: str, device: Optional['torch.device'] = None) -> 'LSTMLanguageModel':
        """Load model from file."""
        if device is None:
            device = get_device()

        state = torch.load(path, map_location=device, weights_only=False)

        vocab = None
        if state['vocab'] is not None:
            vocab = Vocabulary.from_dict(state['vocab'])

        model = cls(
            vocab_size=state['vocab_size'],
            embed_dim=state['embed_dim'],
            hidden_dim=state['hidden_dim'],
            num_layers=state['num_layers'],
            dropout=state['dropout'],
            vocab=vocab,
        )

        model.load_state_dict(state['model_state_dict'])
        model.to(device)
        model.eval()

        return model


def load_corpus(path: str) -> List[List[str]]:
    """Load corpus from file (one sentence per line, space-separated tokens)."""
    sentences = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                sentences.append(line.split())
    return sentences
