#!/usr/bin/env python3
"""Neural cloze model for learning word context representations.

Trains a model P(w_target | w_{-k}, ..., w_{-1}, w_{+1}, ..., w_{+k})
using a fixed context window of size k on each side.

The trained model can then be used to:
  - Compute Rényi divergences between words' context distributions
  - Identify anchor words for PCFG learning
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


BOUNDARY = '<s>'


class ClozeDataset(Dataset):
    """Dataset of (context, target) pairs extracted from sentences.

    For each word w_i in the corpus, the context is the k words on each
    side: (w_{i-k}, ..., w_{i-1}, w_{i+1}, ..., w_{i+k}).
    Sentence boundaries are padded with the BOUNDARY token.
    """

    def __init__(self, sentences, word2idx, k=2):
        """
        Args:
            sentences: list of lists of tokens
            word2idx: dict mapping token -> integer index
            k: context window size on each side
        """
        self.k = k
        self.bdy_idx = word2idx[BOUNDARY]
        self.contexts = []  # (2k,) int arrays
        self.targets = []   # int targets

        for sent in sentences:
            # Convert sentence to indices, with k boundary tokens on each side
            ids = [self.bdy_idx] * k + [word2idx[w] for w in sent] + [self.bdy_idx] * k
            for i in range(k, len(ids) - k):
                ctx = ids[i - k:i] + ids[i + 1:i + k + 1]
                self.contexts.append(ctx)
                self.targets.append(ids[i])

        self.contexts = np.array(self.contexts, dtype=np.int64)
        self.targets = np.array(self.targets, dtype=np.int64)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.contexts[idx], self.targets[idx]


class ClozeModel(nn.Module):
    """Neural cloze model: predict center word from context window.

    Architecture:
        context words -> shared embedding -> average -> hidden (ReLU) -> output (softmax)

    The context embeddings are averaged (bag-of-words style), then
    passed through a hidden layer with ReLU activation, then a linear
    output layer with softmax over the vocabulary.
    """

    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.hidden = nn.Linear(embedding_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)

        # Initialize weights
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.hidden.weight)
        nn.init.zeros_(self.hidden.bias)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, context):
        """
        Args:
            context: (batch, 2k) integer tensor of context word indices

        Returns:
            logits: (batch, vocab_size) unnormalized log-probabilities
        """
        # (batch, 2k, embedding_dim)
        emb = self.embedding(context)
        # Average over context positions: (batch, embedding_dim)
        avg = emb.mean(dim=1)
        # Hidden layer with ReLU: (batch, hidden_dim)
        h = torch.relu(self.hidden(avg))
        # Output logits: (batch, vocab_size)
        logits = self.output(h)
        return logits

    def log_probs(self, context):
        """Compute log P(w | context) for all words w.

        Args:
            context: (batch, 2k) integer tensor

        Returns:
            log_probs: (batch, vocab_size) log-probabilities
        """
        logits = self.forward(context)
        return torch.log_softmax(logits, dim=-1)


class PairClozeDataset(Dataset):
    """Dataset of (context, target1, target2) triples from sentences.

    For each adjacent pair (w_i, w_{i+1}), the context is the k words
    on each side of the pair:
        (w_{i-k}, ..., w_{i-1}, w_{i+2}, ..., w_{i+k+1})
    Sentence boundaries are padded with the BOUNDARY token.
    """

    def __init__(self, sentences, word2idx, k=2):
        self.k = k
        self.bdy_idx = word2idx[BOUNDARY]
        self.contexts = []   # (2k,) int arrays
        self.targets1 = []   # left word of pair
        self.targets2 = []   # right word of pair

        for sent in sentences:
            ids = [self.bdy_idx] * k + [word2idx[w] for w in sent] + [self.bdy_idx] * k
            # Each pair (ids[i], ids[i+1]) for valid positions
            for i in range(k, len(ids) - k - 1):
                ctx = ids[i - k:i] + ids[i + 2:i + 2 + k]
                self.contexts.append(ctx)
                self.targets1.append(ids[i])
                self.targets2.append(ids[i + 1])

        self.contexts = np.array(self.contexts, dtype=np.int64)
        self.targets1 = np.array(self.targets1, dtype=np.int64)
        self.targets2 = np.array(self.targets2, dtype=np.int64)

    def __len__(self):
        return len(self.targets1)

    def __getitem__(self, idx):
        return self.contexts[idx], self.targets1[idx], self.targets2[idx]


class PairClozeModel(nn.Module):
    """Neural pair cloze model: predict two adjacent words from context.

    Uses autoregressive factorization:
        P(w1, w2 | ctx) = P(w1 | ctx) * P(w2 | ctx, w1)

    Architecture:
        context words -> shared embedding -> average -> hidden1 (ReLU) -> w1 logits
        [hidden1 ; w1_embedding] -> hidden2 (ReLU) -> w2 logits
    """

    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.hidden1 = nn.Linear(embedding_dim, hidden_dim)
        self.output1 = nn.Linear(hidden_dim, vocab_size)
        self.hidden2 = nn.Linear(hidden_dim + embedding_dim, hidden_dim)
        self.output2 = nn.Linear(hidden_dim, vocab_size)

        # Initialize weights
        nn.init.xavier_uniform_(self.embedding.weight)
        for layer in [self.hidden1, self.output1, self.hidden2, self.output2]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, context, w1=None):
        """
        Args:
            context: (batch, 2k) integer tensor of context word indices
            w1: (batch,) integer tensor of first word (for teacher forcing).
                If None, only returns logits1.

        Returns:
            logits1: (batch, vocab_size) logits for first word
            logits2: (batch, vocab_size) logits for second word (only if w1 given)
        """
        emb = self.embedding(context)       # (batch, 2k, emb_dim)
        avg = emb.mean(dim=1)               # (batch, emb_dim)
        h1 = torch.relu(self.hidden1(avg))  # (batch, hidden_dim)
        logits1 = self.output1(h1)          # (batch, vocab_size)

        if w1 is None:
            return logits1

        w1_emb = self.embedding(w1)                       # (batch, emb_dim)
        h2 = torch.relu(self.hidden2(torch.cat([h1, w1_emb], dim=-1)))
        logits2 = self.output2(h2)          # (batch, vocab_size)
        return logits1, logits2

    def log_prob_w1(self, context):
        """log P(w1 | context) for all w1."""
        logits1 = self.forward(context)
        return torch.log_softmax(logits1, dim=-1)

    def log_prob_w2(self, context, w1):
        """log P(w2 | context, w1) for all w2."""
        _, logits2 = self.forward(context, w1)
        return torch.log_softmax(logits2, dim=-1)

    def log_prob_pair(self, context, w1):
        """log P(w1, w2 | context) = log P(w1|ctx) + log P(w2|ctx,w1) for all w2.

        Args:
            context: (batch, 2k) integer tensor
            w1: (batch,) integer tensor of first word

        Returns:
            log_p_w1: (batch,) log P(w1 | context) for the given w1
            log_p_w2: (batch, vocab_size) log P(w2 | context, w1) for all w2
        """
        logits1, logits2 = self.forward(context, w1)
        log_p1_all = torch.log_softmax(logits1, dim=-1)   # (batch, V)
        log_p_w1 = log_p1_all.gather(1, w1.unsqueeze(1)).squeeze(1)  # (batch,)
        log_p_w2 = torch.log_softmax(logits2, dim=-1)     # (batch, V)
        return log_p_w1, log_p_w2


class PositionalClozeModel(nn.Module):
    """Position-aware cloze model: predict center word from context window.

    Architecture:
        context words -> shared embedding -> concatenate positions ->
        hidden1 (ReLU) -> hidden2 (ReLU) -> output (softmax)

    Unlike ClozeModel which averages context embeddings (bag-of-words),
    this model concatenates them, preserving positional information and
    allowing the hidden layers to learn joint interactions between
    context positions.
    """

    def __init__(self, vocab_size, n_context, embedding_dim=64, hidden_dim=128):
        """
        Args:
            vocab_size: number of words in vocabulary
            n_context: number of context positions (2k for window size k)
            embedding_dim: dimension of word embeddings
            hidden_dim: dimension of hidden layers
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.n_context = n_context
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        input_dim = n_context * embedding_dim
        self.hidden1 = nn.Linear(input_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)

        # Initialize weights
        nn.init.xavier_uniform_(self.embedding.weight)
        for layer in [self.hidden1, self.hidden2, self.output]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, context):
        """
        Args:
            context: (batch, 2k) integer tensor of context word indices

        Returns:
            logits: (batch, vocab_size) unnormalized log-probabilities
        """
        # (batch, 2k, embedding_dim)
        emb = self.embedding(context)
        # Concatenate all positions: (batch, 2k * embedding_dim)
        flat = emb.view(emb.size(0), -1)
        # Two hidden layers with ReLU
        h = torch.relu(self.hidden1(flat))
        h = torch.relu(self.hidden2(h))
        # Output logits: (batch, vocab_size)
        logits = self.output(h)
        return logits

    def log_probs(self, context):
        """Compute log P(w | context) for all words w."""
        logits = self.forward(context)
        return torch.log_softmax(logits, dim=-1)


class PositionalPairClozeModel(nn.Module):
    """Position-aware pair cloze model: predict two adjacent words from context.

    Uses autoregressive factorization:
        P(w1, w2 | ctx) = P(w1 | ctx) * P(w2 | ctx, w1)

    Architecture:
        context words -> shared embedding -> concatenate positions ->
        hidden1a (ReLU) -> hidden1b (ReLU) -> w1 logits
        [hidden1b ; w1_embedding] -> hidden2a (ReLU) -> hidden2b (ReLU) -> w2 logits

    Unlike PairClozeModel which averages context embeddings, this model
    concatenates them to preserve positional and joint information.
    """

    def __init__(self, vocab_size, n_context, embedding_dim=64, hidden_dim=128):
        """
        Args:
            vocab_size: number of words in vocabulary
            n_context: number of context positions (2k for window size k)
            embedding_dim: dimension of word embeddings
            hidden_dim: dimension of hidden layers
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.n_context = n_context
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        input_dim = n_context * embedding_dim

        # First word prediction: two hidden layers
        self.hidden1a = nn.Linear(input_dim, hidden_dim)
        self.hidden1b = nn.Linear(hidden_dim, hidden_dim)
        self.output1 = nn.Linear(hidden_dim, vocab_size)

        # Second word prediction: two hidden layers, conditioned on w1
        self.hidden2a = nn.Linear(hidden_dim + embedding_dim, hidden_dim)
        self.hidden2b = nn.Linear(hidden_dim, hidden_dim)
        self.output2 = nn.Linear(hidden_dim, vocab_size)

        # Initialize weights
        nn.init.xavier_uniform_(self.embedding.weight)
        for layer in [self.hidden1a, self.hidden1b, self.output1,
                      self.hidden2a, self.hidden2b, self.output2]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, context, w1=None):
        """
        Args:
            context: (batch, 2k) integer tensor of context word indices
            w1: (batch,) integer tensor of first word (for teacher forcing).
                If None, only returns logits1.

        Returns:
            logits1: (batch, vocab_size) logits for first word
            logits2: (batch, vocab_size) logits for second word (only if w1 given)
        """
        emb = self.embedding(context)            # (batch, 2k, emb_dim)
        flat = emb.view(emb.size(0), -1)         # (batch, 2k * emb_dim)
        h1 = torch.relu(self.hidden1a(flat))     # (batch, hidden_dim)
        h1 = torch.relu(self.hidden1b(h1))       # (batch, hidden_dim)
        logits1 = self.output1(h1)               # (batch, vocab_size)

        if w1 is None:
            return logits1

        w1_emb = self.embedding(w1)              # (batch, emb_dim)
        h2 = torch.relu(self.hidden2a(torch.cat([h1, w1_emb], dim=-1)))
        h2 = torch.relu(self.hidden2b(h2))       # (batch, hidden_dim)
        logits2 = self.output2(h2)               # (batch, vocab_size)
        return logits1, logits2

    def log_prob_w1(self, context):
        """log P(w1 | context) for all w1."""
        logits1 = self.forward(context)
        return torch.log_softmax(logits1, dim=-1)

    def log_prob_w2(self, context, w1):
        """log P(w2 | context, w1) for all w2."""
        _, logits2 = self.forward(context, w1)
        return torch.log_softmax(logits2, dim=-1)

    def log_prob_pair(self, context, w1):
        """log P(w1, w2 | context) = log P(w1|ctx) + log P(w2|ctx,w1) for all w2.

        Args:
            context: (batch, 2k) integer tensor
            w1: (batch,) integer tensor of first word

        Returns:
            log_p_w1: (batch,) log P(w1 | context) for the given w1
            log_p_w2: (batch, vocab_size) log P(w2 | context, w1) for all w2
        """
        logits1, logits2 = self.forward(context, w1)
        log_p1_all = torch.log_softmax(logits1, dim=-1)
        log_p_w1 = log_p1_all.gather(1, w1.unsqueeze(1)).squeeze(1)
        log_p_w2 = torch.log_softmax(logits2, dim=-1)
        return log_p_w1, log_p_w2


def train_pair_cloze_model(sentences, k=2, embedding_dim=64, hidden_dim=128,
                           n_epochs=5, batch_size=4096, lr=1e-3,
                           min_count=1, seed=42, verbose=True, device=None):
    """Train a pair cloze model on the given sentences.

    Returns:
        model: trained PairClozeModel
        word2idx: vocabulary mapping
        idx2word: reverse vocabulary mapping
        history: dict with 'epoch_loss', 'batch_losses' lists
    """
    if device is None:
        device = torch.device('mps' if torch.backends.mps.is_available()
                              else 'cuda' if torch.cuda.is_available()
                              else 'cpu')

    torch.manual_seed(seed)
    np.random.seed(seed)

    if verbose:
        print(f"Building vocabulary (min_count={min_count})...")
    word2idx, idx2word = build_vocab(sentences, min_count=min_count)
    vocab_size = len(word2idx)
    if verbose:
        print(f"  Vocabulary size: {vocab_size}")

    if verbose:
        print(f"Extracting pair contexts (k={k})...")
    t0 = time.time()
    dataset = PairClozeDataset(sentences, word2idx, k=k)
    if verbose:
        print(f"  {len(dataset)} training examples ({time.time()-t0:.1f}s)")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=0, pin_memory=False)

    model = PairClozeModel(vocab_size, embedding_dim, hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model: {n_params:,} parameters, device={device}")
        print(f"Training for {n_epochs} epochs...")

    history = {'epoch_loss': [], 'batch_losses': []}

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for ctx_batch, tgt1_batch, tgt2_batch in dataloader:
            ctx_batch = ctx_batch.to(device)
            tgt1_batch = tgt1_batch.to(device)
            tgt2_batch = tgt2_batch.to(device)

            optimizer.zero_grad()
            logits1, logits2 = model(ctx_batch, tgt1_batch)
            loss = criterion(logits1, tgt1_batch) + criterion(logits2, tgt2_batch)
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            total_loss += batch_loss
            n_batches += 1
            history['batch_losses'].append(batch_loss)

        avg_loss = total_loss / n_batches
        history['epoch_loss'].append(avg_loss)
        elapsed = time.time() - t0

        if verbose:
            ppl = np.exp(avg_loss / 2)  # per-word perplexity (loss is sum of 2)
            print(f"  Epoch {epoch+1}/{n_epochs}: "
                  f"loss={avg_loss:.4f}, ppl/word={ppl:.1f}, "
                  f"time={elapsed:.1f}s")

    model.eval()
    return model, word2idx, idx2word, history


def build_vocab(sentences, min_count=1):
    """Build vocabulary from sentences.

    Args:
        sentences: list of lists of tokens
        min_count: minimum token frequency to include

    Returns:
        word2idx: dict mapping token -> index
        idx2word: list mapping index -> token
    """
    counts = {}
    for sent in sentences:
        for w in sent:
            counts[w] = counts.get(w, 0) + 1

    # Always include boundary token at index 0
    idx2word = [BOUNDARY]
    word2idx = {BOUNDARY: 0}

    for w, c in sorted(counts.items(), key=lambda x: -x[1]):
        if c >= min_count and w not in word2idx:
            word2idx[w] = len(idx2word)
            idx2word.append(w)

    return word2idx, idx2word


def train_cloze_model(sentences, k=2, embedding_dim=64, hidden_dim=128,
                      n_epochs=5, batch_size=4096, lr=1e-3,
                      min_count=1, seed=42, verbose=True, device=None):
    """Train a cloze model on the given sentences.

    Args:
        sentences: list of lists of tokens (already tokenized)
        k: context window size on each side
        embedding_dim: dimension of word embeddings
        hidden_dim: dimension of hidden layer
        n_epochs: number of training epochs
        batch_size: mini-batch size
        lr: learning rate
        min_count: minimum word count for vocabulary
        seed: random seed
        verbose: print training progress
        device: torch device (None = auto-detect)

    Returns:
        model: trained ClozeModel
        word2idx: vocabulary mapping
        idx2word: reverse vocabulary mapping
        history: dict with 'epoch_loss', 'batch_losses' lists
    """
    if device is None:
        device = torch.device('mps' if torch.backends.mps.is_available()
                              else 'cuda' if torch.cuda.is_available()
                              else 'cpu')

    torch.manual_seed(seed)
    np.random.seed(seed)

    if verbose:
        print(f"Building vocabulary (min_count={min_count})...")
    word2idx, idx2word = build_vocab(sentences, min_count=min_count)
    vocab_size = len(word2idx)
    if verbose:
        print(f"  Vocabulary size: {vocab_size}")

    if verbose:
        print(f"Extracting contexts (k={k})...")
    t0 = time.time()
    dataset = ClozeDataset(sentences, word2idx, k=k)
    if verbose:
        print(f"  {len(dataset)} training examples ({time.time()-t0:.1f}s)")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=0, pin_memory=False)

    model = ClozeModel(vocab_size, embedding_dim, hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model: {n_params:,} parameters, device={device}")
        print(f"Training for {n_epochs} epochs...")

    history = {'epoch_loss': [], 'batch_losses': []}

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for ctx_batch, tgt_batch in dataloader:
            ctx_batch = ctx_batch.to(device)
            tgt_batch = tgt_batch.to(device)

            optimizer.zero_grad()
            logits = model(ctx_batch)
            loss = criterion(logits, tgt_batch)
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            total_loss += batch_loss
            n_batches += 1
            history['batch_losses'].append(batch_loss)

        avg_loss = total_loss / n_batches
        history['epoch_loss'].append(avg_loss)
        elapsed = time.time() - t0

        if verbose:
            # Compute perplexity
            ppl = np.exp(avg_loss)
            print(f"  Epoch {epoch+1}/{n_epochs}: "
                  f"loss={avg_loss:.4f}, ppl={ppl:.1f}, "
                  f"time={elapsed:.1f}s")

    model.eval()
    return model, word2idx, idx2word, history


def train_positional_cloze_model(sentences, k=2, embedding_dim=64, hidden_dim=128,
                                 n_epochs=5, batch_size=4096, lr=1e-3,
                                 min_count=1, seed=42, verbose=True, device=None):
    """Train a positional cloze model on the given sentences.

    Same interface as train_cloze_model but returns a PositionalClozeModel.
    """
    if device is None:
        device = torch.device('mps' if torch.backends.mps.is_available()
                              else 'cuda' if torch.cuda.is_available()
                              else 'cpu')

    torch.manual_seed(seed)
    np.random.seed(seed)

    if verbose:
        print(f"Building vocabulary (min_count={min_count})...")
    word2idx, idx2word = build_vocab(sentences, min_count=min_count)
    vocab_size = len(word2idx)
    if verbose:
        print(f"  Vocabulary size: {vocab_size}")

    if verbose:
        print(f"Extracting contexts (k={k})...")
    t0 = time.time()
    dataset = ClozeDataset(sentences, word2idx, k=k)
    if verbose:
        print(f"  {len(dataset)} training examples ({time.time()-t0:.1f}s)")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=0, pin_memory=False)

    n_context = 2 * k
    model = PositionalClozeModel(vocab_size, n_context, embedding_dim, hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"PositionalClozeModel: {n_params:,} parameters, device={device}")
        print(f"  input_dim={n_context}*{embedding_dim}={n_context*embedding_dim}, "
              f"hidden_dim={hidden_dim}")
        print(f"Training for {n_epochs} epochs...")

    history = {'epoch_loss': [], 'batch_losses': []}

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for ctx_batch, tgt_batch in dataloader:
            ctx_batch = ctx_batch.to(device)
            tgt_batch = tgt_batch.to(device)

            optimizer.zero_grad()
            logits = model(ctx_batch)
            loss = criterion(logits, tgt_batch)
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            total_loss += batch_loss
            n_batches += 1
            history['batch_losses'].append(batch_loss)

        avg_loss = total_loss / n_batches
        history['epoch_loss'].append(avg_loss)
        elapsed = time.time() - t0

        if verbose:
            ppl = np.exp(avg_loss)
            print(f"  Epoch {epoch+1}/{n_epochs}: "
                  f"loss={avg_loss:.4f}, ppl={ppl:.1f}, "
                  f"time={elapsed:.1f}s")

    model.eval()
    return model, word2idx, idx2word, history


def train_positional_pair_cloze_model(sentences, k=2, embedding_dim=64, hidden_dim=128,
                                       n_epochs=5, batch_size=4096, lr=1e-3,
                                       min_count=1, seed=42, verbose=True, device=None):
    """Train a positional pair cloze model on the given sentences.

    Same interface as train_pair_cloze_model but returns a PositionalPairClozeModel.
    """
    if device is None:
        device = torch.device('mps' if torch.backends.mps.is_available()
                              else 'cuda' if torch.cuda.is_available()
                              else 'cpu')

    torch.manual_seed(seed)
    np.random.seed(seed)

    if verbose:
        print(f"Building vocabulary (min_count={min_count})...")
    word2idx, idx2word = build_vocab(sentences, min_count=min_count)
    vocab_size = len(word2idx)
    if verbose:
        print(f"  Vocabulary size: {vocab_size}")

    if verbose:
        print(f"Extracting pair contexts (k={k})...")
    t0 = time.time()
    dataset = PairClozeDataset(sentences, word2idx, k=k)
    if verbose:
        print(f"  {len(dataset)} training examples ({time.time()-t0:.1f}s)")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=0, pin_memory=False)

    n_context = 2 * k
    model = PositionalPairClozeModel(vocab_size, n_context, embedding_dim, hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"PositionalPairClozeModel: {n_params:,} parameters, device={device}")
        print(f"  input_dim={n_context}*{embedding_dim}={n_context*embedding_dim}, "
              f"hidden_dim={hidden_dim}")
        print(f"Training for {n_epochs} epochs...")

    history = {'epoch_loss': [], 'batch_losses': []}

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for ctx_batch, tgt1_batch, tgt2_batch in dataloader:
            ctx_batch = ctx_batch.to(device)
            tgt1_batch = tgt1_batch.to(device)
            tgt2_batch = tgt2_batch.to(device)

            optimizer.zero_grad()
            logits1, logits2 = model(ctx_batch, tgt1_batch)
            loss = criterion(logits1, tgt1_batch) + criterion(logits2, tgt2_batch)
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            total_loss += batch_loss
            n_batches += 1
            history['batch_losses'].append(batch_loss)

        avg_loss = total_loss / n_batches
        history['epoch_loss'].append(avg_loss)
        elapsed = time.time() - t0

        if verbose:
            ppl = np.exp(avg_loss / 2)
            print(f"  Epoch {epoch+1}/{n_epochs}: "
                  f"loss={avg_loss:.4f}, ppl/word={ppl:.1f}, "
                  f"time={elapsed:.1f}s")

    model.eval()
    return model, word2idx, idx2word, history


# ============================================================
# RNN-based cloze model with variable-length contexts
# ============================================================

class RNNClozeDataset(Dataset):
    """Dataset of (left_context, right_context, target) for RNN cloze model.

    For each word w_i in the corpus, the left context is the sequence
    from BOUNDARY through w_{i-1}, and the right context is from
    w_{i+1+gap} through BOUNDARY.  Contexts are variable length.

    Args:
        sentences: list of word lists
        word2idx: vocabulary mapping
        gap: number of extra words to skip on the right side (default 0).
             gap=0 is the normal single-word cloze model.
             gap=1 skips one word on the right, for pair prediction.

    Uses flat arrays with offsets for memory efficiency.
    """

    def __init__(self, sentences, word2idx, gap=0):
        bdy_idx = word2idx[BOUNDARY]
        self.gap = gap

        # First pass: count total examples and total context tokens
        n_examples = sum(len(s) for s in sentences)

        targets = np.empty(n_examples, dtype=np.int64)
        left_offsets = np.empty(n_examples + 1, dtype=np.int64)
        right_offsets = np.empty(n_examples + 1, dtype=np.int64)

        # Compute offsets
        # Left context: [bdy] + ids[:i], length = i + 1
        # Right context: ids[i+1+gap:] + [bdy], length = max(1, n - i - gap)
        left_offsets[0] = 0
        right_offsets[0] = 0
        idx = 0
        for sent in sentences:
            n = len(sent)
            for i in range(n):
                left_offsets[idx + 1] = left_offsets[idx] + (i + 1)
                r_len = max(1, n - i - gap)
                right_offsets[idx + 1] = right_offsets[idx] + r_len
                idx += 1

        total_left = int(left_offsets[n_examples])
        total_right = int(right_offsets[n_examples])

        left_data = np.empty(total_left, dtype=np.int64)
        right_data = np.empty(total_right, dtype=np.int64)

        # Second pass: fill data
        idx = 0
        for sent in sentences:
            ids = [word2idx.get(w, 0) for w in sent]
            n = len(ids)
            for i in range(n):
                # Left context: [bdy] + ids[:i]
                lo = int(left_offsets[idx])
                left_data[lo] = bdy_idx
                for j in range(i):
                    left_data[lo + 1 + j] = ids[j]

                # Right context: ids[i+1+gap:] + [bdy]
                ro = int(right_offsets[idx])
                r_start = i + 1 + gap
                r_count = 0
                for j in range(r_start, n):
                    right_data[ro + r_count] = ids[j]
                    r_count += 1
                right_data[ro + r_count] = bdy_idx

                targets[idx] = ids[i]
                idx += 1

        self.targets = targets
        self.left_data = left_data
        self.right_data = right_data
        self.left_offsets = left_offsets
        self.right_offsets = right_offsets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        lo = int(self.left_offsets[idx])
        lend = int(self.left_offsets[idx + 1])
        ro = int(self.right_offsets[idx])
        rend = int(self.right_offsets[idx + 1])
        return (torch.from_numpy(self.left_data[lo:lend]),
                torch.from_numpy(self.right_data[ro:rend]),
                self.targets[idx])


def rnn_collate_fn(batch):
    """Collate variable-length left/right contexts into padded batches."""
    lefts, rights, targets = zip(*batch)
    left_lens = [len(l) for l in lefts]
    right_lens = [len(r) for r in rights]
    max_left = max(left_lens)
    max_right = max(right_lens)

    # Pad with zeros (will be masked by pack_padded_sequence)
    left_padded = torch.zeros(len(lefts), max_left, dtype=torch.long)
    right_padded = torch.zeros(len(rights), max_right, dtype=torch.long)
    for i, (l, r) in enumerate(zip(lefts, rights)):
        left_padded[i, :len(l)] = l
        right_padded[i, :len(r)] = r

    targets = torch.tensor(targets, dtype=torch.long)
    left_lens = torch.tensor(left_lens, dtype=torch.long)
    right_lens = torch.tensor(right_lens, dtype=torch.long)
    return left_padded, right_padded, left_lens, right_lens, targets


class SentenceRNNDataset(Dataset):
    """Dataset of whole sentences as token ID sequences.

    Stores one entry per sentence rather than one per position,
    enabling the model to share GRU computation across all positions
    in a sentence (O(n) instead of O(n^2) per sentence).
    """

    def __init__(self, sentences, word2idx):
        self.data = []
        self.total_tokens = 0
        for sent in sentences:
            ids = np.array([word2idx.get(w, 0) for w in sent], dtype=np.int64)
            self.data.append(ids)
            self.total_tokens += len(ids)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx])


def sentence_collate_fn(batch):
    """Collate variable-length sentences into a padded batch."""
    lengths = torch.tensor([len(s) for s in batch], dtype=torch.long)
    max_len = lengths.max().item()
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, s in enumerate(batch):
        padded[i, :len(s)] = s
    return padded, lengths


class RNNClozeModel(nn.Module):
    """Bidirectional RNN cloze model with variable-length contexts.

    Architecture:
        left context  -> GRU (left-to-right)  -> h_L
        right context -> GRU (right-to-left)  -> h_R
        [h_L, h_R] -> hidden (ReLU) -> output (softmax)

    Shared embedding between both directions.
    """

    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128,
                 gru_dim=128, n_gru_layers=1, gap=0):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.gru_dim = gru_dim
        self.n_gru_layers = n_gru_layers
        self.gap = gap  # 0=normal, 1=gap model for pair prediction

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.left_gru = nn.GRU(embedding_dim, gru_dim,
                               num_layers=n_gru_layers, batch_first=True)
        self.right_gru = nn.GRU(embedding_dim, gru_dim,
                                num_layers=n_gru_layers, batch_first=True)
        self.hidden = nn.Linear(2 * gru_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)

        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.hidden.weight)
        nn.init.zeros_(self.hidden.bias)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward_unpacked(self, left_ctx, right_ctx):
        """Forward pass without packing (for inference with small batches).

        Args:
            left_ctx:  (batch, L) int tensor — left context, left-to-right
            right_ctx: (batch, R) int tensor — right context, left-to-right

        Returns:
            logits: (batch, vocab_size)
        """
        left_emb = self.embedding(left_ctx)
        right_emb = self.embedding(right_ctx)

        # Reverse right context for right-to-left processing
        right_emb_rev = torch.flip(right_emb, [1])

        _, h_L = self.left_gru(left_emb)    # h_L: (n_layers, batch, gru_dim)
        _, h_R = self.right_gru(right_emb_rev)  # h_R: same

        # Take final layer hidden state
        h_L = h_L[-1]  # (batch, gru_dim)
        h_R = h_R[-1]

        combined = torch.cat([h_L, h_R], dim=-1)  # (batch, 2*gru_dim)
        h = torch.relu(self.hidden(combined))
        return self.output(h)

    def forward_sentence(self, sentences, lengths):
        """Forward pass on whole sentences, sharing GRU computation.

        Instead of processing each position independently (O(n^2) GRU
        steps per sentence of length n), runs one left-to-right and one
        right-to-left GRU pass per sentence (O(n) total).

        Args:
            sentences: (batch, max_len) int tensor of token IDs
            lengths: (batch,) int tensor of sentence lengths

        Returns:
            logits: (N_valid, vocab_size) predictions at valid positions
            targets: (N_valid,) ground truth token IDs
        """
        batch_size, max_len = sentences.shape
        device = sentences.device
        gap = self.gap

        # Left input: [BDY, w_0, ..., w_{n-2}], same length as sentence.
        # GRU output[i] = hidden after processing [BDY, w_0, ..., w_{i-1}].
        left_input = torch.zeros_like(sentences)
        # BDY is always index 0 in our vocabulary
        if max_len > 1:
            left_input[:, 1:] = sentences[:, :-1]

        # Right input: reverse sentence, prepend BDY, length = n - gap.
        # Full reversed seq = [BDY, w_{n-1}, ..., w_{1+gap}].
        # GRU output[j] = hidden after [BDY, w_{n-1}, ..., w_{n-j}].
        # Position i uses output[max(0, n-1-gap-i)].
        idx = torch.arange(max_len, device=device).unsqueeze(0)
        rev_idx = (lengths.unsqueeze(1) - 1 - idx).clamp(min=0)
        reversed_sents = sentences.gather(1, rev_idx.expand(batch_size, -1))

        right_input = torch.zeros(batch_size, max_len, dtype=torch.long,
                                  device=device)
        if max_len > 1:
            right_input[:, 1:] = reversed_sents[:, :max_len - 1]
        right_lengths = (lengths - gap).clamp(min=1)

        # Embed and run GRUs
        left_emb = self.embedding(left_input)
        right_emb = self.embedding(right_input)

        left_packed = nn.utils.rnn.pack_padded_sequence(
            left_emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        left_out, _ = self.left_gru(left_packed)
        left_out, _ = nn.utils.rnn.pad_packed_sequence(
            left_out, batch_first=True)

        right_packed = nn.utils.rnn.pack_padded_sequence(
            right_emb, right_lengths.cpu(), batch_first=True,
            enforce_sorted=False)
        right_out, _ = self.right_gru(right_packed)
        right_out, _ = nn.utils.rnn.pad_packed_sequence(
            right_out, batch_first=True)

        # Gather right hidden state for each position
        right_idx = (lengths.unsqueeze(1) - 1 - gap - idx).clamp(
            min=0, max=right_out.size(1) - 1)
        right_h = right_out.gather(
            1, right_idx.expand(batch_size, -1).unsqueeze(-1).expand(
                -1, -1, self.gru_dim))

        combined = torch.cat([left_out, right_h], dim=-1)
        h = torch.relu(self.hidden(combined))
        logits = self.output(h)

        # Select valid positions only
        mask = idx.squeeze(0) < lengths.unsqueeze(1)
        return logits[mask], sentences[mask]

    def forward(self, left_padded, right_padded, left_lens, right_lens):
        """Forward pass with packed sequences for efficient training.

        Args:
            left_padded:  (batch, max_L) int tensor
            right_padded: (batch, max_R) int tensor
            left_lens:  (batch,) int tensor of actual lengths
            right_lens: (batch,) int tensor of actual lengths

        Returns:
            logits: (batch, vocab_size)
        """
        left_emb = self.embedding(left_padded)
        right_emb = self.embedding(right_padded)

        # Reverse right context for right-to-left processing
        # Use index_select for vectorized reversal of valid portions
        batch_size, max_R, edim = right_emb.shape
        # Build reversed indices: for each row, reverse [0..rlen-1] within padding
        idx = torch.arange(max_R, device=right_emb.device).unsqueeze(0).expand(batch_size, -1)
        rlens = right_lens.unsqueeze(1)  # (batch, 1)
        # reversed index: rlen-1-i for i < rlen, else 0 (padding, will be masked)
        rev_idx = (rlens - 1 - idx).clamp(min=0)
        right_emb_rev = right_emb.gather(1, rev_idx.unsqueeze(-1).expand(-1, -1, edim))

        # Pack sequences
        left_packed = nn.utils.rnn.pack_padded_sequence(
            left_emb, left_lens.cpu(), batch_first=True, enforce_sorted=False)
        right_packed = nn.utils.rnn.pack_padded_sequence(
            right_emb_rev, right_lens.cpu(), batch_first=True, enforce_sorted=False)

        _, h_L = self.left_gru(left_packed)
        _, h_R = self.right_gru(right_packed)

        h_L = h_L[-1]  # (batch, gru_dim)
        h_R = h_R[-1]

        combined = torch.cat([h_L, h_R], dim=-1)
        h = torch.relu(self.hidden(combined))
        return self.output(h)


def train_rnn_cloze_model(sentences, embedding_dim=64, hidden_dim=128,
                          gru_dim=128, n_gru_layers=1,
                          n_epochs=5, batch_size=512, lr=1e-3,
                          min_count=1, seed=42, verbose=True, device=None,
                          gap=0, word2idx=None, idx2word=None):
    """Train an RNN cloze model on the given sentences.

    Args:
        gap: number of words to skip on the right side of context.
             gap=0 is the normal model; gap=1 is the gap model for
             pair prediction.
        word2idx, idx2word: if provided, reuse this vocabulary instead
             of building a new one (useful when training gap model to
             share vocab with normal model).

    Returns:
        model, word2idx, idx2word, history
    """
    if device is None:
        device = torch.device('mps' if torch.backends.mps.is_available()
                              else 'cuda' if torch.cuda.is_available()
                              else 'cpu')

    torch.manual_seed(seed)
    np.random.seed(seed)

    if word2idx is None:
        if verbose:
            print(f"Building vocabulary (min_count={min_count})...", flush=True)
        word2idx, idx2word = build_vocab(sentences, min_count=min_count)
    vocab_size = len(word2idx)
    if verbose:
        print(f"  Vocabulary size: {vocab_size}", flush=True)

    gap_label = f" (gap={gap})" if gap > 0 else ""
    if verbose:
        print(f"Building sentence dataset{gap_label}...", flush=True)
    t0 = time.time()
    dataset = SentenceRNNDataset(sentences, word2idx)
    if verbose:
        print(f"  {len(dataset)} sentences, {dataset.total_tokens} tokens "
              f"({time.time()-t0:.1f}s)", flush=True)

    use_pin = (device.type == 'cuda')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=0, pin_memory=use_pin,
                            collate_fn=sentence_collate_fn)

    model = RNNClozeModel(vocab_size, embedding_dim, hidden_dim,
                          gru_dim, n_gru_layers, gap=gap).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        label = "RNNClozeModel" + (f" [gap={gap}]" if gap > 0 else "")
        print(f"{label}: {n_params:,} parameters, device={device}", flush=True)
        print(f"  embedding={embedding_dim}, gru={gru_dim}x{n_gru_layers}, "
              f"hidden={hidden_dim}", flush=True)
        print(f"Training for {n_epochs} epochs...", flush=True)

    history = {'epoch_loss': [], 'batch_losses': []}

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        t0 = time.time()

        for sents, lens in dataloader:
            sents = sents.to(device)
            lens = lens.to(device)

            optimizer.zero_grad()
            logits, targets = model.forward_sentence(sents, lens)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * targets.size(0)
            total_tokens += targets.size(0)
            history['batch_losses'].append(loss.item())

        avg_loss = total_loss / total_tokens
        history['epoch_loss'].append(avg_loss)
        elapsed = time.time() - t0

        if verbose:
            ppl = np.exp(avg_loss)
            print(f"  Epoch {epoch+1}/{n_epochs}: "
                  f"loss={avg_loss:.4f}, ppl={ppl:.1f}, "
                  f"time={elapsed:.1f}s", flush=True)

    model.eval()
    return model, word2idx, idx2word, history


def save_model(model, word2idx, idx2word, path, k=2, history=None):
    """Save trained model and vocabulary to disk.

    Args:
        model: trained ClozeModel or PairClozeModel
        word2idx: vocabulary mapping
        idx2word: reverse vocabulary mapping
        path: file path to save to (.pt)
        k: context window size used during training
        history: training history dict (optional)
    """
    if isinstance(model, RNNClozeModel):
        model_type = 'rnn_single'
    elif isinstance(model, PositionalPairClozeModel):
        model_type = 'positional_pair'
    elif isinstance(model, PositionalClozeModel):
        model_type = 'positional_single'
    elif isinstance(model, PairClozeModel):
        model_type = 'pair'
    else:
        model_type = 'single'

    checkpoint = {
        'model_type': model_type,
        'model_state_dict': model.state_dict(),
        'vocab_size': model.vocab_size,
        'embedding_dim': model.embedding_dim,
        'hidden_dim': model.hidden_dim,
        'word2idx': word2idx,
        'idx2word': idx2word,
        'k': k,
    }
    if hasattr(model, 'n_context'):
        checkpoint['n_context'] = model.n_context
    if hasattr(model, 'gru_dim'):
        checkpoint['gru_dim'] = model.gru_dim
        checkpoint['n_gru_layers'] = model.n_gru_layers
    if hasattr(model, 'gap'):
        checkpoint['gap'] = model.gap
    if history is not None:
        checkpoint['history'] = history
    torch.save(checkpoint, path)


def load_model(path, device=None):
    """Load a trained model from disk.

    Args:
        path: file path to load from (.pt)
        device: torch device (None = auto-detect)

    Returns:
        model: ClozeModel or PairClozeModel (eval mode)
        word2idx: vocabulary mapping
        idx2word: reverse vocabulary mapping
        k: context window size
        history: training history (or None)
    """
    if device is None:
        device = torch.device('mps' if torch.backends.mps.is_available()
                              else 'cuda' if torch.cuda.is_available()
                              else 'cpu')

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model_type = checkpoint.get('model_type', 'single')

    if model_type == 'rnn_single':
        model = RNNClozeModel(
            checkpoint['vocab_size'],
            checkpoint['embedding_dim'],
            checkpoint['hidden_dim'],
            checkpoint.get('gru_dim', 128),
            checkpoint.get('n_gru_layers', 1),
            gap=checkpoint.get('gap', 0),
        ).to(device)
    elif model_type == 'positional_pair':
        model = PositionalPairClozeModel(
            checkpoint['vocab_size'],
            checkpoint['n_context'],
            checkpoint['embedding_dim'],
            checkpoint['hidden_dim'],
        ).to(device)
    elif model_type == 'positional_single':
        model = PositionalClozeModel(
            checkpoint['vocab_size'],
            checkpoint['n_context'],
            checkpoint['embedding_dim'],
            checkpoint['hidden_dim'],
        ).to(device)
    elif model_type == 'pair':
        model = PairClozeModel(
            checkpoint['vocab_size'],
            checkpoint['embedding_dim'],
            checkpoint['hidden_dim'],
        ).to(device)
    else:
        model = ClozeModel(
            checkpoint['vocab_size'],
            checkpoint['embedding_dim'],
            checkpoint['hidden_dim'],
        ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return (model, checkpoint['word2idx'], checkpoint['idx2word'],
            checkpoint['k'], checkpoint.get('history'))


def load_sentences(corpus_path):
    """Load sentences from a corpus file (one sentence per line)."""
    sentences = []
    with open(corpus_path) as f:
        for line in f:
            tokens = line.strip().split()
            if tokens:
                sentences.append(tokens)
    return sentences


def save_kernels(kernels, path, metadata=None):
    """Save kernel list to a simple text file.

    Format:
        # key=value metadata lines
        S
        word1
        word2
        ...
    """
    with open(path, 'w') as f:
        if metadata:
            for k, v in metadata.items():
                f.write(f"# {k}={v}\n")
        for w in kernels:
            f.write(w + '\n')


def load_kernels(path):
    """Load kernel list from file.

    Returns:
        (kernels, metadata) where kernels is a list of strings
        and metadata is a dict of key-value pairs from comment lines.
    """
    kernels = []
    metadata = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'):
                parts = line[1:].strip().split('=', 1)
                if len(parts) == 2:
                    metadata[parts[0].strip()] = parts[1].strip()
            elif line:
                kernels.append(line)
    return kernels, metadata


# ============================================================
# Command-line interface for training
# ============================================================

def main():
    import argparse
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(
        description='Train a neural cloze model on a corpus')
    parser.add_argument('corpus', help='Path to corpus file')
    parser.add_argument('output', help='Path to save model (.pt)')
    parser.add_argument('--k', type=int, default=2,
                        help='Context window size on each side (default: 2)')
    parser.add_argument('--embedding-dim', type=int, default=64,
                        help='Embedding dimension (default: 64)')
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='Hidden layer dimension (default: 128)')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs (default: 5)')
    parser.add_argument('--batch-size', type=int, default=4096,
                        help='Batch size (default: 4096)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--min-count', type=int, default=1,
                        help='Min word count for vocabulary (default: 1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--plot', default=None,
                        help='Path to save loss curve plot (optional)')
    args = parser.parse_args()

    print(f"Loading corpus from {args.corpus}...")
    sentences = load_sentences(args.corpus)
    print(f"  {len(sentences)} sentences, "
          f"{sum(len(s) for s in sentences)} tokens")

    model, word2idx, idx2word, history = train_cloze_model(
        sentences, k=args.k,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        min_count=args.min_count,
        seed=args.seed,
    )

    save_model(model, word2idx, idx2word, args.output,
               k=args.k, history=history)
    print(f"\nModel saved to {args.output}")

    if args.plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Epoch loss
        ax = axes[0]
        epochs = range(1, len(history['epoch_loss']) + 1)
        ax.plot(epochs, history['epoch_loss'], 'o-', color='#2196F3')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (cross-entropy)')
        ax.set_title('Epoch-average loss')
        ax.grid(True, alpha=0.3)

        # Batch loss (smoothed)
        ax = axes[1]
        bl = history['batch_losses']
        # Smooth with running average
        window = max(1, len(bl) // 200)
        if window > 1:
            smoothed = np.convolve(bl, np.ones(window)/window, mode='valid')
            ax.plot(smoothed, color='#FF9800', linewidth=0.5)
        else:
            ax.plot(bl, color='#FF9800', linewidth=0.5)
        ax.set_xlabel('Batch')
        ax.set_ylabel('Loss (cross-entropy)')
        ax.set_title('Per-batch loss (smoothed)')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(args.plot, dpi=150, bbox_inches='tight')
        print(f"Loss curves saved to {args.plot}")


if __name__ == '__main__':
    main()
