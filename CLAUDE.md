# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LocalLearner is a research implementation for reproducing experiments from "Consistent unsupervised estimators for anchored PCFGs" (TACL, 2020) by Alexander Clark and Nathanael Fijalkow. It learns probabilistic context-free grammars (PCFGs) from unlabeled text using anchor word methods. The project includes both the original NMF-based approach and a newer neural pipeline that uses cloze models (bag-of-words, positional, and bidirectional RNN) for anchor word discovery.

## Build Commands

### C/C++ Binaries (required for full pipeline)
```bash
cd lib/
make io          # Primary Inside-Outside algorithm binary
make ioh         # Alternative IO variant
make bayes       # Bayesian variant
make clean       # Clean build artifacts
```

### Full Experiment Pipeline
```bash
cd scripts/
make -j 50 json  # Parallel data generation and evaluation
make thin        # Remove large intermediate files only
make clean       # Remove all intermediate files
```

### Neural Experiment Pipeline
```bash
cd scripts/
make -f Makefile.neural corpus    # Sample training corpus
make -f Makefile.neural models    # Train cloze models (RNN + positional)
make -f Makefile.neural kernels   # Find kernels from trained models
make -f Makefile.neural all       # Full neural pipeline
```

## Running Tests

```bash
pytest tests/                     # Run all tests
pytest tests/ -x                  # Stop on first failure
pytest tests/test_wcfg_basics.py  # Run a specific test file
```

Test suite covers: WCFG basics/parsing/sampling/algorithms, LocalLearner, NMF, NMF autodetect, neural learner, neural baseline, clustering, KL divergence, parameter roundtrips, and integration tests.

## Running Python Scripts

No installation required - scripts run directly from the `locallearner/` directory:

```bash
# Learn grammar from corpus (NMF-based)
python3 locallearner/run_quick_learner.py \
  --nonterminals 3 --seed 1 --number_clusters 10 --min_count_nmf 100 \
  input_corpus.txt output_grammar.pcfg

# Extract kernels only
python3 locallearner/run_find_kernels.py \
  --nonterminals 3 --cheat target_grammar.pcfg \
  corpus.txt kernels.json

# Neural pipeline (train cloze models, find kernels, estimate params, evaluate)
python3 locallearner/run_neural_pipeline.py \
  --nonterminals 3 --model_type rnn --epochs 20 \
  target_grammar.pcfg

# Neural pipeline with gold (oracle) kernels
python3 locallearner/run_gold_pipeline.py target_grammar.pcfg

# Train neural baseline LSTM language model
python3 locallearner/run_neural_baseline.py target_grammar.pcfg

# SGD-based Inside-Outside training
python3 locallearner/run_sgd_io.py grammar.pcfg corpus.txt

# Parameter sweeps
python3 locallearner/run_sweep.py target_grammar.pcfg
python3 locallearner/run_rnn_sweep.py target_grammar.pcfg
python3 locallearner/run_rnn_alpha_sweep.py target_grammar.pcfg

# Evaluate grammar against gold standard
python3 locallearner/evaluate_pcfg.py \
  --target target_grammar.pcfg --maxlength 20 --json results.json \
  hypothesis.pcfg

# Format conversions
python3 locallearner/convert_wcfg_to_pcfg.py input.wcfg output.pcfg
python3 locallearner/convert_trees_to_yields.py trees.txt yields.txt
python3 locallearner/convert_trees_to_pcfg.py --length 10 trees.txt grammar.pcfg
python3 locallearner/convert_pcfg_to_mjio.py grammar.pcfg output.mjio
python3 locallearner/convert_mjio_to_pcfg.py input.mjio output.pcfg
```

## Architecture

### Core Learning Flow (locallearner/locallearner.py::LocalLearner)
1. Load corpus and compute lexical statistics
2. Cluster words using Neyessen algorithm (`neyessen.py`)
3. Compute distributional features from context windows
4. Apply NMF (`nmf.py`) to find anchor words (kernels)
5. Estimate parameters via Frank-Wolfe algorithm
6. Build and normalize WCFG to proper PCFG

### Neural Learning Flow (locallearner/neural_learner.py::NeuralLearner)
1. Sample corpus from target grammar
2. Train cloze models on corpus (predict masked word from context)
3. Compute Rényi divergences between word context distributions
4. Find anchor words (kernels) using divergence-based selection
5. Estimate grammar parameters from kernel distributions
6. Optionally refine with SGD via Inside-Outside

### Key Modules
- `locallearner.py` - Core LocalLearner class (NMF-based anchor learning)
- `neural_learner.py` - NeuralLearner class (neural cloze model-based learning)
- `neural_features.py` - PyTorch cloze models: ClozeModel, PairClozeModel, PositionalClozeModel, PositionalPairClozeModel, RNNClozeModel
- `neural_baseline.py` - LSTM language model baseline for comparison
- `wcfg.py` - Weighted Context-Free Grammar data structure and operations
- `evaluation.py` - Grammar evaluation (KL divergence, Parseval metrics, alignment)
- `nmf.py` - Non-negative matrix factorization for kernel extraction
- `neyessen.py` - Word clustering algorithm
- `utility.py` - Tree/parse utilities
- `sgt.py` - Simple Good Turing smoothing
- `gold_kernels.py` - Oracle kernel finding via Hungarian algorithm

### C/C++ Components (lib/)
- `io.c` - Inside-Outside algorithm (performance-critical E-M)
- `ioh.c` - Alternative IO variant
- `bayes.cc` - Bayesian IO variant with Dirichlet priors
- `grammar.c/h` - Grammar structures
- `tree.c/h` - Parse tree structures
- `expected-counts.c/h` - Expected count computation for E-M
- `digamma.c/h` - Digamma function for Bayesian inference

## File Formats

**Grammar (.pcfg, .wcfg)**: Production per line with probability
```
S -> NT_a NT_b  0.5
NT_a -> word    0.3
```

**MJIO (.mjio)**: Alternate grammar format consumed by C binaries (convert with `convert_pcfg_to_mjio.py`)

**Corpus (.yld, .txt)**: One sentence per line, space-separated tokens

**Trees (.trees)**: S-expression format `(S (NT a) (NT (NT b) (NT c)))`

**Kernels (.json)**: JSON list of anchor words per nonterminal

## Key Hyperparameters

### LocalLearner.__init__
- `nonterminals`: Number of grammar nonterminals
- `number_clusters`: Neyessen clustering parameter
- `min_count_nmf`: Minimum word frequency for anchor selection
- `width`: Context window width
- `renyi`: Rényi divergence alpha parameter for binary rule estimation
- `ssf`: Small sample factor (penalizes rare words in NMF)
- `feature_mode`: `'marginal'` or `'joint'` context representation
- `nmf_ratio_threshold`: Chi-squared ratio threshold for NMF auto-stopping
- `nmf_min_divergence`: Minimum KL divergence between kernels
- `binary_smoothing`, `unary_smoothing`: Grammar smoothing parameters

### NeuralLearner
- `model_type`: `'bow'`, `'positional'`, `'rnn'` (cloze model architecture)
- `alpha`: Rényi divergence alpha parameter
- `k`: Context window size for positional models
- `epochs`: Training epochs for cloze models
- `hidden_dim`, `embedding_dim`: Neural model dimensions

## Dependencies

Install Python dependencies:
```bash
pip install -e .           # Install package with dependencies
pip install -e ".[dev]"    # Include dev dependencies (pytest)
pip install -e ".[neural]" # Include PyTorch for neural pipeline
```

- Core: numpy>=1.18, scipy>=1.5, matplotlib>=3.0
- Neural (optional): torch>=2.0
- Dev: pytest>=7.0, pytest-cov>=4.0
- C/C++: gcc with -O3 optimization
- Python: >=3.8

## Related Repositories
- syntheticpcfg: https://github.com/alexc17/syntheticpcfg.git
- testpcfg: https://github.com/alexc17/testpcfg.git
