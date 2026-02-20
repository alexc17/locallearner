#!/usr/bin/env python3
"""Train cloze models for a grammar directory.

Trains single-word and pair/gap cloze models, saving them to the
grammar directory.  Skips training if the model file already exists.

Usage:
    python train_cloze_models.py <grammar_dir> --config config.json
    python train_cloze_models.py <grammar_dir> --model_type rnn
    python train_cloze_models.py <grammar_dir> --model_type positional --k 2
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(line_buffering=True)

import neural_learner as nl_module


def load_config(config_path):
    """Load experiment config, returning training params as a flat dict."""
    with open(config_path) as f:
        cfg = json.load(f)
    training = cfg.get('training', {})
    pipeline = cfg.get('pipeline', {})
    return {
        'n_epochs': training.get('n_epochs', 10),
        'embedding_dim': training.get('embedding_dim', 64),
        'hidden_dim': training.get('hidden_dim', 128),
        'batch_size': training.get('batch_size', 4096),
        'seed': training.get('seed', 42),
        'model_type': pipeline.get('model_type', 'rnn'),
        'k': pipeline.get('pos_k', 2),
    }


def train_models(grammar_dir, model_type='rnn', k=2, n_epochs=10,
                 embedding_dim=64, hidden_dim=128, batch_size=4096,
                 seed=42, force=False):
    """Train cloze models for a grammar directory.

    Args:
        grammar_dir: directory containing corpus.txt
        model_type: 'rnn', 'positional', or 'bow'
        k: context window width (ignored for rnn)
        n_epochs: training epochs
        force: retrain even if model files exist

    Returns:
        dict with paths to saved models and training times.
    """
    corpus_path = os.path.join(grammar_dir, 'corpus.txt')
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")

    mt_tag = {'positional': 'pos', 'rnn': 'rnn', 'bow': 'bow'}[model_type]

    if model_type == 'rnn':
        single_path = os.path.join(grammar_dir, 'rnn_cloze.pt')
        pair_path = os.path.join(grammar_dir, 'rnn_gap_cloze.pt')
    else:
        single_path = os.path.join(grammar_dir, f'{mt_tag}_cloze_k{k}.pt')
        pair_path = os.path.join(grammar_dir, f'{mt_tag}_pair_cloze_k{k}.pt')

    results = {'model_type': model_type, 'k': k, 'grammar_dir': grammar_dir}

    single_exists = os.path.exists(single_path) and not force
    pair_exists = os.path.exists(pair_path) and not force

    if single_exists and pair_exists:
        print(f"Both models already exist, skipping training.")
        print(f"  single: {single_path}")
        print(f"  pair:   {pair_path}")
        results['single_path'] = single_path
        results['pair_path'] = pair_path
        results['status'] = 'cached'
        return results

    nl = nl_module.NeuralLearner(corpus_path)
    nl.model_type = model_type
    nl.k = k
    nl.n_epochs = n_epochs
    nl.embedding_dim = embedding_dim
    nl.hidden_dim = hidden_dim
    nl.batch_size = batch_size
    nl.seed = seed
    nl.single_model_file = single_path if not force else None
    nl.pair_model_file = pair_path if (not force and model_type != 'rnn') else None
    nl.gap_model_file = pair_path if (not force and model_type == 'rnn') else None

    # Train single model
    t0 = time.time()
    if single_exists:
        print(f"Single model exists: {single_path}")
        nl.single_model_file = single_path
    else:
        nl.single_model_file = single_path
        if force and os.path.exists(single_path):
            os.remove(single_path)
    nl.train_single_model(verbose=True)
    single_time = time.time() - t0
    results['single_path'] = single_path
    results['single_time'] = round(single_time, 1)
    print(f"Single model: {single_time:.1f}s -> {single_path}")

    # Train pair/gap model
    t1 = time.time()
    if pair_exists:
        print(f"Pair model exists: {pair_path}")
        if model_type == 'rnn':
            nl.gap_model_file = pair_path
        else:
            nl.pair_model_file = pair_path
    else:
        if model_type == 'rnn':
            nl.gap_model_file = pair_path
        else:
            nl.pair_model_file = pair_path
        if force and os.path.exists(pair_path):
            os.remove(pair_path)
    nl.train_pair_model(verbose=True)
    pair_time = time.time() - t1
    results['pair_path'] = pair_path
    results['pair_time'] = round(pair_time, 1)
    print(f"Pair model: {pair_time:.1f}s -> {pair_path}")

    results['status'] = 'trained'
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Train cloze models for grammar experiments')
    parser.add_argument('grammar_dir',
                        help='Directory containing corpus.txt')
    parser.add_argument('--config', default=None,
                        help='JSON config file (overridden by explicit flags)')
    parser.add_argument('--model_type', default=None,
                        choices=['rnn', 'positional', 'bow'])
    parser.add_argument('--k', type=int, default=None)
    parser.add_argument('--n_epochs', type=int, default=None)
    parser.add_argument('--embedding_dim', type=int, default=None)
    parser.add_argument('--hidden_dim', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--force', action='store_true',
                        help='Retrain even if model files exist')

    args = parser.parse_args()

    # Defaults <- config <- CLI
    params = {
        'model_type': 'rnn', 'k': 2, 'n_epochs': 10,
        'embedding_dim': 64, 'hidden_dim': 128,
        'batch_size': 4096, 'seed': 42,
    }
    if args.config:
        params.update(load_config(args.config))
    for key in params:
        cli_val = getattr(args, key, None)
        if cli_val is not None:
            params[key] = cli_val

    results = train_models(
        args.grammar_dir, force=args.force, **params)

    print(f"\nDone: {results['status']}")


if __name__ == '__main__':
    main()
