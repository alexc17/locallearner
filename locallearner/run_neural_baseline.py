#!/usr/bin/env python3
"""
CLI script for training and evaluating LSTM language model baseline.

Usage:
    # Train
    python run_neural_baseline.py train --corpus corpus.txt --output model.pt

    # Evaluate against PCFG
    python run_neural_baseline.py evaluate --model model.pt --target target.pcfg
"""

import argparse
import json
import sys

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def train(args):
    """Train the neural language model."""
    from neural_baseline import LSTMLanguageModel, load_corpus, get_device

    # Load corpus
    print(f"Loading corpus from {args.corpus}")
    sentences = load_corpus(args.corpus)
    print(f"Loaded {len(sentences)} sentences")

    # Select device
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()
    print(f"Using device: {device}")

    # Create model
    model = LSTMLanguageModel(
        vocab_size=1,  # Will be updated during fit
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        dropout=args.dropout,
    )

    # Train
    print(f"Training for {args.epochs} epochs...")
    losses = model.fit(
        sentences,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=device,
        verbose=True,
        min_count=args.min_count,
    )

    # Save model
    model.save(args.output)
    print(f"Model saved to {args.output}")

    # Print final statistics
    print(f"\nVocabulary size: {len(model.vocab)}")
    print(f"Final training loss: {losses[-1]:.4f}")

    # Compute perplexity on training data
    model.eval()
    sentences_as_tuples = [tuple(s) for s in sentences[:min(100, len(sentences))]]
    ppl = model.perplexity(sentences_as_tuples, device)
    print(f"Training perplexity (sample): {ppl:.2f}")

    # Save loss curve if requested
    if args.loss_curve:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(losses) + 1), losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.grid(True, alpha=0.3)
        plt.savefig(args.loss_curve, format='pdf', bbox_inches='tight')
        plt.close()
        print(f"Loss curve saved to {args.loss_curve}")


def evaluate(args):
    """Evaluate neural model against a target PCFG."""
    from neural_baseline import LSTMLanguageModel, get_device
    import wcfg
    import numpy as np

    # Load model
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()

    print(f"Loading model from {args.model}")
    model = LSTMLanguageModel.load(args.model, device)
    print(f"Vocabulary size: {len(model.vocab)}")

    # Load target PCFG
    print(f"Loading target PCFG from {args.target}")
    target_pcfg = wcfg.load_wcfg_from_file(args.target)

    # Sample from target and compute KLD
    if args.seed is not None:
        rng = np.random.RandomState(args.seed)
    else:
        rng = np.random.RandomState()

    sampler = wcfg.Sampler(target_pcfg, random=rng)
    insider = wcfg.InsideComputation(target_pcfg)

    total_kld = 0.0
    total_length = 0
    n_samples = 0
    failures = 0

    print(f"Sampling {args.samples} strings from target PCFG...")

    for i in range(args.samples):
        try:
            tree = sampler.sample_tree()
            from utility import collect_yield
            s = collect_yield(tree)

            if len(s) > args.maxlength:
                continue

            # Target log prob
            target_lp = insider.inside_log_probability(s)

            # Neural log prob
            neural_lp = model.log_probability(s, device)

            total_kld += target_lp - neural_lp
            total_length += len(s)
            n_samples += 1

        except Exception as e:
            failures += 1
            if args.verbose:
                print(f"Sample {i} failed: {e}")

    if n_samples == 0:
        print("Error: No valid samples collected")
        sys.exit(1)

    # Compute metrics
    string_kld = total_kld / n_samples
    avg_length = total_length / n_samples

    results = {
        'string_kld': string_kld,
        'samples': n_samples,
        'failures': failures,
        'avg_length': avg_length,
        'maxlength': args.maxlength,
        'model': args.model,
        'target': args.target,
    }

    print(f"\nResults:")
    print(f"  String KLD: {string_kld:.4f}")
    print(f"  Samples: {n_samples}")
    print(f"  Failures: {failures}")
    print(f"  Avg length: {avg_length:.2f}")

    if args.json:
        with open(args.json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.json}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Neural baseline LSTM language model for PCFG comparison',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
commands:
  train       Train the language model
              --corpus FILE      Path to training corpus (required)
              --output FILE      Path to save model (required)
              --embed-dim N      Embedding dimension (default: 128)
              --hidden-dim N     LSTM hidden dimension (default: 256)
              --layers N         Number of LSTM layers (default: 2)
              --epochs N         Training epochs (default: 100)
              --loss-curve FILE  Save loss curve as PDF
              --device DEVICE    Device: mps, cuda, cpu (default: auto)

  evaluate    Evaluate against a target PCFG
              --model FILE       Path to trained model (required)
              --target FILE      Path to target PCFG (required)
              --samples N        Number of samples (default: 1000)
              --maxlength N      Maximum string length (default: 20)
              --json FILE        Save results as JSON

examples:
  python run_neural_baseline.py train --corpus data.txt --output model.pt
  python run_neural_baseline.py evaluate --model model.pt --target grammar.pcfg
""",
    )
    subparsers = parser.add_subparsers(dest='command', metavar='command')

    # Train subcommand
    train_parser = subparsers.add_parser('train', help='Train the language model')
    train_parser.add_argument('--corpus', required=True, help='Path to training corpus')
    train_parser.add_argument('--output', required=True, help='Path to save model')
    train_parser.add_argument('--embed-dim', type=int, default=128, help='Embedding dimension (default: 128)')
    train_parser.add_argument('--hidden-dim', type=int, default=256, help='LSTM hidden dimension (default: 256)')
    train_parser.add_argument('--layers', type=int, default=2, help='Number of LSTM layers (default: 2)')
    train_parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate (default: 0.1)')
    train_parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs (default: 100)')
    train_parser.add_argument('--batch-size', type=int, default=32, help='Batch size (default: 32)')
    train_parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    train_parser.add_argument('--min-count', type=int, default=1, help='Minimum word count for vocabulary (default: 1)')
    train_parser.add_argument('--device', type=str, help='Device to use (mps, cuda, cpu)')
    train_parser.add_argument('--loss-curve', type=str, help='Save training loss curve to PDF file')

    # Evaluate subcommand
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate against a target PCFG')
    eval_parser.add_argument('--model', required=True, help='Path to trained model')
    eval_parser.add_argument('--target', required=True, help='Path to target PCFG')
    eval_parser.add_argument('--samples', type=int, default=1000, help='Number of samples (default: 1000)')
    eval_parser.add_argument('--maxlength', type=int, default=20, help='Maximum string length (default: 20)')
    eval_parser.add_argument('--seed', type=int, default=None, help='Random seed')
    eval_parser.add_argument('--json', type=str, help='Path to save results as JSON')
    eval_parser.add_argument('--device', type=str, help='Device to use (mps, cuda, cpu)')
    eval_parser.add_argument('--verbose', action='store_true', help='Print verbose output')

    args = parser.parse_args()

    if not TORCH_AVAILABLE:
        print("Error: PyTorch is required for neural baseline.")
        print("Install with: pip install torch")
        sys.exit(1)

    if args.command == 'train':
        train(args)
    elif args.command == 'evaluate':
        evaluate(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
