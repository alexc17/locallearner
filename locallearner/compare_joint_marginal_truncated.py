#!/usr/bin/env python3
"""Compare marginal vs joint NMF using truncated grammars.

Two-phase workflow:
  1. generate: Create grammars and sample corpora, saving to disk.
  2. run:      Run experiments on pre-generated data.

All data and results are stored under a persistent directory
(default: locallearner/tmp/jm_experiment/) that can be deleted manually.

Usage:
    # Phase 1: generate grammars and corpora
    python3 compare_joint_marginal_truncated.py generate --n-grammars 20 --sentences 1000000

    # Phase 2: run experiments on pre-generated data
    python3 compare_joint_marginal_truncated.py run

    # Or run on a subset
    python3 compare_joint_marginal_truncated.py run --grammars 0,1,2,3
"""

import sys
import os
import json
import argparse
import time

import numpy as np
import numpy.random

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
SYNTHETICPCFG_PATH = os.path.join(os.path.dirname(__file__),
    '..', '..', '..', 'cursor_syntheticpcfg', 'syntheticpcfg')
sys.path.insert(0, SYNTHETICPCFG_PATH)

import wcfg
import locallearner as ll_module
import evaluation

from syntheticpcfg import pcfgfactory, pcfg as spcfg, utility as sutil

# Default base directory for all experiment data
DEFAULT_BASE = os.path.join(os.path.dirname(__file__), '..', 'tmp', 'jm_experiment')


# ============================================================
# Grammar / corpus generation
# ============================================================

def generate_grammar(n_nt, n_t, seed):
    """Generate a random PCFG with n_nt nonterminals."""
    np.random.seed(seed)
    factory = pcfgfactory.FullPCFGFactory(
        nonterminals=n_nt, terminals=n_t)
    factory.lexical_distribution = pcfgfactory.LogNormalPrior(sigma=4.0)
    alpha = 1.0 / ((n_nt - 1) ** 2 + 1)
    factory.binary_distribution = pcfgfactory.LexicalDirichlet(dirichlet=alpha)
    factory.length_distribution = pcfgfactory.LengthDistribution()
    factory.length_distribution.set_poisson(5.0, 20)
    return factory.sample_uniform()


def truncate_grammar(grammar, ent_threshold=0.1, eprod_threshold=1e-5):
    """Truncate grammar: remove NTs with expectation < ent_threshold."""
    pe = grammar.production_expectations()
    nte = grammar.nonterminal_expectations()

    kept_nts = set(nt for nt, e in nte.items() if e > ent_threshold)
    if grammar.start not in kept_nts:
        return None

    new = spcfg.PCFG()
    new.start = grammar.start
    new.nonterminals = kept_nts

    new.productions = []
    for prod in grammar.productions:
        if prod[0] not in kept_nts:
            continue
        if pe[prod] < eprod_threshold:
            continue
        if len(prod) == 3:
            if prod[1] not in kept_nts or prod[2] not in kept_nts:
                continue
        new.productions.append(prod)

    lhs_nts = set(prod[0] for prod in new.productions)
    if lhs_nts != kept_nts:
        return None

    rhs_nts = set()
    for prod in new.productions:
        if len(prod) == 3:
            rhs_nts.add(prod[1])
            rhs_nts.add(prod[2])
    non_start = kept_nts - {grammar.start}
    if not non_start.issubset(rhs_nts):
        return None

    new.terminals = set(prod[1] for prod in new.productions if len(prod) == 2)
    new.parameters = {prod: grammar.parameters[prod] for prod in new.productions}
    new.normalise()
    return new


def sample_corpus(grammar, n_sentences, seed):
    """Sample sentences from grammar."""
    rng = np.random.RandomState(seed)
    sampler = spcfg.Sampler(grammar, random=rng)
    sentences = []
    for _ in range(n_sentences):
        tree = sampler.sample_tree()
        s = sutil.collect_yield(tree)
        sentences.append(' '.join(s))
    return sentences


def cmd_generate(args):
    """Phase 1: Generate grammars and corpora, save to disk."""
    base = os.path.abspath(args.base)
    os.makedirs(base, exist_ok=True)

    manifest = {
        'n_grammars': args.n_grammars,
        'n_sentences': args.sentences,
        'original_nts': 10,
        'n_terminals': 1000,
        'ent_threshold': 0.1,
        'grammars': [],
    }

    for g in range(args.n_grammars):
        grammar_seed = 100 * g + 7
        corpus_seed = 2000 * g + 1
        gdir = os.path.join(base, f'g{g:03d}')
        os.makedirs(gdir, exist_ok=True)

        print(f"[{g+1}/{args.n_grammars}] seed={grammar_seed} ...",
              end=' ', flush=True)

        # Generate and truncate
        grammar = generate_grammar(10, 1000, grammar_seed)
        nte = grammar.nonterminal_expectations()
        truncated = truncate_grammar(grammar, ent_threshold=0.1)

        if truncated is None:
            print("SKIP (truncation failed)")
            manifest['grammars'].append({
                'index': g, 'grammar_seed': grammar_seed,
                'corpus_seed': corpus_seed, 'error': 'truncation failed',
            })
            continue

        n_nt = len(truncated.nonterminals)
        nte_trunc = truncated.nonterminal_expectations()
        min_nte = min(nte_trunc.values())

        # Save grammar
        grammar_path = os.path.join(gdir, 'grammar.pcfg')
        truncated.store(grammar_path)

        # Sample and save corpus
        t0 = time.time()
        sentences = sample_corpus(truncated, args.sentences, corpus_seed)
        corpus_path = os.path.join(gdir, 'corpus.txt')
        with open(corpus_path, 'w') as f:
            for s in sentences:
                f.write(s + '\n')
        sample_time = time.time() - t0

        info = {
            'index': g,
            'grammar_seed': grammar_seed,
            'corpus_seed': corpus_seed,
            'n_nonterminals': n_nt,
            'min_nt_expectation': round(min_nte, 4),
            'n_sentences': args.sentences,
            'sample_time': round(sample_time, 1),
            'grammar_path': os.path.relpath(grammar_path, base),
            'corpus_path': os.path.relpath(corpus_path, base),
        }
        manifest['grammars'].append(info)
        print(f"{n_nt} NTs, min_expect={min_nte:.3f}, "
              f"sampled in {sample_time:.1f}s")

    manifest_path = os.path.join(base, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to {manifest_path}")
    print(f"Data directory: {base}")


# ============================================================
# Experiment runner
# ============================================================

def evaluate_kernels_against_target(target_path, kernels):
    """Evaluate kernel quality against target grammar."""
    target = wcfg.load_wcfg_from_file(target_path)
    n_target = len(target.nonterminals)
    if len(kernels) == n_target:
        result = evaluation.evaluate_kernels_hungarian(target, kernels)
        return {
            'mean_posterior': result['mean_posterior'],
            'min_posterior': result['min_posterior'],
            'accuracy': result['accuracy'],
        }
    else:
        return {
            'mean_posterior': None,
            'min_posterior': None,
            'accuracy': None,
        }


def run_detection(corpus_path, n_nonterminals, seed=42, min_count=10,
                  number_clusters=10, feature_mode='marginal',
                  auto=False, max_nt=30, distance_threshold=0.0,
                  cluster_file=None, bigram_file=None, trigram_file=None):
    """Run NMF kernel detection."""
    learner = ll_module.LocalLearner(corpus_path)
    learner.number_clusters = number_clusters
    learner.min_count_nmf = min_count
    learner.seed = seed
    learner.feature_mode = feature_mode
    learner.distance_threshold = distance_threshold
    learner.cluster_file = cluster_file
    learner.bigram_file = bigram_file
    learner.trigram_file = trigram_file

    if auto:
        learner.nonterminals = 0
        learner.max_nonterminals = max_nt
    else:
        learner.nonterminals = n_nonterminals

    t0 = time.time()
    kernels = learner.find_kernels(verbose=False)
    elapsed = time.time() - t0
    return kernels, learner, elapsed


def run_single(grammar_info, base):
    """Run marginal/joint experiments on one pre-generated grammar."""
    grammar_path = os.path.join(base, grammar_info['grammar_path'])
    corpus_path = os.path.join(base, grammar_info['corpus_path'])
    n_nt = grammar_info['n_nonterminals']
    n_sentences = grammar_info['n_sentences']

    result = dict(grammar_info)
    min_count = max(5, n_sentences // 1000)
    n_clusters = grammar_info.get('n_clusters', 10)
    fw_dist = grammar_info.get('distance_threshold', 0.0)

    # Cache files: shared across feature modes and auto/fixed runs
    gdir = os.path.dirname(corpus_path)
    cluster_file = os.path.join(gdir, f'clusters_c{n_clusters}_s42.clusters')
    bigram_file = os.path.join(gdir, 'bigrams.gz')
    trigram_file = os.path.join(gdir, 'trigrams.gz')

    for mode in ['marginal', 'joint']:
        p = mode[:4]

        # Fixed-count
        kernels_f, _, elapsed_f = run_detection(
            corpus_path, n_nt, seed=42, min_count=min_count,
            number_clusters=n_clusters, feature_mode=mode, auto=False,
            cluster_file=cluster_file,
            bigram_file=bigram_file, trigram_file=trigram_file)
        eval_f = evaluate_kernels_against_target(grammar_path, kernels_f)
        result[f'{p}_fixed_accuracy'] = eval_f['accuracy']
        result[f'{p}_fixed_mean_post'] = eval_f['mean_posterior']
        result[f'{p}_fixed_elapsed'] = round(elapsed_f, 2)

        # Auto-detection
        kernels_a, learner_a, elapsed_a = run_detection(
            corpus_path, n_nt, seed=42, min_count=min_count,
            number_clusters=n_clusters, feature_mode=mode,
            auto=True, max_nt=20,
            distance_threshold=fw_dist,
            cluster_file=cluster_file,
            bigram_file=bigram_file, trigram_file=trigram_file)
        eval_a = evaluate_kernels_against_target(grammar_path, kernels_a)
        result[f'{p}_auto_n_detected'] = len(kernels_a)
        result[f'{p}_auto_correct'] = len(kernels_a) == n_nt
        result[f'{p}_auto_delta'] = len(kernels_a) - n_nt
        result[f'{p}_auto_accuracy'] = eval_a['accuracy']
        result[f'{p}_auto_elapsed'] = round(elapsed_a, 2)
        result[f'{p}_auto_stop'] = getattr(learner_a, 'stop_reason', None)

    return result


def cmd_run(args):
    """Phase 2: Run experiments on pre-generated data."""
    base = os.path.abspath(args.base)
    manifest_path = os.path.join(base, 'manifest.json')

    with open(manifest_path) as f:
        manifest = json.load(f)

    grammars = [g for g in manifest['grammars'] if 'error' not in g]

    # Filter to requested subset
    if args.grammars is not None:
        indices = set(int(x) for x in args.grammars.split(','))
        grammars = [g for g in grammars if g['index'] in indices]

    # Inject run-time parameters into each grammar info
    for g in grammars:
        g['n_clusters'] = args.clusters
        g['distance_threshold'] = args.distance

    total = len(grammars)
    results = []

    for i, ginfo in enumerate(grammars):
        label = (f"g{ginfo['index']:03d} seed={ginfo['grammar_seed']}, "
                 f"{ginfo['n_nonterminals']} NTs, "
                 f"N={ginfo['n_sentences']}")
        print(f"[{i+1}/{total}] {label} ...", flush=True)

        try:
            result = run_single(ginfo, base)
            results.append(result)

            for mode in ['marginal', 'joint']:
                p = mode[:4]
                fa = result.get(f'{p}_fixed_accuracy')
                ad = result.get(f'{p}_auto_delta')
                ac = result.get(f'{p}_auto_correct', False)
                ok = 'OK' if ac else 'WRONG'
                fa_str = f'{fa:.3f}' if fa is not None else 'N/A'
                print(f"    {mode:8s}: fixed_acc={fa_str}, "
                      f"auto_delta={ad:+d} ({ok}), "
                      f"stop={result.get(f'{p}_auto_stop', '?')}")
        except Exception as e:
            import traceback
            print(f"    ERROR: {e}")
            results.append({**ginfo, 'error': str(e)})

    # Save results
    tag = f'_c{args.clusters}' if args.clusters != 10 else ''
    if args.distance > 0:
        tag += f'_d{args.distance:.3f}'
    results_path = os.path.join(base, f'results{tag}.json')
    save_results = []
    for r in results:
        sr = dict(r)
        sr.pop('traceback', None)
        save_results.append(sr)
    with open(results_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    print_summary(results)

    plot_path = os.path.join(base, f'comparison{tag}.png')
    plot_results(results, plot_path)


# ============================================================
# Summary and plotting
# ============================================================

def print_summary(results):
    """Print comparison summary."""
    valid = [r for r in results if not r.get('error')]
    if not valid:
        print("No valid results.")
        return

    from collections import defaultdict
    by_nt = defaultdict(list)
    for r in valid:
        by_nt[r['n_nonterminals']].append(r)

    n_sentences = valid[0].get('n_sentences', '?')

    print("\n" + "=" * 110)
    print(f"TRUNCATED GRAMMAR RESULTS  (N={n_sentences} sentences)")
    print(f"Total valid trials: {len(valid)}")
    print(f"NT count distribution: " +
          ", ".join(f"{k}:{len(v)}" for k, v in sorted(by_nt.items())))
    print("=" * 110)

    print(f"\n{'NTs':>4s} {'N':>4s} "
          f"{'Marg Fix':>9s} {'Joint Fix':>10s} "
          f"{'Marg Auto':>10s} {'Joint Auto':>11s} "
          f"{'Marg MnD':>9s} {'Joint MnD':>10s}")
    print("-" * 80)

    total_mc = total_jc = total_n = 0

    for n_nt in sorted(by_nt.keys()):
        rs = by_nt[n_nt]
        n = len(rs)
        total_n += n

        mfa = [r['marg_fixed_accuracy'] for r in rs
               if r.get('marg_fixed_accuracy') is not None]
        jfa = [r['join_fixed_accuracy'] for r in rs
               if r.get('join_fixed_accuracy') is not None]
        mc = sum(1 for r in rs if r.get('marg_auto_correct'))
        jc = sum(1 for r in rs if r.get('join_auto_correct'))
        total_mc += mc
        total_jc += jc
        md = np.mean([r['marg_auto_delta'] for r in rs
                      if r.get('marg_auto_delta') is not None])
        jd = np.mean([r['join_auto_delta'] for r in rs
                      if r.get('join_auto_delta') is not None])

        mfa_str = f"{np.mean(mfa):.3f}" if mfa else "N/A"
        jfa_str = f"{np.mean(jfa):.3f}" if jfa else "N/A"
        ma_str = f"{mc}/{n}={mc/n:.0%}"
        ja_str = f"{jc}/{n}={jc/n:.0%}"

        print(f"{n_nt:4d} {n:4d} "
              f"{mfa_str:>9s} {jfa_str:>10s} "
              f"{ma_str:>10s} {ja_str:>11s} "
              f"{md:+9.1f} {jd:+10.1f}")

    print("-" * 80)
    mr = total_mc / total_n if total_n else 0
    jr = total_jc / total_n if total_n else 0
    ma_str = f"{total_mc}/{total_n}={mr:.0%}"
    ja_str = f"{total_jc}/{total_n}={jr:.0%}"
    print(f"{'ALL':>4s} {total_n:4d} "
          f"{'':>9s} {'':>10s} "
          f"{ma_str:>10s} {ja_str:>11s}")


def plot_results(results, output_path):
    """Generate comparison plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    valid = [r for r in results if not r.get('error')]
    if not valid:
        print("No valid results to plot.")
        return

    from collections import defaultdict
    by_nt = defaultdict(list)
    for r in valid:
        by_nt[r['n_nonterminals']].append(r)

    nt_vals = sorted(by_nt.keys())
    n_sentences = valid[0].get('n_sentences', '?')

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f'Marginal vs Joint: Truncated 10-NT grammars, '
        f'N={n_sentences} sentences',
        fontsize=13, fontweight='bold', y=0.98)

    bar_w = 0.35

    # --- Plot 1: Fixed-count accuracy ---
    ax = axes[0, 0]
    x_pos = np.arange(len(nt_vals))
    for j, (mode, label, color) in enumerate([
            ('marg', 'Marginal', '#2196F3'),
            ('join', 'Joint', '#FF9800')]):
        rates = []
        annots = []
        for n_nt in nt_vals:
            rs = by_nt[n_nt]
            accs = [r[f'{mode}_fixed_accuracy'] for r in rs
                    if r.get(f'{mode}_fixed_accuracy') is not None]
            rates.append(np.mean(accs) if accs else 0)
            annots.append(f'{len(accs)}')
        offset = (j - 0.5) * bar_w
        bars = ax.bar(x_pos + offset, rates, bar_w * 0.9, label=label,
                      color=color, alpha=0.85)
        for k, (bar, ct) in enumerate(zip(bars, annots)):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01, f'n={ct}',
                    ha='center', va='bottom', fontsize=7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{nt} NTs' for nt in nt_vals])
    ax.set_ylabel('Mean accuracy (Hungarian)')
    ax.set_title('Fixed-count: kernel quality')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.2, axis='y')

    # --- Plot 2: Fixed-count mean posterior ---
    ax = axes[0, 1]
    for j, (mode, label, color) in enumerate([
            ('marg', 'Marginal', '#2196F3'),
            ('join', 'Joint', '#FF9800')]):
        posts = []
        for n_nt in nt_vals:
            rs = by_nt[n_nt]
            ps = [r[f'{mode}_fixed_mean_post'] for r in rs
                  if r.get(f'{mode}_fixed_mean_post') is not None]
            posts.append(np.mean(ps) if ps else 0)
        offset = (j - 0.5) * bar_w
        ax.bar(x_pos + offset, posts, bar_w * 0.9, label=label,
               color=color, alpha=0.85)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{nt} NTs' for nt in nt_vals])
    ax.set_ylabel('Mean posterior P(NT|anchor)')
    ax.set_title('Fixed-count: anchor quality')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.2, axis='y')

    # --- Plot 3: Auto-detection accuracy ---
    ax = axes[1, 0]
    for j, (mode, label, color) in enumerate([
            ('marg', 'Marginal', '#2196F3'),
            ('join', 'Joint', '#FF9800')]):
        rates = []
        annots = []
        for n_nt in nt_vals:
            rs = by_nt[n_nt]
            n = len(rs)
            c = sum(1 for r in rs if r.get(f'{mode}_auto_correct'))
            rates.append(c / n if n > 0 else 0)
            annots.append(f'{c}/{n}')
        offset = (j - 0.5) * bar_w
        bars = ax.bar(x_pos + offset, rates, bar_w * 0.9, label=label,
                      color=color, alpha=0.85)
        for k, (bar, ct) in enumerate(zip(bars, annots)):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02, ct,
                    ha='center', va='bottom', fontsize=7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{nt} NTs' for nt in nt_vals])
    ax.set_ylabel('Accuracy (exact NT count)')
    ax.set_title('Auto-detection accuracy')
    ax.legend()
    ax.set_ylim(0, 1.3)
    ax.grid(True, alpha=0.2, axis='y')

    # --- Plot 4: Per-trial scatter ---
    ax = axes[1, 1]
    for j, (mode, label, color, marker) in enumerate([
            ('marg', 'Marginal', '#2196F3', 'o'),
            ('join', 'Joint', '#FF9800', 's')]):
        true_nts = []
        det_nts = []
        for r in valid:
            if r.get(f'{mode}_auto_n_detected') is not None:
                true_nts.append(r['n_nonterminals'])
                det_nts.append(r[f'{mode}_auto_n_detected'])
        jitter = np.random.RandomState(j).uniform(-0.15, 0.15, len(true_nts))
        ax.scatter(np.array(true_nts) + jitter, det_nts,
                   alpha=0.5, s=30, c=color, marker=marker, label=label)
    mn = min(nt_vals) - 1
    mx = max(max(nt_vals) + 2, max(
        r.get('marg_auto_n_detected', 0) for r in valid) + 1,
        max(r.get('join_auto_n_detected', 0) for r in valid) + 1)
    ax.plot([mn, mx], [mn, mx], 'k--', alpha=0.5, label='perfect')
    ax.set_xlabel('True NTs (after truncation)')
    ax.set_ylabel('Detected NTs')
    ax.set_title('Detected vs True')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compare marginal vs joint on truncated grammars')
    subparsers = parser.add_subparsers(dest='command')

    # --- generate ---
    p_gen = subparsers.add_parser('generate',
        help='Generate grammars and sample corpora')
    p_gen.add_argument('--n-grammars', type=int, default=20,
                       help='Number of 10-NT grammars (default 20)')
    p_gen.add_argument('--sentences', type=int, default=100000,
                       help='Sentences per corpus (default 100000)')
    p_gen.add_argument('--base', default=DEFAULT_BASE,
                       help=f'Base directory (default {DEFAULT_BASE})')

    # --- run ---
    p_run = subparsers.add_parser('run',
        help='Run experiments on pre-generated data')
    p_run.add_argument('--base', default=DEFAULT_BASE,
                       help=f'Base directory (default {DEFAULT_BASE})')
    p_run.add_argument('--grammars', default=None,
                       help='Comma-separated grammar indices to run '
                            '(default: all)')
    p_run.add_argument('--clusters', type=int, default=10,
                       help='Number of Ney-Essen clusters (default 10)')
    p_run.add_argument('--distance', type=float, default=0.0,
                       help='Hyperplane distance threshold for auto-detection '
                            '(0=disabled)')

    args = parser.parse_args()

    if args.command == 'generate':
        cmd_generate(args)
    elif args.command == 'run':
        cmd_run(args)
    else:
        parser.print_help()
