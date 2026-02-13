#!/usr/bin/env python3
"""Visualise pairwise divergences between NMF kernels.

Runs NMF with a deliberately generous count (true N + extra),
then computes and plots pairwise divergence matrices — including
asymmetric and Renyi divergences — to see whether spurious kernels
can be identified by their distributional similarity to genuine ones.

Usage:
    python3 visualise_kernel_divergences.py [--extra 5] [--output divergences.png]
"""

import sys
import os
import argparse
import math
import tempfile
import logging

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
SYNTHETICPCFG_PATH = os.path.join(os.path.dirname(__file__),
    '..', '..', '..', 'cursor_syntheticpcfg', 'syntheticpcfg')
sys.path.insert(0, SYNTHETICPCFG_PATH)

import locallearner as ll_module
import nmf as nmf_mod
import wcfg

from syntheticpcfg import pcfgfactory, pcfg as spcfg, utility as sutil
from scipy.stats import chi2_contingency


def generate_grammar(n_nonterminals, n_terminals, seed):
    """Generate a random PCFG."""
    np.random.seed(seed)
    factory = pcfgfactory.FullPCFGFactory(
        nonterminals=n_nonterminals, terminals=n_terminals)
    factory.lexical_distribution = pcfgfactory.LogNormalPrior(sigma=4.0)
    alpha = 1.0 / ((n_nonterminals - 1) ** 2 + 1)
    factory.binary_distribution = pcfgfactory.LexicalDirichlet(dirichlet=alpha)
    factory.length_distribution = pcfgfactory.LengthDistribution()
    factory.length_distribution.set_poisson(5.0, 20)
    return factory.sample_uniform()


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


# ---- Divergence functions ----

def kl_divergence(p, q):
    """KL(p || q). Returns inf if any p_i > 0 where q_i == 0."""
    d = 0.0
    for pi, qi in zip(p, q):
        if pi > 1e-15:
            if qi <= 1e-15:
                return float('inf')
            d += pi * math.log(pi / qi)
    return d


def renyi_divergence(p, q, alpha):
    """Renyi divergence D_alpha(p || q).

    D_alpha = 1/(alpha-1) * log( sum_i p_i^alpha * q_i^(1-alpha) )

    Special cases:
      alpha -> 1: KL divergence
      alpha = 0.5: -2 * log(Bhattacharyya coefficient)
      alpha = 2: log( sum_i p_i^2 / q_i )
      alpha -> inf: log( max_i p_i / q_i )
    """
    if alpha == 1.0:
        return kl_divergence(p, q)

    if alpha == float('inf'):
        max_ratio = 0.0
        for pi, qi in zip(p, q):
            if pi > 1e-15:
                if qi <= 1e-15:
                    return float('inf')
                max_ratio = max(max_ratio, pi / qi)
        return math.log(max_ratio) if max_ratio > 0 else 0.0

    s = 0.0
    for pi, qi in zip(p, q):
        if pi > 1e-15:
            if qi <= 1e-15:
                if alpha > 1:
                    return float('inf')
                # alpha < 1: q_i^(1-alpha) -> inf, but p_i^alpha -> 0
                # skip this term (it's 0 * inf, treat as 0)
                continue
            s += (pi ** alpha) * (qi ** (1 - alpha))
    if s <= 0:
        return float('inf')
    return math.log(s) / (alpha - 1)


def compute_pairwise_divergence(nmf_obj, kernel_indices, div_func):
    """Compute asymmetric pairwise divergence D(row || col)."""
    k = len(kernel_indices)
    matrix = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            p = nmf_obj.raw_data[kernel_indices[i], :]
            q = nmf_obj.raw_data[kernel_indices[j], :]
            matrix[i, j] = div_func(p, q)
    return matrix


def compute_pairwise_cramers_v(learner, kernel_names):
    """Compute pairwise Cramer's V from word-level bigram contexts."""
    M = learner._build_word_context_matrix()
    k = len(kernel_names)
    v_matrix = np.zeros((k, k))

    for i in range(k):
        for j in range(i + 1, k):
            wi = learner.word2idx.get(kernel_names[i])
            wj = learner.word2idx.get(kernel_names[j])
            if wi is None or wj is None:
                v_matrix[i, j] = v_matrix[j, i] = float('nan')
                continue

            row_i = M[wi, :]
            row_j = M[wj, :]

            if np.sum(row_i) == 0 or np.sum(row_j) == 0:
                v_matrix[i, j] = v_matrix[j, i] = float('nan')
                continue

            table = np.vstack([row_i, row_j])
            col_sums = table.sum(axis=0)
            mask = col_sums > 5
            if np.sum(mask) < 2:
                v_matrix[i, j] = v_matrix[j, i] = float('nan')
                continue

            table_f = table[:, mask]
            try:
                chi2_stat, p_val, dof, _ = chi2_contingency(table_f)
            except ValueError:
                v_matrix[i, j] = v_matrix[j, i] = float('nan')
                continue

            n_total = table_f.sum()
            cramers_v = math.sqrt(chi2_stat / n_total) if n_total > 0 else 0.0
            v_matrix[i, j] = cramers_v
            v_matrix[j, i] = cramers_v

    return v_matrix


def compute_fw_posteriors(nmf_obj, kernel_indices):
    """For each kernel, compute its FW decomposition using all OTHER kernels.

    Returns a (k, k) matrix where row i = FW weights for kernel i
    expressed in terms of all k kernels (weight on self forced to 0,
    then renormalised from the leave-one-out decomposition).

    Also returns the FW residual (distance) for each kernel.
    """
    k = len(kernel_indices)
    posteriors = np.zeros((k, k))
    residuals = np.zeros(k)

    for i in range(k):
        # Build basis from all kernels EXCEPT i
        others = [j for j in range(k) if j != i]
        if not others:
            continue

        f = nmf_obj.f
        MM = np.zeros((f, len(others)))
        for col, j in enumerate(others):
            MM[:, col] = nmf_obj.data[kernel_indices[j], :]

        # FW on kernel i using the other kernels
        y = nmf_obj.data[kernel_indices[i], :]
        nk = len(others)
        x = np.ones(nk) / nk
        e = np.eye(nk)
        oldd2 = float('inf')
        for iteration in range(100):
            y0 = MM @ x
            v1 = y - y0
            d2 = np.linalg.norm(v1)
            if d2 < 1e-5 or abs(d2 - oldd2) < 1e-5:
                break
            oldd2 = d2
            gradient = v1 @ MM
            idx = np.argmax(gradient)
            y1 = MM[:, idx]
            if np.linalg.norm(y1 - y0) < 1e-5:
                break
            # Line search
            alpha_num = np.dot(y0 - y1, y0 - y)
            beta = np.dot(y1 - y0, y1 - y0)
            gamma = alpha_num / beta if beta > 0 else 0.0
            gamma = max(0.0, min(1.0, gamma))
            x += gamma * (e[idx, :] - x)

        residuals[i] = d2
        # Map back: others[col] -> full index
        for col, j in enumerate(others):
            posteriors[i, j] = x[col]

    return posteriors, residuals


def label_kernels(kernel_names, grammar_path, true_nt):
    """Label each kernel as genuine or spurious."""
    labels = []
    for i, name in enumerate(kernel_names):
        if i == 0:
            labels.append('S')
        elif i < true_nt:
            labels.append(f'G{i}')
        else:
            labels.append(f'X{i}')
    return labels


def plot_heatmap(ax, matrix, tick_labels, title, n_nt, cmap='YlOrRd',
                 vmin=None, vmax=None, annotate=True, fmt='.2f'):
    """Plot a single heatmap with genuine-kernel box."""
    masked = matrix.copy()
    np.fill_diagonal(masked, np.nan)

    # Clip inf for display
    finite_vals = masked[np.isfinite(masked)]
    if len(finite_vals) == 0:
        return
    if vmax is None:
        vmax = np.percentile(finite_vals, 95)
    if vmin is None:
        vmin = np.min(finite_vals)

    display = np.where(np.isfinite(masked), masked, vmax)
    np.fill_diagonal(display, np.nan)

    import matplotlib.pyplot as plt

    im = ax.imshow(display, cmap=cmap, aspect='equal',
                   interpolation='nearest', vmin=vmin, vmax=vmax)

    rect = plt.Rectangle((-0.5, -0.5), n_nt, n_nt,
                         linewidth=2.5, edgecolor='blue',
                         facecolor='none', linestyle='--')
    ax.add_patch(rect)

    ax.set_xticks(range(len(tick_labels)))
    ax.set_xticklabels(tick_labels, fontsize=6, rotation=45, ha='right')
    ax.set_yticks(range(len(tick_labels)))
    ax.set_yticklabels(tick_labels, fontsize=6)
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.8)

    if annotate:
        k = len(tick_labels)
        for i in range(k):
            for j in range(k):
                if i == j:
                    continue
                val = matrix[i, j]
                if np.isfinite(val):
                    fs = 5 if k <= 12 else 4
                    color = ('white'
                             if val > np.nanmedian(finite_vals) else 'black')
                    ax.text(j, i, f'{val:{fmt}}', ha='center', va='center',
                            fontsize=fs, color=color)


def run_and_visualise(n_nt, n_sentences, grammar_seed, corpus_seed,
                      n_extra, output_path, feature_mode='marginal'):
    """Run NMF with extra kernels and visualise divergences."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_total = n_nt + n_extra
    print(f"Grammar: {n_nt} NTs, seed={grammar_seed}")
    print(f"Corpus: {n_sentences} sentences, seed={corpus_seed}")
    print(f"Selecting {n_total} kernels ({n_nt} genuine + {n_extra} extra)")
    print(f"Feature mode: {feature_mode}")

    grammar = generate_grammar(n_nt, 1000, grammar_seed)

    with tempfile.TemporaryDirectory() as tmpdir:
        grammar_path = os.path.join(tmpdir, 'grammar.pcfg')
        grammar.store(grammar_path)

        corpus_path = os.path.join(tmpdir, 'corpus.txt')
        sentences = sample_corpus(grammar, n_sentences, corpus_seed)
        with open(corpus_path, 'w') as f:
            for s in sentences:
                f.write(s + '\n')

        learner = ll_module.LocalLearner(corpus_path)
        learner.nonterminals = n_total
        learner.number_clusters = 10
        learner.min_count_nmf = max(5, n_sentences // 1000)
        learner.seed = 42
        learner.feature_mode = feature_mode

        kernels = learner.find_kernels(verbose=False)
        print(f"Selected kernels: {kernels}")

        nmf_obj = learner.nmf
        kernel_indices = list(nmf_obj.bases)
        kernel_names = kernels
        labels = label_kernels(kernel_names, grammar_path, n_nt)

        tick_labels = [f'{labels[i]}\n{kernel_names[i]}'
                       for i in range(len(kernel_names))]

        # ---- Compute all pairwise divergences ----
        print("Computing pairwise divergences...")

        kl_fwd = compute_pairwise_divergence(
            nmf_obj, kernel_indices, lambda p, q: kl_divergence(p, q))
        kl_rev = compute_pairwise_divergence(
            nmf_obj, kernel_indices, lambda p, q: kl_divergence(q, p))

        renyi_half = compute_pairwise_divergence(
            nmf_obj, kernel_indices,
            lambda p, q: renyi_divergence(p, q, 0.5))
        renyi_2 = compute_pairwise_divergence(
            nmf_obj, kernel_indices,
            lambda p, q: renyi_divergence(p, q, 2.0))
        renyi_inf = compute_pairwise_divergence(
            nmf_obj, kernel_indices,
            lambda p, q: renyi_divergence(p, q, float('inf')))

        v_matrix = compute_pairwise_cramers_v(learner, kernel_names)

        # FW posteriors (leave-one-out)
        print("Computing FW leave-one-out posteriors...")
        fw_post, fw_resid = compute_fw_posteriors(nmf_obj, kernel_indices)

        # Posterior entropy: H(FW weights) for each kernel
        fw_entropy = np.zeros(len(kernel_names))
        fw_max_weight = np.zeros(len(kernel_names))
        for i in range(len(kernel_names)):
            w = fw_post[i, :]
            w = w[w > 1e-10]
            fw_entropy[i] = -np.sum(w * np.log(w)) if len(w) > 0 else 0
            fw_max_weight[i] = np.max(fw_post[i, :])

        # ---- Figure 1: Asymmetric KL + Renyi heatmaps ----
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(
            f'Asymmetric divergences ({feature_mode}): {n_nt} NTs + {n_extra} extra '
            f'(seed={grammar_seed}, N={n_sentences//1000}K)\n'
            f'Row = P, Column = Q.  Blue box = genuine kernels.',
            fontsize=13, fontweight='bold')

        plot_heatmap(axes[0, 0], kl_fwd, tick_labels,
                     'KL(row || col)\n(cluster features)', n_nt)
        plot_heatmap(axes[0, 1], kl_rev, tick_labels,
                     'KL(col || row)\n(cluster features)', n_nt)

        # Asymmetry: KL(row||col) - KL(col||row)
        kl_asym = np.where(
            np.isfinite(kl_fwd) & np.isfinite(kl_rev),
            kl_fwd - kl_rev, np.nan)
        plot_heatmap(axes[0, 2], kl_asym, tick_labels,
                     'KL asymmetry: KL(row||col) - KL(col||row)',
                     n_nt, cmap='RdBu_r', vmin=None, vmax=None)

        plot_heatmap(axes[1, 0], renyi_half, tick_labels,
                     r'Renyi $D_{0.5}$(row || col)', n_nt)
        plot_heatmap(axes[1, 1], renyi_2, tick_labels,
                     r'Renyi $D_2$(row || col)', n_nt)
        plot_heatmap(axes[1, 2], renyi_inf, tick_labels,
                     r'Renyi $D_\infty$(row || col)', n_nt)

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        out1 = output_path.replace('.png', '_asymmetric.png')
        plt.savefig(out1, dpi=150, bbox_inches='tight')
        print(f"Asymmetric divergence plot saved to {out1}")
        plt.close()

        # ---- Figure 2: FW posterior analysis ----
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle(
            f'FW posterior analysis ({feature_mode}): {n_nt} NTs + {n_extra} extra '
            f'(seed={grammar_seed}, N={n_sentences//1000}K)',
            fontsize=13, fontweight='bold')

        # FW posterior heatmap
        ax = axes[0]
        im = ax.imshow(fw_post, cmap='YlOrRd', aspect='equal',
                       interpolation='nearest', vmin=0, vmax=1)
        ax.set_xticks(range(len(kernel_names)))
        ax.set_xticklabels(tick_labels, fontsize=6, rotation=45, ha='right')
        ax.set_yticks(range(len(kernel_names)))
        ax.set_yticklabels(tick_labels, fontsize=6)
        ax.set_title('Leave-one-out FW weights\n(row = kernel, col = weight on other)',
                      fontsize=10)
        ax.set_xlabel('Weight on kernel')
        ax.set_ylabel('Kernel being decomposed')
        rect = plt.Rectangle((-0.5, -0.5), n_nt, n_nt,
                             linewidth=2.5, edgecolor='blue',
                             facecolor='none', linestyle='--')
        ax.add_patch(rect)
        plt.colorbar(im, ax=ax, shrink=0.8)
        for i in range(len(kernel_names)):
            for j in range(len(kernel_names)):
                val = fw_post[i, j]
                if val > 0.01:
                    fs = 5 if len(kernel_names) <= 12 else 4
                    color = 'white' if val > 0.5 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                            fontsize=fs, color=color)

        # Entropy and max weight bar chart
        ax = axes[1]
        x = np.arange(len(kernel_names))
        colors = ['#2196F3' if i < n_nt else '#F44336'
                  for i in range(len(kernel_names))]
        bars = ax.bar(x, fw_entropy, color=colors, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{labels[i]}\n{kernel_names[i]}'
                            for i in range(len(kernel_names))],
                           fontsize=6, rotation=45, ha='right')
        ax.set_ylabel('Entropy of FW posterior (nats)')
        ax.set_title('FW posterior entropy\n(low = peaked, high = diffuse)')
        ax.axvline(x=n_nt - 0.5, color='black', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.2, axis='y')
        # Add max_weight as text
        for i, (bar, ent, mw) in enumerate(
                zip(bars, fw_entropy, fw_max_weight)):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f'{mw:.2f}', ha='center', va='bottom', fontsize=6)
        ax.text(0.02, 0.95, 'Numbers = max FW weight',
                transform=ax.transAxes, fontsize=8, va='top')

        # FW residual bar chart
        ax = axes[2]
        bars = ax.bar(x, fw_resid, color=colors, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{labels[i]}\n{kernel_names[i]}'
                            for i in range(len(kernel_names))],
                           fontsize=6, rotation=45, ha='right')
        ax.set_ylabel('FW residual distance')
        ax.set_title('FW leave-one-out residual\n(high = unique, low = redundant)')
        ax.axvline(x=n_nt - 0.5, color='black', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.2, axis='y')
        # Add word count as text
        for i, bar in enumerate(bars):
            count = int(nmf_obj.counts[kernel_indices[i]])
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.001,
                    f'n={count}', ha='center', va='bottom', fontsize=5)

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        out2 = output_path.replace('.png', '_fw_posterior.png')
        plt.savefig(out2, dpi=150, bbox_inches='tight')
        print(f"FW posterior plot saved to {out2}")
        plt.close()

        # ---- Print summary table ----
        print(f"\n{'':>3s} {'Kernel':>6s} {'Label':>5s} {'Count':>6s} "
              f"{'FW_resid':>8s} {'H(FW)':>6s} {'maxW':>5s} "
              f"{'minKL>':>7s} {'minKL<':>7s} {'minR2>':>7s} {'minR2<':>7s} "
              f"{'minV':>6s} {'closest':>8s}")
        print("-" * 100)
        for i in range(len(kernel_names)):
            count = int(nmf_obj.counts[kernel_indices[i]])
            # Min divergence to any other kernel
            min_kl_fwd = float('inf')
            min_kl_rev = float('inf')
            min_r2_fwd = float('inf')
            min_r2_rev = float('inf')
            min_v = float('inf')
            closest = ''
            closest_val = float('inf')
            for j in range(len(kernel_names)):
                if i == j:
                    continue
                if kl_fwd[i, j] < min_kl_fwd:
                    min_kl_fwd = kl_fwd[i, j]
                if kl_rev[i, j] < min_kl_rev:
                    min_kl_rev = kl_rev[i, j]
                if renyi_2[i, j] < min_r2_fwd:
                    min_r2_fwd = renyi_2[i, j]
                if renyi_2[j, i] < min_r2_rev:
                    min_r2_rev = renyi_2[j, i]
                if np.isfinite(v_matrix[i, j]) and v_matrix[i, j] < min_v:
                    min_v = v_matrix[i, j]
                # Closest by L2
                l2_d = np.linalg.norm(
                    nmf_obj.data[kernel_indices[i], :] -
                    nmf_obj.data[kernel_indices[j], :])
                if l2_d < closest_val:
                    closest_val = l2_d
                    closest = kernel_names[j]

            marker = ' *' if i >= n_nt else ''
            print(f"{i:3d} {kernel_names[i]:>6s} {labels[i]:>5s} {count:6d} "
                  f"{fw_resid[i]:8.5f} {fw_entropy[i]:6.3f} {fw_max_weight[i]:5.3f} "
                  f"{min_kl_fwd:7.4f} {min_kl_rev:7.4f} "
                  f"{min_r2_fwd:7.4f} {min_r2_rev:7.4f} "
                  f"{min_v:6.4f} {closest:>8s}{marker}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualise pairwise kernel divergences')
    parser.add_argument('--nt', type=int, default=10,
                        help='True number of nonterminals')
    parser.add_argument('--sentences', type=int, default=100000,
                        help='Corpus size')
    parser.add_argument('--grammar-seed', type=int, default=71,
                        help='Grammar random seed')
    parser.add_argument('--corpus-seed', type=int, default=42,
                        help='Corpus random seed')
    parser.add_argument('--extra', type=int, default=5,
                        help='Number of extra kernels beyond true count')
    parser.add_argument('--output', default='/tmp/kernel_divergences.png',
                        help='Output plot path')
    parser.add_argument('--feature-mode', default='marginal',
                        choices=['marginal', 'joint'],
                        help='NMF feature mode (default: marginal)')
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    run_and_visualise(
        n_nt=args.nt,
        n_sentences=args.sentences,
        grammar_seed=args.grammar_seed,
        corpus_seed=args.corpus_seed,
        n_extra=args.extra,
        output_path=args.output,
        feature_mode=args.feature_mode,
    )
