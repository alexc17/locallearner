#!/usr/bin/env python3
"""Visualize the antichain anchor-selection algorithm.

Generates a diagram showing each selected anchor as a cluster, with
satellite words that were classified as 'equal' (below) or 'larger'
(above). Each word is drawn as a mini pie chart of its posterior
distribution P(NT|word), sized by frequency.

Usage:
    python plot_antichain.py <grammar_dir> <output.png> [--json log.json]

    grammar_dir: directory containing grammar.pcfg, corpus.txt,
                 and rnn_cloze.pt (trained RNN model).
    output.png:  path to save the output image.
    --json:      optionally save/load the antichain decision log as JSON.
                 If the file exists, it is loaded instead of recomputed.

Example:
    python plot_antichain.py tmp/jm_experiment/g007 antichain_g007.png
"""
import argparse
import json
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Wedge, Circle
import numpy as np

import wcfg
from neural_learner import NeuralLearner
from run_gold_pipeline import get_gold_kernels

NT_COLORS = {
    'NT1': '#e57373', 'NT2': '#81c784', 'NT3': '#64b5f6',
    'NT4': '#ffb74d', 'NT5': '#ba68c8', 'NT6': '#4dd0e1',
    'NT7': '#f06292', 'NT8': '#aed581', 'NT9': '#b39ddb',
    'S': '#000000',
}
NT_LIGHT = {
    'NT1': '#ffcdd2', 'NT2': '#c8e6c9', 'NT3': '#bbdefb',
    'NT4': '#ffe0b2', 'NT5': '#e1bee7', 'NT6': '#b2ebf2',
    'NT7': '#f8bbd0', 'NT8': '#dcedc8', 'NT9': '#d1c4e9',
}


def compute_posteriors(grammar):
    """Compute P(NT|word) for every terminal, with S as remainder."""
    te = grammar.terminal_expectations()
    pe = grammar.production_expectations()
    non_start = sorted(
        nt for nt in grammar.nonterminals if nt != grammar.start)

    posteriors = {}
    for w in grammar.terminals:
        ea = te.get(w, 0.0)
        if ea <= 0:
            continue
        post = {}
        nt_total = 0.0
        for nt in non_start:
            ep = pe.get((nt, w), 0.0)
            if ep > 0:
                post[nt] = ep / ea
                nt_total += ep / ea
        if nt_total < 1.0:
            post['S'] = 1.0 - nt_total
        posteriors[w] = post
    return posteriors


def compute_word_to_nt(grammar):
    """Map each terminal to its dominant non-start NT."""
    te = grammar.terminal_expectations()
    pe = grammar.production_expectations()
    non_start = sorted(
        nt for nt in grammar.nonterminals if nt != grammar.start)
    mapping = {}
    for w in grammar.terminals:
        ea = te.get(w, 0.0)
        if ea <= 0:
            continue
        best_nt, best_post = None, 0
        for nt in non_start:
            ep = pe.get((nt, w), 0.0)
            if ep / ea > best_post:
                best_post = ep / ea
                best_nt = nt
        if best_nt:
            mapping[w] = best_nt
    return mapping


def log_antichain(grammar_dir, max_terminals=300):
    """Run the antichain algorithm with full decision logging.

    Returns a dict with keys: grammar, sigma, n_candidates, n_gold,
    gold_anchors, final_anchors, terminal_info, decisions.
    """
    grammar_path = os.path.join(grammar_dir, 'grammar.pcfg')
    corpus_path = os.path.join(grammar_dir, 'corpus.txt')
    model_path = os.path.join(grammar_dir, 'rnn_cloze.pt')

    target = wcfg.load_wcfg_from_file(grammar_path)
    gold_anchors = get_gold_kernels(target)
    gold_set = set(gold_anchors)
    word_to_nt = compute_word_to_nt(target)

    nl = NeuralLearner(corpus_path)
    nl.model_type = 'rnn'
    nl.single_model_file = model_path
    nl.n_epochs = 3
    nl.n_context_samples = 500
    nl.train_single_model(verbose=False)

    terminals = sorted(nl.vocab, key=lambda w: -nl.word_counts.get(w, 0))
    terminals = [w for w in terminals if w in nl.w2i][:max_terminals]
    terminal_log_p = nl._compute_terminal_log_probs(terminals)
    candidates = [w for w in terminals if w in terminal_log_p]

    epsilon = 1.5
    sigma = nl._estimate_noise_sigma(terminal_log_p, candidates, epsilon)
    vf_floor = 0.10

    def is_subset(a, b):
        fa = nl.word_counts.get(a, 1)
        fb = nl.word_counts.get(b, 1)
        thresh = max(
            nl._adaptive_vf_threshold(sigma, fa, fb, epsilon), vf_floor)
        vf = nl._violation_fraction(terminal_log_p, a, b, epsilon)
        return vf <= thresh, vf, thresh

    def is_larger_than_S(a):
        E_a = nl.E_sent.get(a, 0)
        E1_a = nl.E_length1.get(a, 0)
        return E_a > 0 and E1_a / E_a > 1e-3

    terminal_info = {}
    for w in candidates:
        terminal_info[w] = {
            'word': w,
            'freq': nl.word_counts.get(w, 0),
            'nt': word_to_nt.get(w, '?'),
            'gold': w in gold_set,
        }

    antichain = ['<S>']
    decisions = []

    for step, a in enumerate(candidates, 1):
        comparisons = []
        smaller_than, equal_to, larger_than = [], [], []

        for b in antichain:
            if b == '<S>':
                if is_larger_than_S(a):
                    larger_than.append(b)
                    comparisons.append({
                        'with': b, 'relation': 'a > S',
                        'vf_ab': None, 'vf_ba': None, 'thresh': None,
                    })
                continue

            a_sub_b, vf_ab, thresh_ab = is_subset(a, b)
            b_sub_a, vf_ba, thresh_ba = is_subset(b, a)

            comp = {
                'with': b,
                'vf_ab': round(float(vf_ab), 4),
                'vf_ba': round(float(vf_ba), 4),
                'thresh_ab': round(float(thresh_ab), 4),
                'thresh_ba': round(float(thresh_ba), 4),
                'a_sub_b': bool(a_sub_b),
                'b_sub_a': bool(b_sub_a),
            }

            if a_sub_b and not b_sub_a:
                comp['relation'] = 'a < b'
                smaller_than.append(b)
            elif a_sub_b and b_sub_a:
                comp['relation'] = 'a = b'
                equal_to.append(b)
            elif not a_sub_b and b_sub_a:
                comp['relation'] = 'a > b'
                larger_than.append(b)
            else:
                comp['relation'] = 'incomparable'

            comparisons.append(comp)

        if equal_to:
            action, reason_word, replaces = 'skip_equal', equal_to[0], []
        elif larger_than:
            action, reason_word, replaces = 'skip_larger', larger_than[0], []
        elif smaller_than:
            action, reason_word = 'replace', None
            replaces = list(smaller_than)
            for b in smaller_than:
                antichain.remove(b)
            antichain.append(a)
        else:
            action, reason_word, replaces = 'add_incomparable', None, []
            antichain.append(a)

        decisions.append({
            'step': step, 'word': a,
            'freq': nl.word_counts.get(a, 0),
            'nt': word_to_nt.get(a, '?'),
            'gold': a in gold_set,
            'action': action, 'reason_word': reason_word,
            'replaces': replaces, 'comparisons': comparisons,
            'antichain_after': [w for w in antichain if w != '<S>'],
            'antichain_size': len([w for w in antichain if w != '<S>']),
        })

    return {
        'grammar': os.path.basename(grammar_dir),
        'sigma': round(sigma, 1),
        'n_candidates': len(candidates),
        'n_gold': len(gold_anchors),
        'gold_anchors': gold_anchors,
        'final_anchors': [w for w in antichain if w != '<S>'],
        'terminal_info': terminal_info,
        'decisions': decisions,
    }


def draw_pie(ax, cx, cy, radius, posteriors, linewidth=0.5):
    """Draw a mini pie chart of P(NT|word) at (cx, cy)."""
    if not posteriors:
        ax.add_patch(Circle((cx, cy), radius, facecolor='#e0e0e0',
                            edgecolor='#999999', linewidth=linewidth))
        return
    items = sorted(posteriors.items(), key=lambda x: -x[1])
    total = sum(v for _, v in items)
    if total <= 0:
        ax.add_patch(Circle((cx, cy), radius, facecolor='#e0e0e0',
                            edgecolor='#999999', linewidth=linewidth))
        return
    angle = 90
    for nt, val in items:
        frac = val / total
        sweep = frac * 360
        if sweep < 0.5:
            continue
        color = NT_COLORS.get(nt, '#bdbdbd')
        ax.add_patch(Wedge((cx, cy), radius, angle - sweep, angle,
                           facecolor=color, edgecolor='white',
                           linewidth=linewidth * 0.3))
        angle -= sweep
    ax.add_patch(Circle((cx, cy), radius, facecolor='none',
                        edgecolor='#333333', linewidth=linewidth))


def plot_antichain(data, grammar, output_path, n_equal=4, n_larger=4):
    """Render the antichain diagram to a PNG file.

    Args:
        data: decision log dict from log_antichain().
        grammar: loaded WCFG (for computing posteriors).
        output_path: where to save the PNG.
        n_equal: max 'equal' satellites to show per anchor.
        n_larger: max 'larger' words to show per anchor.
    """
    non_start = sorted(
        nt for nt in grammar.nonterminals if nt != grammar.start)
    word_posteriors = compute_posteriors(grammar)

    decisions = data['decisions']
    terminal_info = data['terminal_info']
    gold_anchors_set = set(data['gold_anchors'])
    final_anchors = data['final_anchors']

    word_dec = {}
    for d in decisions:
        word_dec[d['word']] = d

    anchor_equals = {a: [] for a in final_anchors}
    anchor_equals['<S>'] = []
    anchor_largers = {a: [] for a in final_anchors}
    anchor_largers['<S>'] = []

    for w, d in word_dec.items():
        tgt = d.get('reason_word')
        if d['action'] == 'skip_equal' and tgt in anchor_equals:
            anchor_equals[tgt].append(w)
        elif d['action'] == 'skip_larger' and tgt in anchor_largers:
            anchor_largers[tgt].append(w)

    for a in anchor_equals:
        anchor_equals[a].sort(key=lambda w: -word_dec[w]['freq'])
    for a in anchor_largers:
        anchor_largers[a].sort(key=lambda w: -word_dec[w]['freq'])

    anchor_n_eq = {a: len(anchor_equals[a]) for a in final_anchors}
    anchor_n_lg = {a: len(anchor_largers[a]) for a in final_anchors}

    for a in anchor_equals:
        anchor_equals[a] = anchor_equals[a][:n_equal]
    for a in anchor_largers:
        anchor_largers[a] = anchor_largers[a][:n_larger]

    all_clusters = final_anchors + ['<S>']
    n_cols = 3
    n_rows = math.ceil(len(all_clusters) / n_cols)

    PIE_R_ANCHOR = 0.32
    PIE_R_SAT = 0.18
    PIE_R_LG = 0.20

    fig_w = n_cols * 4.2
    fig_h = n_rows * 4.5
    fig, axes_arr = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
    axes = np.array(axes_arr).flatten()

    for ax in axes:
        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(-2.0, 1.8)
        ax.set_aspect('equal')
        ax.axis('off')

    for idx, a in enumerate(all_clusters):
        ax = axes[idx]

        if a == '<S>':
            draw_pie(ax, 0, 0, PIE_R_ANCHOR, {'S': 1.0}, linewidth=1.5)
            ax.text(0, 0, 'S', ha='center', va='center',
                    fontsize=10, fontweight='bold', color='white')

            largers = anchor_largers.get('<S>', [])
            for mi, w in enumerate(largers):
                d = word_dec[w]
                bx = (mi - len(largers) / 2 + 0.5) * 0.85
                by = 1.1
                draw_pie(ax, bx, by, PIE_R_LG, word_posteriors.get(w, {}))
                mark = ' *' if d['gold'] else ''
                ax.text(bx, by - PIE_R_LG - 0.06,
                        f'{w}{mark}\n{d["freq"]:,}',
                        ha='center', va='top', fontsize=5.5,
                        color='#333333')
                ax.annotate('', xy=(0, PIE_R_ANCHOR),
                           xytext=(bx, by - PIE_R_LG),
                           arrowprops=dict(arrowstyle='->',
                                          color='#cc0000',
                                          linestyle='dashed', lw=1.0))

            n_s_lg = sum(1 for w, d in word_dec.items()
                         if d['action'] == 'skip_larger'
                         and d['reason_word'] == '<S>')
            ax.set_title(f'S (boundary, {n_s_lg} larger)',
                         fontsize=9, color='#444444')
            continue

        info = terminal_info.get(a, {})
        nt = info.get('nt', '?')
        freq = info.get('freq', 0)
        is_gold = a in gold_anchors_set
        color = NT_COLORS.get(nt, '#bdbdbd')
        light = NT_LIGHT.get(nt, '#e0e0e0')

        ax.add_patch(FancyBboxPatch(
            (-1.9, -1.9), 3.8, 3.6,
            boxstyle="round,pad=0.1",
            facecolor=light, edgecolor=color, linewidth=1.2, alpha=0.25))

        draw_pie(ax, 0, 0, PIE_R_ANCHOR, word_posteriors.get(a, {}),
                 linewidth=2.0)
        if is_gold:
            ax.add_patch(Circle((0, 0), PIE_R_ANCHOR + 0.05,
                                facecolor='none', edgecolor='black',
                                linewidth=1.5))
        ax.text(0, 0, a, ha='center', va='center',
                fontsize=9, fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.08', facecolor='black',
                         alpha=0.4, edgecolor='none'))
        ax.text(0, -PIE_R_ANCHOR - 0.08, f'f={freq:,}', ha='center',
                va='top', fontsize=6.5, color='#333333')

        # "Larger" words above
        largers = anchor_largers.get(a, [])
        for mi, w in enumerate(largers):
            d = word_dec[w]
            bx = (mi - len(largers) / 2 + 0.5) * 0.85
            by = 1.15
            draw_pie(ax, bx, by, PIE_R_LG, word_posteriors.get(w, {}),
                     linewidth=1.5 if d['gold'] else 0.8)
            if d['gold']:
                ax.add_patch(Circle((bx, by), PIE_R_LG + 0.04,
                                    facecolor='none', edgecolor='red',
                                    linewidth=2.0, linestyle='dotted'))
            mark = ' GOLD' if d['gold'] else ''
            ax.text(bx, by - PIE_R_LG - 0.06,
                    f'{w}{mark}\n{d["freq"]:,}',
                    ha='center', va='top', fontsize=5.5,
                    color='red' if d['gold'] else '#333333',
                    fontweight='bold' if d['gold'] else 'normal')
            ax.annotate('', xy=(0, PIE_R_ANCHOR + 0.02),
                       xytext=(bx, by - PIE_R_LG),
                       arrowprops=dict(arrowstyle='->',
                                      color='#cc0000',
                                      linestyle='--', lw=1.2))

        # "Equal" words below
        equals = anchor_equals.get(a, [])
        for mi, w in enumerate(equals):
            d = word_dec[w]
            row = mi // 3
            col = mi % 3
            n_this = min(3, len(equals) - row * 3)
            bx = (col - n_this / 2 + 0.5) * 0.85
            by = -0.85 - row * 0.65
            draw_pie(ax, bx, by, PIE_R_SAT, word_posteriors.get(w, {}))
            ax.text(bx, by - PIE_R_SAT - 0.05,
                    f'{w}\n{d["freq"]:,}',
                    ha='center', va='top', fontsize=5, color='#555555')
            ax.annotate('', xy=(0, -PIE_R_ANCHOR),
                       xytext=(bx, by + PIE_R_SAT),
                       arrowprops=dict(arrowstyle='->', color=color,
                                      lw=0.6, alpha=0.4))

        gold_str = 'gold' if is_gold else 'spurious'
        ax.set_title(
            f'{nt} ({gold_str}, {anchor_n_eq.get(a, 0)} equal, '
            f'{anchor_n_lg.get(a, 0)} larger)',
            fontsize=8, color=color, fontweight='bold')

    for idx in range(len(all_clusters), len(axes)):
        axes[idx].set_visible(False)

    legend_els = []
    for nt in non_start:
        legend_els.append(plt.Line2D(
            [0], [0], marker='o', color='w',
            markerfacecolor=NT_COLORS.get(nt, '#bdbdbd'),
            markersize=8, label=nt))
    legend_els.append(plt.Line2D(
        [0], [0], marker='o', color='w',
        markerfacecolor='black', markersize=8, label='S'))
    legend_els.append(plt.Line2D(
        [0], [0], color='#cc0000', linestyle='--', lw=1.2,
        label='larger \u2192 smaller'))
    legend_els.append(plt.Line2D(
        [0], [0], color='#999999', lw=0.8,
        label='equal \u2192 anchor'))

    fig.legend(handles=legend_els, loc='lower center',
               ncol=6, fontsize=7.5,
               frameon=True, fancybox=True, framealpha=0.8)

    gname = data.get('grammar', '')
    fig.suptitle(
        f'{gname} Antichain \u2014 '
        f'Word Posterior Distributions P(NT|word)\n'
        'Above: words classified as LARGER (\u2283 anchor).  '
        'Below: words classified as EQUAL (\u2248 anchor).\n'
        'Double ring = gold anchor.  '
        'Red dotted ring = missed gold anchor.',
        fontsize=11, fontweight='bold', y=1.0)
    plt.tight_layout(rect=[0, 0.05, 1, 0.92])
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)
    print(f'Saved {output_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Visualize the antichain anchor-selection algorithm.')
    parser.add_argument('grammar_dir',
                        help='Directory with grammar.pcfg, corpus.txt, '
                             'rnn_cloze.pt')
    parser.add_argument('output', help='Output PNG path')
    parser.add_argument('--json', default=None,
                        help='Save/load decision log as JSON')
    parser.add_argument('--max-terminals', type=int, default=300)
    args = parser.parse_args()

    grammar_path = os.path.join(args.grammar_dir, 'grammar.pcfg')
    grammar = wcfg.load_wcfg_from_file(grammar_path)

    if args.json and os.path.exists(args.json):
        print(f'Loading decisions from {args.json}')
        with open(args.json) as f:
            data = json.load(f)
    else:
        print(f'Running antichain on {args.grammar_dir}...')
        data = log_antichain(args.grammar_dir, args.max_terminals)
        if args.json:
            with open(args.json, 'w') as f:
                json.dump(data, f, indent=2)
            print(f'Saved decisions to {args.json}')

    plot_antichain(data, grammar, args.output)


if __name__ == '__main__':
    main()
