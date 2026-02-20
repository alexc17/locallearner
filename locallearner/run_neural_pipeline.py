#!/usr/bin/env python3
"""Full pipeline for a single grammar: neural kernel finding -> param estimation -> SGD -> eval.

Unlike run_gold_pipeline.py which uses gold (oracle) kernels, this discovers
kernels from scratch using the antichain-based anchor selection algorithm
(NeuralLearner.select_anchors_minimal).

The pipeline proceeds to parameter estimation and evaluation ONLY if the
discovered number of nonterminals matches the target grammar.

Usage:
    python run_neural_pipeline.py --config config.json g007
    python run_neural_pipeline.py --config config.json --all
    python run_neural_pipeline.py --config config.json --kernels-only g005
"""
import argparse
import glob
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
from scipy.optimize import linear_sum_assignment

import wcfg
import evaluation
import neural_learner as nl_module
from run_sgd_io import run_epoch
from run_gold_pipeline import get_gold_kernels

DEFAULT_BASE = '/Users/alexc/research/cursor_locallearner/locallearner/tmp/jm_experiment'

DEFAULTS = {
    'basedir': DEFAULT_BASE,
    'corpus': {'n_sentences': 1000000, 'seed': 1},
    'training': {
        'n_epochs': 10, 'embedding_dim': 64, 'hidden_dim': 128,
        'batch_size': 4096, 'seed': 42,
    },
    'pipeline': {
        'model_type': 'rnn', 'pos_k': 2, 'ml_maxlength': 10,
        'maxlength': 15, 'maxcount': 1000, 'stepsize': 0.5,
        'eval_corpus_size': 100000, 'max_terminals': 300,
        'epsilon': 1.5, 'pipeline_epochs': 3,
    },
}


def load_config(config_path):
    """Load experiment config, merging with defaults."""
    with open(config_path) as f:
        cfg = json.load(f)
    merged = dict(DEFAULTS)
    for section in ('corpus', 'training', 'pipeline'):
        merged[section] = dict(DEFAULTS.get(section, {}))
        merged[section].update(cfg.get(section, {}))
    if 'basedir' in cfg:
        merged['basedir'] = cfg['basedir']
    return merged


def _prepare_hypothesis(target, hyp):
    """Renormalize, relabel to match target NTs, return ready hypothesis."""
    h = hyp.copy()
    h.renormalise_locally()
    h.set_log_parameters()
    if set(target.nonterminals) != set(h.nonterminals):
        eval_kwargs = {'max_length': 20, 'seed': 42, 'samples': 1000}
        mapping = evaluation.estimate_bijection(target, h, **eval_kwargs)
        h = h.relabel({a: b for b, a in mapping.items()})
    return h


def _extract_scores(scores, idx):
    """Extract per-hypothesis metrics from combined parseval scores."""
    denom = scores['trees_denominator']
    lab_d = scores['labeled_denominator']
    ulab_d = scores['unlabeled_denominator']
    tag = f'hypothesis{idx}'
    return {
        'labeled_exact': scores[f'original:{tag}:labeled:exact_match'] / denom,
        'unlabeled_exact': scores[f'original:{tag}:unlabeled:exact_match'] / denom,
        'labeled_micro': scores[f'original:{tag}:labeled:microaveraged'] / lab_d,
        'unlabeled_micro': scores[f'original:{tag}:unlabeled:microaveraged'] / ulab_d,
    }


def run_pipeline(grammar_id, cfg, kernels_only=False):
    """Run full neural pipeline for one grammar.

    Returns dict of results, including whether the NT count matched.
    """
    base_dir = cfg['basedir']
    p = cfg['pipeline']
    maxlength = p['maxlength']
    maxcount = p['maxcount']
    stepsize = p['stepsize']
    n_corpus = p['eval_corpus_size']
    n_epochs = p['pipeline_epochs']
    max_terminals = p['max_terminals']
    epsilon = p['epsilon']

    gdir = os.path.join(base_dir, grammar_id)
    grammar_path = os.path.join(gdir, 'grammar.pcfg')
    corpus_path = os.path.join(gdir, 'corpus.txt')
    model_path = os.path.join(gdir, 'rnn_cloze.pt')
    ml_path = os.path.join(gdir, 'ml.pcfg')

    for path, label in [(grammar_path, 'Grammar'), (corpus_path, 'Corpus'),
                         (model_path, 'RNN model')]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} not found: {path}")

    target = wcfg.load_wcfg_from_file(grammar_path)
    n_target_nts = len(target.nonterminals)
    gold_anchors = get_gold_kernels(target)
    print(f"[{grammar_id}] Target: {n_target_nts} NTs "
          f"({n_target_nts - 1} non-start), "
          f"{len(target.terminals)} terminals")

    # 1. Discover kernels
    t0 = time.time()
    nl = nl_module.NeuralLearner(corpus_path)
    nl.model_type = 'rnn'
    nl.single_model_file = model_path
    nl.n_epochs = n_epochs
    nl.n_context_samples = 500
    nl.train_single_model(verbose=False)

    anchors = nl.select_anchors_minimal(
        max_terminals=max_terminals, epsilon=epsilon, verbose=True)
    kernel_time = time.time() - t0

    n_found = len(anchors) + 1  # +1 for S
    nt_match = (n_found == n_target_nts)

    gold_set = set(gold_anchors)
    n_gold_recalled = sum(1 for a in anchors if a in gold_set)

    results = {
        'grammar_id': grammar_id,
        'n_target_nts': n_target_nts,
        'n_found_nts': n_found,
        'nt_match': nt_match,
        'anchors': anchors,
        'gold_anchors': gold_anchors,
        'gold_recall': n_gold_recalled / len(gold_anchors) if gold_anchors else 0,
        'kernel_time': round(kernel_time, 1),
    }

    print(f"[{grammar_id}] Found {n_found} NTs (target {n_target_nts}), "
          f"gold recall {n_gold_recalled}/{len(gold_anchors)}, "
          f"{kernel_time:.1f}s")

    if kernels_only:
        results['status'] = 'kernels_only'
        _save_results(results, gdir)
        return results

    if not nt_match:
        print(f"[{grammar_id}] NT count mismatch "
              f"({n_found} != {n_target_nts}), skipping evaluation.")
        results['status'] = 'nt_mismatch'
        _save_results(results, gdir)
        return results

    # 2. Estimate parameters and build WCFG
    t1 = time.time()
    nl.train_pair_model(verbose=True)
    nl.estimate_lexical_xi(verbose=False)
    nl.estimate_binary_xi(verbose=False)
    nl.build_wcfg(verbose=False)

    nl.xi_wcfg.store(os.path.join(gdir, 'neural_init.wcfg'))
    init_pcfg = nl.convert_to_pcfg(verbose=False)
    init_pcfg.store(os.path.join(gdir, 'neural_init.pcfg'))
    param_time = time.time() - t1
    print(f"[{grammar_id}] Init PCFG: E[len]={init_pcfg.expected_length():.3f}, "
          f"{len(init_pcfg.terminals)} terminals, {param_time:.1f}s")

    # 3. Prepare corpus subset for SGD
    corpus_subset = os.path.join(gdir, f'corpus_{n_corpus // 1000}k.txt')
    if not os.path.exists(corpus_subset):
        with open(corpus_path) as inf, open(corpus_subset, 'w') as outf:
            for i, line in enumerate(inf):
                if i >= n_corpus:
                    break
                outf.write(line)

    # 4. One epoch of SGD (linear)
    t2 = time.time()
    sgd_pcfg = run_epoch(
        init_pcfg, corpus_subset,
        maxlength=maxlength, maxcount=maxcount, stepsize=stepsize,
        update='linear', verbose=0)
    sgd_time = time.time() - t2
    sgd_pcfg.store(os.path.join(gdir, 'neural_sgd.pcfg'))
    print(f"[{grammar_id}] SGD: {sgd_time:.1f}s, "
          f"E[len]={sgd_pcfg.expected_length():.3f}")

    # 5. Evaluate all hypotheses together (same sampled trees)
    eval_kwargs = {'max_length': 20, 'seed': 42, 'samples': 1000}

    # Collect hypotheses: ML (if available), init, SGD
    hyp_labels = []
    hyp_list = []

    if os.path.exists(ml_path):
        ml_pcfg = wcfg.load_wcfg_from_file(ml_path)
        ml_prep = _prepare_hypothesis(target, ml_pcfg)
        hyp_labels.append('ml')
        hyp_list.append(ml_prep)
        print(f"[{grammar_id}] ML PCFG: E[len]={ml_pcfg.expected_length():.3f}, "
              f"{len(ml_pcfg.terminals)} terminals")

    init_prep = _prepare_hypothesis(target, init_pcfg)
    hyp_labels.append('init')
    hyp_list.append(init_prep)

    sgd_prep = _prepare_hypothesis(target, sgd_pcfg)
    hyp_labels.append('sgd')
    hyp_list.append(sgd_prep)

    # Single evaluation call: all hypotheses share the same sampled trees
    scores = evaluation.do_parseval_monte_carlo(
        target, hyp_list, **eval_kwargs)

    # Extract per-hypothesis results
    for i, label in enumerate(hyp_labels):
        hyp_scores = _extract_scores(scores, i)
        orig_pcfg = {'ml': ml_pcfg, 'init': init_pcfg, 'sgd': sgd_pcfg}[label]
        kld = evaluation.smoothed_kld_exact(
            target, orig_pcfg, compute_bijection=True)
        hyp_scores['kld'] = kld
        hyp_scores['elen'] = orig_pcfg.expected_length()
        hyp_scores['terminals'] = len(orig_pcfg.terminals)
        results[label] = hyp_scores

    # Target ceiling (gold viterbi vs original)
    td = scores['trees_denominator']
    tld = scores['labeled_denominator']
    tud = scores['unlabeled_denominator']
    results['target'] = {
        'labeled_exact': scores['original:gold viterbi:labeled:exact_match'] / td,
        'unlabeled_exact': scores['original:gold viterbi:unlabeled:exact_match'] / td,
        'labeled_micro': scores['original:gold viterbi:labeled:microaveraged'] / tld,
        'unlabeled_micro': scores['original:gold viterbi:unlabeled:microaveraged'] / tud,
        'kld': 0.0,
        'elen': target.expected_length(),
        'terminals': len(target.terminals),
    }

    results.update({
        'status': 'evaluated',
        'param_time': round(param_time, 1),
        'sgd_time': round(sgd_time, 1),
    })

    # Print summary line for each hypothesis
    for label in hyp_labels:
        lm = results[label]['labeled_micro']
        kld = results[label]['kld']
        print(f"[{grammar_id}] {label:>4}: lab_micro={lm:.4f}, kld={kld:.4f}")
    print(f"[{grammar_id}] target ceiling: "
          f"lab_micro={results['target']['labeled_micro']:.4f}")

    _save_results(results, gdir)
    return results


def _save_results(results, gdir):
    out_path = os.path.join(gdir, 'neural_pipeline_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[{results['grammar_id']}] Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Neural pipeline: anchor discovery -> '
                    'param estimation -> SGD -> eval')
    parser.add_argument('grammar_ids', nargs='*',
                        help='Grammar IDs (e.g. g000 g007)')
    parser.add_argument('--config', default=None,
                        help='JSON config file')
    parser.add_argument('--all', action='store_true',
                        help='Run all g??? grammars in base dir')
    parser.add_argument('--base', type=str, default=None,
                        help='Base experiment directory (overrides config)')
    parser.add_argument('--kernels-only', action='store_true',
                        help='Only discover kernels, skip param estimation '
                             'and evaluation')

    args = parser.parse_args()

    if args.config:
        cfg = load_config(args.config)
    else:
        cfg = dict(DEFAULTS)
        cfg['corpus'] = dict(DEFAULTS['corpus'])
        cfg['training'] = dict(DEFAULTS['training'])
        cfg['pipeline'] = dict(DEFAULTS['pipeline'])

    if args.base:
        cfg['basedir'] = args.base

    if args.all:
        dirs = sorted(glob.glob(os.path.join(cfg['basedir'], 'g???')))
        grammar_ids = [os.path.basename(d) for d in dirs]
    elif args.grammar_ids:
        grammar_ids = args.grammar_ids
    else:
        parser.error('Provide grammar IDs or --all')

    all_results = []
    for gid in grammar_ids:
        try:
            r = run_pipeline(gid, cfg, kernels_only=args.kernels_only)
            all_results.append(r)
        except Exception as e:
            print(f"[{gid}] FAILED: {e}")
            all_results.append({'grammar_id': gid, 'status': 'error',
                                'error': str(e)})

    # Print summary
    print("\n" + "=" * 90)
    print(f"{'Grammar':>8}  {'Target':>6}  {'Found':>5}  {'Match':>5}  "
          f"{'Gold%':>5}  {'Status':<12}  "
          f"{'ML LM':>7}  {'Init LM':>7}  {'SGD LM':>7}")
    print("-" * 90)
    for r in all_results:
        gid = r['grammar_id']
        tgt = r.get('n_target_nts', '?')
        fnd = r.get('n_found_nts', '?')
        match = 'Y' if r.get('nt_match') else 'N'
        recall = f"{r.get('gold_recall', 0) * 100:.0f}" if 'gold_recall' in r else '?'
        status = r.get('status', '?')
        ml_lm = f"{r['ml']['labeled_micro']:.4f}" if 'ml' in r else '-'
        init_lm = f"{r['init']['labeled_micro']:.4f}" if 'init' in r else '-'
        sgd_lm = f"{r['sgd']['labeled_micro']:.4f}" if 'sgd' in r else '-'
        print(f"{gid:>8}  {tgt:>6}  {fnd:>5}  {match:>5}  "
              f"{recall:>5}  {status:<12}  "
              f"{ml_lm:>7}  {init_lm:>7}  {sgd_lm:>7}")

    n_match = sum(1 for r in all_results if r.get('nt_match'))
    n_eval = sum(1 for r in all_results if r.get('status') == 'evaluated')
    print(f"\nNT match: {n_match}/{len(all_results)}, "
          f"Evaluated: {n_eval}/{len(all_results)}")

    summary_path = os.path.join(cfg['basedir'], 'neural_pipeline_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Summary saved to {summary_path}")


if __name__ == '__main__':
    main()
