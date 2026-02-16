#!/usr/bin/env python3
"""Full pipeline for a single grammar: gold kernels -> Renyi init -> C SGD -> eval.

Usage:
    python run_gold_pipeline.py g000
    python run_gold_pipeline.py --base /path/to/experiment g005
"""
import argparse
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


DEFAULT_BASE = '/Users/alexc/research/cursor_locallearner/locallearner/tmp/jm_experiment'


def get_gold_kernels(grammar):
    """Find optimal anchor word for each non-start NT using Hungarian algorithm."""
    te = grammar.terminal_expectations()
    pe = grammar.production_expectations()
    non_start_nts = sorted([nt for nt in grammar.nonterminals if nt != grammar.start])
    terminals = sorted(grammar.terminals)
    nn = len(non_start_nts)
    nt_count = len(terminals)

    score_matrix = np.full((nn, nt_count), -1e30)
    for j, ntj in enumerate(non_start_nts):
        for i, a in enumerate(terminals):
            ea = te.get(a, 0.0)
            if ea <= 0: continue
            e_prod = pe.get((ntj, a), 0.0)
            if e_prod <= 0: continue
            posterior = e_prod / ea
            score_matrix[j, i] = 0.5 * math.log(posterior) + 0.5 * math.log(ea)

    dim = max(nn, nt_count)
    padded = np.full((dim, dim), -1e30)
    padded[:nn, :nt_count] = score_matrix
    row_ind, col_ind = linear_sum_assignment(-padded)

    assignment = {}
    for j, i in zip(row_ind, col_ind):
        if j < nn and i < nt_count:
            assignment[non_start_nts[j]] = terminals[i]

    return [assignment[ntj] for ntj in non_start_nts]


def run_pipeline(grammar_id, base_dir=DEFAULT_BASE, maxlength=15, maxcount=1000,
                 stepsize=0.5, n_corpus=100000, k=2, model_type='bow'):
    """Run full pipeline for one grammar. Returns dict of results."""
    gdir = os.path.join(base_dir, grammar_id)
    grammar_path = os.path.join(gdir, 'grammar.pcfg')
    corpus_path = os.path.join(gdir, 'corpus.txt')

    if not os.path.exists(grammar_path):
        raise FileNotFoundError(f"Grammar not found: {grammar_path}")
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")

    target = wcfg.load_wcfg_from_file(grammar_path)
    mt_tag = 'pos' if model_type == 'positional' else 'bow'
    print(f"[{grammar_id}] Target: {len(target.nonterminals)} NTs, "
          f"{len(target.terminals)} terminals, k={k}, model={mt_tag}")

    # 1. Gold kernels
    gold_anchors = get_gold_kernels(target)
    n_nts = len(gold_anchors)
    print(f"[{grammar_id}] Gold anchors ({n_nts}): {gold_anchors[:5]}...")

    # 2. NeuralLearner with gold anchors
    nl = nl_module.NeuralLearner(corpus_path)
    nl.k = k
    nl.model_type = model_type
    # Use distinct filenames for positional vs bow models
    prefix = f'{mt_tag}_' if model_type == 'positional' else ''
    nl.single_model_file = os.path.join(gdir, f'{prefix}cloze_k{k}.pt')
    nl.pair_model_file = os.path.join(gdir, f'{prefix}pair_cloze_k{k}.pt')

    nl.train_single_model(verbose=False)
    nl.train_pair_model(verbose=True)  # verbose for training progress

    # Set gold anchors
    nl.anchors = gold_anchors
    nl.nonterminals = ['S'] + [f'NT_{w}' for w in gold_anchors]
    nl.anchor2nt = {w: f'NT_{w}' for w in gold_anchors}

    # 3. Estimate xi and build WCFG
    nl.estimate_lexical_xi(verbose=False)
    nl.estimate_binary_xi(verbose=False)
    nl.build_wcfg(verbose=False)
    nl.xi_wcfg.store(os.path.join(gdir, f'gold_init_{mt_tag}_k{k}.wcfg'))
    init_pcfg = nl.convert_to_pcfg(verbose=False)
    init_pcfg.store(os.path.join(gdir, f'gold_init_{mt_tag}_k{k}.pcfg'))
    print(f"[{grammar_id}] Init: E[len]={init_pcfg.expected_length():.3f}, "
          f"{len(init_pcfg.terminals)} terminals")

    # 4. Prepare corpus file (first n_corpus sentences)
    corpus_subset = os.path.join(gdir, f'corpus_{n_corpus // 1000}k.txt')
    if not os.path.exists(corpus_subset):
        with open(corpus_path) as inf, open(corpus_subset, 'w') as outf:
            for i, line in enumerate(inf):
                if i >= n_corpus:
                    break
                outf.write(line)

    # 5. One epoch of C SGD (linear)
    t0 = time.time()
    sgd_pcfg = run_epoch(
        init_pcfg, corpus_subset,
        maxlength=maxlength, maxcount=maxcount, stepsize=stepsize,
        update='linear', verbose=0)
    sgd_time = time.time() - t0
    sgd_pcfg.store(os.path.join(gdir, f'gold_c_linear_1epoch_{mt_tag}_k{k}.pcfg'))
    print(f"[{grammar_id}] SGD: {sgd_time:.1f}s, "
          f"E[len]={sgd_pcfg.expected_length():.3f}")

    # 6. Evaluate
    eval_kwargs = {'max_length': 20, 'seed': 42, 'samples': 1000}

    def evaluate(hyp, label):
        h = hyp.copy()
        h.renormalise_locally()
        h.set_log_parameters()
        mapping = evaluation.estimate_bijection(target, h, **eval_kwargs)
        h_relab = h.relabel({a: b for b, a in mapping.items()})
        scores = evaluation.do_parseval_monte_carlo(target, [h_relab], **eval_kwargs)
        denom = scores['trees_denominator']
        lab_d = scores['labeled_denominator']
        ulab_d = scores['unlabeled_denominator']
        kld = evaluation.smoothed_kld_exact(target, h, compute_bijection=True)
        return {
            'kld': kld,
            'labeled_exact': scores['original:hypothesis0:labeled:exact_match'] / denom,
            'unlabeled_exact': scores['original:hypothesis0:unlabeled:exact_match'] / denom,
            'labeled_micro': scores['original:hypothesis0:labeled:microaveraged'] / lab_d,
            'unlabeled_micro': scores['original:hypothesis0:unlabeled:microaveraged'] / ulab_d,
            'elen': hyp.expected_length(),
            'terminals': len(hyp.terminals),
        }

    # Target ceiling
    target_scores = evaluation.do_parseval_monte_carlo(target, [], **eval_kwargs)
    td = target_scores['trees_denominator']
    tld = target_scores['labeled_denominator']
    tud = target_scores['unlabeled_denominator']
    target_results = {
        'labeled_exact': target_scores['original:gold viterbi:labeled:exact_match'] / td,
        'unlabeled_exact': target_scores['original:gold viterbi:unlabeled:exact_match'] / td,
        'labeled_micro': target_scores['original:gold viterbi:labeled:microaveraged'] / tld,
        'unlabeled_micro': target_scores['original:gold viterbi:unlabeled:microaveraged'] / tud,
        'kld': 0.0,
        'elen': target.expected_length(),
        'terminals': len(target.terminals),
    }

    init_results = evaluate(init_pcfg, 'init')
    sgd_results = evaluate(sgd_pcfg, 'sgd')

    results = {
        'grammar_id': grammar_id,
        'k': k,
        'model_type': model_type,
        'n_nonterminals': len(target.nonterminals),
        'n_terminals': len(target.terminals),
        'sgd_time': sgd_time,
        'target': target_results,
        'init': init_results,
        'sgd': sgd_results,
    }

    print(f"[{grammar_id}] Results: "
          f"target_lab_micro={target_results['labeled_micro']:.4f}, "
          f"init_lab_micro={init_results['labeled_micro']:.4f}, "
          f"sgd_lab_micro={sgd_results['labeled_micro']:.4f}")

    # Save results
    out_path = os.path.join(gdir, f'gold_pipeline_results_{mt_tag}_k{k}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Full pipeline: gold kernels -> Renyi init -> C SGD -> eval')
    parser.add_argument('grammar_id', type=str, help='Grammar ID (e.g. g000)')
    parser.add_argument('--base', type=str, default=DEFAULT_BASE,
        help='Base experiment directory')
    parser.add_argument('--maxlength', type=int, default=15)
    parser.add_argument('--maxcount', type=int, default=1000)
    parser.add_argument('--stepsize', type=float, default=0.5)
    parser.add_argument('--corpus_size', type=int, default=100000)
    parser.add_argument('--k', type=int, default=2, help='Context width for neural models')
    parser.add_argument('--model_type', type=str, default='bow',
                        choices=['bow', 'positional'],
                        help='Neural model type: bow (bag-of-words) or positional')
    args = parser.parse_args()

    run_pipeline(args.grammar_id, args.base,
                 maxlength=args.maxlength, maxcount=args.maxcount,
                 stepsize=args.stepsize, n_corpus=args.corpus_size,
                 k=args.k, model_type=args.model_type)


if __name__ == '__main__':
    main()
