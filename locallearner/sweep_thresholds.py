#!/usr/bin/env python3
"""Sweep stopping-criterion thresholds on pre-generated 1M-sentence data.

Strategy:
  1. Run NMF with VERY permissive thresholds to collect per-candidate
     diagnostics (Frank-Wolfe distance, bootstrap p, min Cramér's V).
  2. Replay each grammar offline under different threshold combos.
  3. Report accuracy for each combo and suggest the best.

Usage:
    python3 sweep_thresholds.py [--base DIR] [--clusters N] [--feature-mode MODE]
"""

import sys, os, json, time, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
SYNTHETICPCFG_PATH = os.path.join(os.path.dirname(__file__),
    '..', '..', '..', 'cursor_syntheticpcfg', 'syntheticpcfg')
sys.path.insert(0, SYNTHETICPCFG_PATH)

import locallearner as ll_module

DEFAULT_BASE = os.path.join(os.path.dirname(__file__), '..', 'tmp', 'jm_experiment')


def collect_diagnostics(corpus_path, n_clusters, min_count, seed,
                        feature_mode, max_nt=25, cluster_file=None,
                        bigram_file=None, trigram_file=None):
    """Run NMF with very permissive thresholds; return per-candidate diagnostics."""
    learner = ll_module.LocalLearner(corpus_path)
    learner.number_clusters = n_clusters
    learner.min_count_nmf = min_count
    learner.seed = seed
    learner.feature_mode = feature_mode
    learner.cluster_file = cluster_file
    learner.bigram_file = bigram_file
    learner.trigram_file = trigram_file
    learner.nonterminals = 0        # auto mode
    learner.max_nonterminals = max_nt
    learner.min_nonterminals = 2

    # Very permissive thresholds so we collect ALL candidates
    learner.boot_p_threshold = 1.0       # never stop on bootstrap
    learner.cramers_v_threshold = 0.0    # never stop on V

    learner.find_kernels(verbose=False)

    return learner.candidate_diagnostics


def simulate_thresholds(candidates, true_nt, min_nt=2,
                        boot_p_thresh=0.01, v_thresh=0.05,
                        dist_thresh=0.0):
    """Given collected diagnostics, simulate what would happen with given thresholds.

    Returns the number of detected NTs (including S).
    The candidates list starts AFTER min_nt kernels are already in place
    (S + first untested kernel(s)), so we start counting from min_nt.
    """
    n_kernels = min_nt  # min_nt kernels exist before the first candidate is tested
    for c in candidates:
        # c['step'] is how many kernels existed when this candidate was proposed
        if n_kernels >= min_nt:
            # Test 0: Distance threshold (hyperplane)
            if dist_thresh > 0 and c.get('distance', c.get('fw_distance', 0)) < dist_thresh:
                break
            # Test 1: Bootstrap
            if c['boot_p'] > boot_p_thresh:
                break
            # Test 2: Word-context distinctness
            min_v = c.get('min_v')
            max_p = c.get('max_p')
            if min_v is not None and max_p is not None:
                is_distinct = (max_p < 0.001) and (min_v >= v_thresh)
                if not is_distinct:
                    break
        n_kernels += 1
    return n_kernels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default=DEFAULT_BASE)
    parser.add_argument('--clusters', type=int, default=10)
    parser.add_argument('--feature-mode', default='marginal',
                        choices=['marginal', 'joint'])
    parser.add_argument('--grammars', default=None,
                        help='Comma-separated grammar indices')
    args = parser.parse_args()

    base = os.path.abspath(args.base)
    manifest_path = os.path.join(base, 'manifest.json')
    with open(manifest_path) as f:
        manifest = json.load(f)

    grammars = [g for g in manifest['grammars'] if 'error' not in g]
    if args.grammars is not None:
        indices = set(int(x) for x in args.grammars.split(','))
        grammars = [g for g in grammars if g['index'] in indices]

    n_sentences = manifest.get('n_sentences', '?')
    min_count = max(5, n_sentences // 1000) if isinstance(n_sentences, int) else 1000

    # Phase 1: Collect diagnostics
    print(f"Collecting diagnostics for {len(grammars)} grammars "
          f"(mode={args.feature_mode}, clusters={args.clusters})...")

    all_diag = []  # list of (grammar_info, candidates)

    for i, ginfo in enumerate(grammars):
        corpus_path = os.path.join(base, ginfo['corpus_path'])
        gdir = os.path.dirname(corpus_path)
        cluster_file = os.path.join(gdir, f'clusters_c{args.clusters}_s42.clusters')
        bigram_file = os.path.join(gdir, 'bigrams.gz')
        trigram_file = os.path.join(gdir, 'trigrams.gz')
        n_nt = ginfo['n_nonterminals']
        print(f"  [{i+1}/{len(grammars)}] g{ginfo['index']:03d} "
              f"({n_nt} NTs) ...", end=' ', flush=True)
        t0 = time.time()
        candidates = collect_diagnostics(
            corpus_path, args.clusters, min_count, 42,
            args.feature_mode, max_nt=25,
            cluster_file=cluster_file,
            bigram_file=bigram_file, trigram_file=trigram_file)
        elapsed = time.time() - t0
        all_diag.append((ginfo, candidates))
        print(f"{len(candidates)} candidates in {elapsed:.1f}s")

    # Phase 2: Sweep thresholds
    print("\n" + "=" * 90)
    print(f"THRESHOLD SWEEP  (mode={args.feature_mode}, "
          f"clusters={args.clusters}, N={n_sentences})")
    print("=" * 90)

    # ---- Phase 2a: Sweep V and boot_p with no distance threshold ----
    boot_p_values = [0.01, 0.05]
    v_values = [0.05, 0.10, 0.15]
    dist_values = [0.0]  # no distance threshold

    print(f"\n--- V / bootstrap sweep (no distance threshold) ---")
    print(f"{'d_thr':>6s} {'boot_p':>8s} {'V_thr':>6s} | "
          f"{'Exact':>6s} {'±1':>6s} {'MeanΔ':>7s} {'MaxΔ':>5s} "
          f"{'MinΔ':>5s} | Deltas")
    print("-" * 120)

    best_score = None
    best_combo = None
    best_row = None
    best_deltas = None

    all_combos = []

    for dt in dist_values:
        for bp in boot_p_values:
            for vt in v_values:
                all_combos.append((dt, bp, vt))

    # ---- Phase 2b: Sweep distance thresholds with V=0.05 ----
    print()
    print(f"--- Distance threshold sweep (V=0.05, boot_p=0.01) ---")

    for dt in [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05, 0.06, 0.08, 0.10]:
        all_combos.append((dt, 0.01, 0.05))

    # ---- Phase 2c: Distance + V combos ----
    print()
    print(f"--- Distance + V combined sweep ---")
    for dt in [0.02, 0.025, 0.03, 0.035]:
        for vt in [0.05, 0.10, 0.15]:
            all_combos.append((dt, 0.01, vt))

    # Run all combos
    print(f"\n{'d_thr':>6s} {'boot_p':>8s} {'V_thr':>6s} | "
          f"{'Exact':>6s} {'±1':>6s} {'MeanΔ':>7s} {'MaxΔ':>5s} "
          f"{'MinΔ':>5s} | Deltas")
    print("-" * 120)

    seen = set()
    for dt, bp, vt in all_combos:
        key = (dt, bp, vt)
        if key in seen:
            continue
        seen.add(key)

        deltas = []
        for ginfo, candidates in all_diag:
            true_nt = ginfo['n_nonterminals']
            detected = simulate_thresholds(
                candidates, true_nt, min_nt=2,
                boot_p_thresh=bp, v_thresh=vt, dist_thresh=dt)
            deltas.append(detected - true_nt)

        n_exact = sum(1 for d in deltas if d == 0)
        n_close = sum(1 for d in deltas if abs(d) <= 1)
        mean_d = np.mean(deltas)
        max_d = max(deltas)
        min_d = min(deltas)
        delta_str = ' '.join(f'{d:+d}' for d in deltas)

        row = (f"{dt:6.3f} {bp:8.2f} {vt:6.2f} | "
               f"{n_exact:4d}/{len(deltas):d} "
               f"{n_close:4d}/{len(deltas):d} "
               f"{mean_d:+7.2f} {max_d:+5d} {min_d:+5d} "
               f"| {delta_str}")

        # Track best (prefer exact, then close, then low |mean_d|)
        score = (n_exact, n_close, -abs(mean_d))
        if best_score is None or score > best_score:
            best_score = score
            best_combo = (dt, bp, vt)
            best_row = row
            best_deltas = deltas

        print(row)

    print("\n" + "=" * 120)
    dt, bp, vt = best_combo
    print(f"BEST: d_thresh={dt}, boot_p={bp}, V_thresh={vt}")
    print(f"  {best_row}")

    # Per-grammar detail at best thresholds
    print(f"\nPer-grammar detail at best thresholds:")
    print(f"{'Grammar':>8s} {'NTs':>4s} {'Det':>4s} {'Δ':>4s} "
          f"{'StopReason':>12s} {'StopVal':>10s}")
    print("-" * 60)

    for ginfo, candidates in all_diag:
        true_nt = ginfo['n_nonterminals']
        detected = simulate_thresholds(
            candidates, true_nt, min_nt=2,
            boot_p_thresh=bp, v_thresh=vt, dist_thresh=dt)
        delta = detected - true_nt

        # Find what stopped it
        stop_reason = 'max_nt'
        stop_val = ''
        n_k = 2  # min_nt kernels exist before diagnostics begin
        for c in candidates:
            if n_k >= 2:
                d_val = c.get('distance', c.get('fw_distance', 0))
                if dt > 0 and d_val < dt:
                    stop_reason = 'distance'
                    stop_val = f"d={d_val:.4f}"
                    break
                if c['boot_p'] > bp:
                    stop_reason = 'bootstrap'
                    stop_val = f"p={c['boot_p']:.4f}"
                    break
                min_v = c.get('min_v')
                max_p = c.get('max_p')
                if min_v is not None and max_p is not None:
                    is_dist = (max_p < 0.001) and (min_v >= vt)
                    if not is_dist:
                        if max_p >= 0.001:
                            stop_reason = 'wctx_p'
                            stop_val = f"p={max_p:.2e}"
                        else:
                            stop_reason = 'wctx_V'
                            stop_val = f"V={min_v:.4f}"
                        break
            n_k += 1

        ok = 'OK' if delta == 0 else ''
        print(f"g{ginfo['index']:03d}     {true_nt:4d} {detected:4d} {delta:+4d} "
              f"{stop_reason:>12s} {stop_val:>10s}  {ok}")


if __name__ == '__main__':
    main()
