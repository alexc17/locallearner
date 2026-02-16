#!/usr/bin/env python3
"""Launch hyperparameter sweep: stepsize x maxcount, 8-way parallel."""
import subprocess
import os
import sys
import time
import json
import itertools

sys.stdout.reconfigure(line_buffering=True)

BASE = '/Users/alexc/research/cursor_locallearner/locallearner/tmp/jm_experiment/g000/'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(SCRIPT_DIR, 'run_single_epoch.py')
PYTHON = '/opt/anaconda3/bin/python3'
MAX_PARALLEL = 8

STEPSIZES = [0.1, 0.3, 0.5, 0.7, 1.0]
MAXCOUNTS = [1000, 5000, 10000, 50000]
MAXLENGTH = 15

# Output directory
out_dir = os.path.join(BASE, 'sweep_results')
os.makedirs(out_dir, exist_ok=True)

# Build job list
jobs = []
for ss, mc in itertools.product(STEPSIZES, MAXCOUNTS):
    tag = f'ss{ss}_mc{mc}_ml{MAXLENGTH}'
    out_file = os.path.join(out_dir, f'{tag}.json')
    cmd = [
        PYTHON, WORKER,
        '--base', BASE,
        '--stepsize', str(ss),
        '--maxcount', str(mc),
        '--maxlength', str(MAXLENGTH),
        '--output', out_file,
    ]
    jobs.append((tag, cmd, out_file))

print(f'Sweep: {len(jobs)} configs, {MAX_PARALLEL}-way parallel')
print(f'Stepsizes: {STEPSIZES}')
print(f'Maxcounts: {MAXCOUNTS}')
print(f'Maxlength: {MAXLENGTH}')
print(f'Output: {out_dir}')
print()

# Run jobs with bounded parallelism
t_start = time.time()
running = {}  # tag -> (proc, start_time)
pending = list(jobs)
completed = 0

while pending or running:
    # Launch jobs up to MAX_PARALLEL
    while pending and len(running) < MAX_PARALLEL:
        tag, cmd, out_file = pending.pop(0)
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=SCRIPT_DIR,
        )
        running[tag] = (proc, time.time())
        print(f'[START] {tag} (pid={proc.pid}, {len(running)} running, {len(pending)} pending)')

    # Poll for completion
    done_tags = []
    for tag, (proc, t0) in running.items():
        ret = proc.poll()
        if ret is not None:
            elapsed = time.time() - t0
            stdout = proc.stdout.read().decode().strip()
            stderr = proc.stderr.read().decode().strip()
            completed += 1
            if ret == 0:
                print(f'[DONE {completed}/{len(jobs)}] {tag} ({elapsed:.0f}s) {stdout}')
            else:
                print(f'[FAIL {completed}/{len(jobs)}] {tag} ({elapsed:.0f}s) rc={ret}')
                if stderr:
                    print(f'  stderr: {stderr[:200]}')
            done_tags.append(tag)

    for tag in done_tags:
        del running[tag]

    if running and not done_tags:
        time.sleep(2)

total_time = time.time() - t_start
print(f'\nAll {len(jobs)} jobs completed in {total_time:.0f}s')

# Collect results
print('\n=== Results Summary ===')
results = []
for tag, cmd, out_file in jobs:
    if os.path.exists(out_file):
        with open(out_file) as f:
            results.append(json.load(f))

if not results:
    print('No results found!')
    sys.exit(1)

# Print table
print(f'\n{"stepsize":>8} {"maxcount":>8} {"KLD":>8} {"LabExact":>8} {"LabMicro":>8} {"E[len]":>7} {"Time":>6}')
print('-' * 62)
for r in sorted(results, key=lambda x: (x['stepsize'], x['maxcount'])):
    print(f'{r["stepsize"]:>8.1f} {r["maxcount"]:>8d} {r["kld"]:>8.4f} '
          f'{r["labeled_exact"]:>8.4f} {r["labeled_micro"]:>8.4f} '
          f'{r["elen"]:>7.3f} {r["sgd_time"]:>5.0f}s')

# Save collected results
collected_path = os.path.join(out_dir, 'all_results.json')
with open(collected_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nCollected results saved to {collected_path}')
