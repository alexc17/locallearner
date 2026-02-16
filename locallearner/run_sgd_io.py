#run_sgd_io.py
#
# Train a PCFG using stochastic gradient descent.
# The E-step (expected rule counts) is computed via the Inside-Outside algorithm
# using either Mark Johnson's optimized C binary or pure Python.
#
# Two M-step update rules are supported:
#
# 1. Exponentiated gradient (default, update='exponentiated'):
#     theta_r <- theta_r * exp(eta * E_r / (N * theta_r))
#     then renormalize per nonterminal.
#     This is mirror descent with KL divergence as the Bregman divergence.
#
# 2. Linear interpolation (update='linear'):
#     theta_r <- (1 - stepsize) * theta_r + stepsize * (E_r / sum_r' E_r')
#     then renormalize.
#     This ensures productions unseen in a mini-batch shrink but don't vanish.

import argparse
import math
import os
import random
import shutil
import subprocess
import time
from collections import defaultdict
from tempfile import mkdtemp

import wcfg
from utility import ParseFailureException

# Default path to the C IO binary, relative to this file
_DEFAULT_IO_BINARY = os.path.join(
	os.path.dirname(os.path.abspath(__file__)), '..', 'lib', 'io')


def split_data(data_path, tmpdir, batch_size):
	"""Split data file into mini-batch files.

	Returns list of file paths.
	"""
	batch_files = []
	batch_num = 0
	line_count = 0
	outf = None

	with open(data_path) as inf:
		for line in inf:
			if line_count % batch_size == 0:
				if outf is not None:
					outf.close()
				batch_num += 1
				path = os.path.join(tmpdir, f"batch_{batch_num}.txt")
				batch_files.append(path)
				outf = open(path, 'w')
			outf.write(line)
			line_count += 1

	if outf is not None:
		outf.close()

	# Remove last file if it ended up empty
	if batch_files and os.path.getsize(batch_files[-1]) == 0:
		os.remove(batch_files[-1])
		batch_files.pop()

	return batch_files


def run_estep_c(grammar, data_file, tmpdir, io_binary, max_length):
	"""
	Run a single E-step using the C Inside-Outside binary.

	The C binary is called with -e to compute expected rule counts
	without iterating. The counts and neg-log-probability are returned.

	Returns:
		counts: dict mapping production tuples to expected counts
		n_parsed: number of sentences parsed (float, from S1 count)
		neg_log_prob: negative log probability of the data (or None)
	"""
	mjio_in = os.path.join(tmpdir, "grammar_in.mjio")
	mjio_out = os.path.join(tmpdir, "counts_out.mjio")

	grammar.store_mjio(mjio_in)

	cmd = (f"{io_binary} -e -d 1000 -g {mjio_in} -l {max_length} "
		   f"{data_file} > {mjio_out}")

	result = subprocess.run(
		cmd, shell=True, capture_output=True, text=True)

	if result.returncode != 0:
		raise RuntimeError(
			f"IO binary failed (exit {result.returncode}): {result.stderr}")

	counts = wcfg.load_mjio_counts(mjio_out)

	n_parsed = counts.get(('S1',), 0.0)

	# Parse neg-log-prob from stderr.
	neg_log_prob = None
	for line in result.stderr.strip().split('\n'):
		parts = line.strip().split('\t')
		for part in parts:
			part = part.strip()
			if not part:
				continue
			try:
				val = float(part)
				if neg_log_prob is None:
					neg_log_prob = val
			except ValueError:
				continue

	return counts, n_parsed, neg_log_prob


def run_estep_python(grammar, data_file, max_length):
	"""
	Run a single E-step using pure Python Inside-Outside.

	Returns:
		counts: dict mapping production tuples to expected counts
		n_parsed: number of sentences successfully parsed
		n_failed: number of parse failures
	"""
	io = wcfg.InsideComputation(grammar)
	posteriors = defaultdict(float)
	n_parsed = 0
	n_failed = 0

	with open(data_file) as f:
		for line in f:
			s = tuple(line.split())
			if len(s) == 0 or len(s) > max_length:
				continue
			try:
				io.add_posteriors(s, posteriors)
				n_parsed += 1
			except ParseFailureException:
				n_failed += 1

	return dict(posteriors), n_parsed, n_failed


def exponentiated_step(grammar, counts, learning_rate, min_prob=1e-20):
	"""
	Perform one exponentiated gradient step.

	theta_r <- theta_r * exp(eta * E_r / (N * theta_r))
	then renormalize per nonterminal.

	Args:
		grammar: current WCFG (must be normalized as a PCFG)
		counts: expected counts dict from E-step
		learning_rate: step size eta
		min_prob: floor for rule probabilities

	Returns:
		updated WCFG
	"""
	# N is total sentences parsed; for C E-step it's in ('S1',),
	# for Python it's the sum of start symbol counts.
	N = counts.get(('S1',), 0.0)
	if N == 0:
		# Try summing S productions (Python E-step doesn't have S1)
		N = sum(counts.get(prod, 0.0) for prod in grammar.productions
				if prod[0] == grammar.start)
	if N == 0:
		return grammar

	new_grammar = grammar.copy()

	for prod in new_grammar.productions:
		theta_r = new_grammar.parameters.get(prod, 0.0)
		if theta_r <= 0:
			continue

		e_r = counts.get(prod, 0.0)

		# Exponentiated gradient: theta_r *= exp(eta * E_r / (N * theta_r))
		gradient = e_r / (N * theta_r)
		log_update = learning_rate * gradient

		# Clamp to avoid numerical overflow
		log_update = max(min(log_update, 10.0), -10.0)

		new_grammar.parameters[prod] = theta_r * math.exp(log_update)

	new_grammar.locally_normalise_lax()

	# Floor very small probabilities
	for prod in new_grammar.productions:
		if new_grammar.parameters[prod] < min_prob:
			new_grammar.parameters[prod] = min_prob
	new_grammar.locally_normalise_lax()

	new_grammar.trim_zeros()
	new_grammar.set_log_parameters()

	return new_grammar


def linear_step(grammar, counts, stepsize):
	"""
	Perform one linear interpolation step.

	theta_r <- (1 - stepsize) * theta_r + stepsize * MLE_r
	where MLE_r = E_r / sum_{r': lhs(r')=lhs(r)} E_r'

	Productions unseen in the mini-batch shrink but don't vanish.

	Args:
		grammar: current WCFG
		counts: expected counts dict from E-step
		stepsize: interpolation weight (0 = keep old, 1 = full MLE)

	Returns:
		updated WCFG
	"""
	# Compute per-LHS totals for MLE normalization
	totals = defaultdict(float)
	for prod in grammar.productions:
		totals[prod[0]] += counts.get(prod, 0.0)

	new_grammar = grammar.copy()
	for prod in grammar.productions:
		old_p = grammar.parameters[prod]
		lhs_total = totals[prod[0]]
		if lhs_total > 0:
			mle_p = counts.get(prod, 0.0) / lhs_total
		else:
			mle_p = old_p
		new_grammar.parameters[prod] = (1 - stepsize) * old_p + stepsize * mle_p

	new_grammar.locally_normalise_lax()
	new_grammar.trim_zeros()
	new_grammar.set_log_parameters()

	return new_grammar


def run_epoch(grammar, data_path, maxlength=15, maxcount=1000,
			  stepsize=0.5, update='exponentiated',
			  io_binary=None, use_python=False,
			  min_prob=1e-20, verbose=0, shuffle=False, seed=None):
	"""
	Run one full epoch of SGD over the data.

	The data is split into mini-batches of maxcount sentences.
	After each mini-batch, the grammar parameters are updated.

	Args:
		grammar: WCFG object (should be a normalized PCFG)
		data_path: path to corpus file (one sentence per line)
		maxlength: maximum sentence length for IO
		maxcount: mini-batch size (number of sentences)
		stepsize: learning rate (eta for exponentiated, alpha for linear)
		update: 'exponentiated' (default) or 'linear'
		io_binary: path to C IO binary (default: lib/io relative to project)
		use_python: if True, use pure Python IO instead of C binary
		min_prob: floor for rule probabilities (exponentiated only)
		verbose: 0=summary only, N>0=print every N batches
		shuffle: shuffle mini-batch order
		seed: random seed for shuffling

	Returns:
		updated WCFG
	"""
	if io_binary is None:
		io_binary = _DEFAULT_IO_BINARY

	if not use_python and not os.path.isfile(io_binary):
		raise FileNotFoundError(
			f"C IO binary not found at {io_binary}. "
			f"Build it with 'cd lib && make io', or use use_python=True.")

	assert update in ('exponentiated', 'linear'), \
		f"update must be 'exponentiated' or 'linear', got '{update}'"

	t_start = time.time()

	# Set up temp directory and split data into batches
	tmpdir = mkdtemp()
	try:
		batch_files = split_data(data_path, tmpdir, maxcount)
		n_batches = len(batch_files)

		if seed is not None:
			random.seed(seed)

		order = list(range(n_batches))
		if shuffle:
			random.shuffle(order)

		current = grammar
		if not current.is_normalised():
			current = current.copy()
			current.locally_normalise()
			current.set_log_parameters()

		total_parsed = 0
		total_failed = 0
		total_loss = 0.0

		for bi, batch_idx in enumerate(order):
			batch_file = batch_files[batch_idx]

			# E-step
			if use_python:
				counts, n_parsed, n_failed = run_estep_python(
					current, batch_file, maxlength)
				total_failed += n_failed
			else:
				try:
					counts, n_parsed, neg_log_prob = run_estep_c(
						current, batch_file, tmpdir, io_binary, maxlength)
					if neg_log_prob is not None:
						total_loss += neg_log_prob
				except RuntimeError as e:
					print(f"  batch {bi+1}/{n_batches}: E-step failed: {e}")
					continue
				n_parsed = int(n_parsed)

			if n_parsed == 0:
				if verbose > 0:
					print(f"  batch {bi+1}/{n_batches}: no sentences parsed, skipping")
				continue

			total_parsed += n_parsed

			# M-step
			if update == 'exponentiated':
				current = exponentiated_step(current, counts, stepsize, min_prob)
			else:
				current = linear_step(current, counts, stepsize)

			if verbose > 0 and (bi + 1) % verbose == 0:
				elapsed = time.time() - t_start
				msg = (f"  batch {bi+1}/{n_batches}: "
					   f"{total_parsed} parsed")
				if total_failed > 0:
					msg += f", {total_failed} failures"
				msg += (f", E[len]={current.expected_length():.3f}"
						f", terms={len(current.terminals)}"
						f" [{elapsed:.1f}s]")
				print(msg)

		elapsed = time.time() - t_start
		msg = (f"IO epoch: {total_parsed} parsed, "
			   f"{total_failed} parse failures, "
			   f"{n_batches} mini-batch updates")
		if not use_python and total_loss > 0:
			msg += f", -logP={total_loss:.2f}"
		msg += f" [{elapsed:.1f}s]"
		print(msg)

	finally:
		shutil.rmtree(tmpdir, ignore_errors=True)

	return current


# Legacy aliases for backward compatibility
sgd_step = exponentiated_step
run_estep = run_estep_c


def main():
	parser = argparse.ArgumentParser(
		description='SGD training of PCFG. '
		'Uses the C Inside-Outside binary for fast E-step computation.')

	parser.add_argument('grammar', type=str,
		help='Input PCFG file')
	parser.add_argument('data', type=str,
		help='Training data file (one sentence per line)')
	parser.add_argument('output', type=str,
		help='Output PCFG file')

	parser.add_argument('--io', default=None,
		help='Path to Inside-Outside binary (default: lib/io)')
	parser.add_argument('--python', action='store_true',
		help='Use pure Python IO instead of C binary')
	parser.add_argument('--maxlength', type=int, default=15,
		help='Maximum sentence length for IO (default: 15)')
	parser.add_argument('--epochs', type=int, default=1,
		help='Number of training epochs (default: 1)')
	parser.add_argument('--batchsize', type=int, default=1000,
		help='Mini-batch size in sentences (default: 1000)')
	parser.add_argument('--lr', type=float, default=0.5,
		help='Learning rate / stepsize (default: 0.5)')
	parser.add_argument('--update', choices=['exponentiated', 'linear'],
		default='exponentiated',
		help='M-step update rule (default: exponentiated)')
	parser.add_argument('--min_prob', type=float, default=1e-20,
		help='Minimum rule probability floor (default: 1e-20)')
	parser.add_argument('--shuffle', action='store_true',
		help='Shuffle mini-batch order each epoch')
	parser.add_argument('--seed', type=int, default=None,
		help='Random seed for batch shuffling')
	parser.add_argument('--verbose', type=int, default=1,
		help='Print stats every N batches (0=summary only, default: 1)')
	parser.add_argument('--save_every', type=int, default=0,
		help='Save intermediate grammar every N epochs (0 = off)')

	args = parser.parse_args()

	# Load grammar
	grammar = wcfg.load_wcfg_from_file(args.grammar)
	print(f"Loaded grammar: {len(grammar.nonterminals)} nonterminals, "
		  f"{len(grammar.terminals)} terminals, "
		  f"{len(grammar.productions)} productions")

	for epoch in range(1, args.epochs + 1):
		print(f"\n--- Epoch {epoch}/{args.epochs} ---")
		grammar = run_epoch(
			grammar, args.data,
			maxlength=args.maxlength,
			maxcount=args.batchsize,
			stepsize=args.lr,
			update=args.update,
			io_binary=args.io,
			use_python=args.python,
			min_prob=args.min_prob,
			verbose=args.verbose,
			shuffle=args.shuffle,
			seed=args.seed,
		)

		if args.save_every > 0 and epoch % args.save_every == 0:
			ipath = args.output.replace('.pcfg', f'_e{epoch}.pcfg')
			grammar.store(ipath)
			print(f"  Saved to {ipath}")

	# Save final grammar
	grammar.store(args.output)
	print(f"\nSaved final grammar to {args.output}")


if __name__ == '__main__':
	main()
