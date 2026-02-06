#run_sgd_io.py
#
# Train a PCFG using stochastic gradient descent with exponentiated gradient updates.
# The gradient (expected rule counts) is computed via the Inside-Outside algorithm
# using Mark Johnson's optimized C binary.
#
# Update rule (exponentiated gradient on the probability simplex):
#
#     gradient_r = E[count(r)] / (N * theta_r)
#     theta_r <- theta_r * exp(eta * gradient_r)
#     then renormalize per nonterminal
#
# This is mirror descent with KL divergence as the Bregman divergence.
# The E-step is computed by the C binary with the -e flag (single iteration,
# output unnormalized expected counts).

import argparse
import math
import os
import random
import subprocess
from tempfile import mkdtemp

import wcfg


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


def run_estep(grammar, data_file, tmpdir, io_binary, max_length):
	"""
	Run a single E-step using the C Inside-Outside binary.

	The C binary is called with -e to compute expected rule counts
	without iterating. The counts and neg-log-probability are returned.

	Returns:
		counts: dict mapping production tuples to expected counts
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

	# Parse neg-log-prob from stderr.
	# With -d 1000, the C binary prints: \t{neg_log_prob}\t{bits_per_token}
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

	return counts, neg_log_prob


def sgd_step(grammar, counts, learning_rate, min_prob=1e-20):
	"""
	Perform one exponentiated gradient step.

	The gradient of the log-likelihood w.r.t. theta_r is:
		dL/d(theta_r) = E[count(r)] / (N * theta_r)

	The exponentiated gradient update is:
		theta_r <- theta_r * exp(eta * dL/d(theta_r))
	then renormalize per nonterminal to stay on the simplex.

	Args:
		grammar: current WCFG (must be normalized as a PCFG)
		counts: expected counts dict from run_estep
		learning_rate: step size eta
		min_prob: floor for rule probabilities

	Returns:
		updated WCFG
	"""
	N = counts.get(('S1',), 0.0)
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

	# Renormalize per nonterminal to stay on the probability simplex
	new_grammar.locally_normalise_lax()

	# Floor very small probabilities
	for prod in new_grammar.productions:
		if new_grammar.parameters[prod] < min_prob:
			new_grammar.parameters[prod] = min_prob
	new_grammar.locally_normalise_lax()

	new_grammar.trim_zeros()
	new_grammar.set_log_parameters()

	return new_grammar


def main():
	parser = argparse.ArgumentParser(
		description='SGD training of PCFG via exponentiated gradient descent. '
		'Uses the C Inside-Outside binary for fast E-step computation.')

	parser.add_argument('grammar', type=str,
		help='Input PCFG file')
	parser.add_argument('data', type=str,
		help='Training data file (one sentence per line)')
	parser.add_argument('output', type=str,
		help='Output PCFG file')

	parser.add_argument('--io', default='io',
		help='Path to Inside-Outside binary (default: io)')
	parser.add_argument('--maxlength', type=int, default=20,
		help='Maximum sentence length for IO (default: 20)')
	parser.add_argument('--epochs', type=int, default=10,
		help='Number of training epochs (default: 10)')
	parser.add_argument('--batchsize', type=int, default=1000,
		help='Mini-batch size in sentences (default: 1000)')
	parser.add_argument('--lr', type=float, default=0.1,
		help='Initial learning rate eta (default: 0.1)')
	parser.add_argument('--lr_decay', type=float, default=0.0,
		help='Learning rate decay: lr = lr0 / (1 + step * decay) (default: 0.0)')
	parser.add_argument('--min_prob', type=float, default=1e-20,
		help='Minimum rule probability floor (default: 1e-20)')
	parser.add_argument('--shuffle', action='store_true',
		help='Shuffle mini-batch order each epoch')
	parser.add_argument('--seed', type=int, default=None,
		help='Random seed for batch shuffling')
	parser.add_argument('--save_every', type=int, default=0,
		help='Save intermediate grammar every N epochs (0 = off)')

	args = parser.parse_args()

	# Load grammar
	grammar = wcfg.load_wcfg_from_file(args.grammar)
	print(f"Loaded grammar: {len(grammar.nonterminals)} nonterminals, "
		  f"{len(grammar.productions)} productions")

	if not grammar.is_normalised():
		print("Grammar not normalised; normalising.")
		grammar.locally_normalise()

	# Set up temp directory and split data
	tmpdir = mkdtemp()
	print(f"Temp directory: {tmpdir}")

	batch_files = split_data(args.data, tmpdir, args.batchsize)
	n_batches = len(batch_files)
	print(f"Data split into {n_batches} batches of up to {args.batchsize} sentences")

	if args.seed is not None:
		random.seed(args.seed)

	# SGD training loop
	step = 0
	for epoch in range(1, args.epochs + 1):
		order = list(range(n_batches))
		if args.shuffle:
			random.shuffle(order)

		epoch_loss = 0.0
		epoch_sentences = 0

		for bi, batch_idx in enumerate(order):
			batch_file = batch_files[batch_idx]

			# Learning rate schedule
			lr = args.lr / (1.0 + step * args.lr_decay)
			step += 1

			# E-step: compute expected counts via C binary
			try:
				counts, neg_log_prob = run_estep(
					grammar, batch_file, tmpdir, args.io, args.maxlength)
			except RuntimeError as e:
				print(f"  [{epoch}/{bi+1}] E-step failed: {e}")
				continue

			N = counts.get(('S1',), 0)
			if N == 0:
				print(f"  [{epoch}/{bi+1}] No sentences parsed, skipping.")
				continue

			# Gradient step
			grammar = sgd_step(grammar, counts, lr, args.min_prob)

			epoch_sentences += N
			if neg_log_prob is not None:
				epoch_loss += neg_log_prob

			print(f"  Epoch {epoch}, batch {bi+1}/{n_batches}: "
				  f"lr={lr:.5f}, parsed={N:.0f}", end="")
			if neg_log_prob is not None:
				print(f", -logP={neg_log_prob:.2f}", end="")
			print()

		if epoch_sentences > 0:
			print(f"Epoch {epoch}/{args.epochs}: "
				  f"sentences={epoch_sentences:.0f}, "
				  f"total -logP={epoch_loss:.2f}")
		else:
			print(f"Epoch {epoch}/{args.epochs}: no sentences parsed")

		# Save intermediate grammar
		if args.save_every > 0 and epoch % args.save_every == 0:
			ipath = args.output.replace('.pcfg', f'_e{epoch}.pcfg')
			grammar.store(ipath, header=[f'SGD epoch {epoch}'])
			print(f"  Saved to {ipath}")

	# Save final grammar
	header = [
		f'SGD: epochs={args.epochs}, lr={args.lr}, '
		f'batch={args.batchsize}, decay={args.lr_decay}'
	]
	grammar.store(args.output, header=header)
	print(f"Saved final grammar to {args.output}")


if __name__ == '__main__':
	main()
