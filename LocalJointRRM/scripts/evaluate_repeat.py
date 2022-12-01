import os
from argparse import ArgumentParser
from collections import defaultdict, Counter

def read_file(file):
	with open(file, 'r') as f:
		data = f.readlines()
	return data

def eval_pair_repeat(reference, output):
	reference = Counter(reference.split())
	output = Counter(output.split())
	Repeat = 0

	for word, count in output.items():
		if count > 1:
			Repeat += max(0, count - reference[word]) 
	return Repeat

def evaluate(references, outputs):
	cum_repeat = 0
	for reference, output in zip(references, outputs):
		cum_repeat += eval_pair_repeat(reference.strip(), output.strip())
	print('{} sentences with cumulative Repeat {}, average Repeat {} '.format(len(references), cum_repeat, cum_repeat / len(references)))

def run():
	parser = ArgumentParser()
	parser.add_argument("--output-file", type=str, default="", help="")
	parser.add_argument("--reference-file", type=str, default="", help="")
	parser.add_argument("--index-file", type=str, default="", help="")
	args = parser.parse_args()

	outputs = read_file(args.output_file)
	references = read_file(args.reference_file)

	print('Over corpus result')
	evaluate(references, outputs)
	if os.path.exists(args.index_file):
		print('------------------------------')
		sub_outputs = defaultdict(list)
		sub_references = defaultdict(list)
		indexes = read_file(args.index_file)

		for i, (out, ref) in enumerate(zip(outputs, references)):
			symbol = indexes[i].strip().split()[1]
			sub_outputs[symbol].append(out)
			sub_references[symbol].append(ref)

		for length in sub_references.keys():
			print('{} result'.format(length))
			evaluate(sub_references[length], sub_outputs[length])
			print('------------------------------')

if __name__ == "__main__":
	run()
