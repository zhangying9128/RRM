# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 15:10:50 2019

@author: Mac
"""
from collections import defaultdict
from argparse import ArgumentParser


def run():
	parser = ArgumentParser()
	parser.add_argument("--source-file", type=str)
	parser.add_argument("--reference-file", type=str)
	parser.add_argument("--index-file", type=str)
	args = parser.parse_args()


	def read_file(file):
		with open(file, 'r') as f:
			data = f.readlines()
		return data
	
	sources = read_file(args.source_file)
	targets = read_file(args.reference_file)
	indexes = read_file(args.index_file)

	Length_map = {'s':'short', 'm':'medium', 'l':'long'}
	sub_sources = defaultdict(list)
	sub_targets = defaultdict(list)

	for i, (s,t) in enumerate(zip(sources, targets)):
		symbol = indexes[i].strip().split()[1]
		sub_sources[symbol].append(s)
		sub_targets[symbol].append(t)

	for symbol, length in Length_map.items():
		with open(args.source_file+'.' + length, 'w') as fs, open(args.reference_file+'.' + length, 'w') as ft:
			for s, t in zip(sub_sources[symbol], sub_targets[symbol]):
				fs.write(s)
				ft.write(t)

if __name__ == "__main__":
	run()