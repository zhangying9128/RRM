# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 13:17:53 2019

@author: zhang
"""
from argparse import ArgumentParser
def run():
	parser = ArgumentParser()
	parser.add_argument("--result-file", type=str, default="", help="")
	parser.add_argument("--translation-file", type=str, default="", help="")
	args = parser.parse_args()

	with open(args.result_file, 'r') as f:
		results = f.readlines()

	translations = []
	for i in range(7, len(results)-2):
		if results[i].startswith("H-"):
			translations.append(results[i])

	translations.sort(key=lambda x: int(x.split('\t')[0].split("H-")[1]))

	with open(args.translation_file, 'w') as f:
		for translation in translations:
			f.write(translation.split('\t')[2])

if __name__ == "__main__":
	run()
