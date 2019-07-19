#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import json
import pickle

ORDERING = ['Real', 'Length min', 'IAT min', 'Length max', 'IAT max']

with open("categories_mapping.json", "r") as f:
	categories_mapping_content = json.load(f)
categories_mapping, mapping = categories_mapping_content["categories_mapping"], categories_mapping_content["mapping"]
reverse_mapping = {v: k for k, v in mapping.items()}
print("reverse_mapping", reverse_mapping)

file_name = sys.argv[1]
with open(file_name, "rb") as f:
	loaded = pickle.load(f)
results_by_attack_number, sample_indices_by_attack_number = loaded["results_by_attack_number"], loaded["sample_indices_by_attack_number"]

# print("results", results_by_attack_number)
# print("sample_indices", sample_indices_by_attack_number)
lens_results = [len(item) for item in results_by_attack_number]
lens_indices = [len(item) for item in sample_indices_by_attack_number]

print("lens_indices", "\n".join(["{}: {}".format(reverse_mapping[attack], length) for attack, length in zip(range(len(lens_indices)), lens_indices)]))
assert lens_results == lens_indices

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

for attack_type, seqs in enumerate(results_by_attack_number):
	if len(seqs) < 10:
		continue

	seqs = [seq.transpose() for seq in seqs]
	seqs = sorted(seqs, key=lambda x: x.shape[0], reverse=True)
	# print("seqs", [seq.shape for seq in seqs])
	max_length = len(seqs[0])
	# print("max_length", max_length)

	values_by_length = []

	for i in range(max_length):
		values_by_length.append([])
		for seq in seqs:
			if len(seq) < i+1:
				break

			values_by_length[i].append(seq[i:i+1,:])

	for i in range(len(values_by_length)):
		values_by_length[i] = np.concatenate(values_by_length[i], axis=0)

	# print("shape of values", [item.shape for item in values_by_length])

	# means = np.array([np.mean(item, axis=0) for item in values_by_length])
	medians = np.array([np.median(item, axis=0) for item in values_by_length])
	# print("means.shape", means.shape)
	# stds = np.array([np.std(item, axis=0) for item in values_by_length])
	first_quartiles = np.array([np.quantile(item, 0.25, axis=0) for item in values_by_length])
	third_quartiles = np.array([np.quantile(item, 0.75, axis=0) for item in values_by_length])

	# print(medians.shape, first_quartiles.shape, third_quartiles.shape)
	# quit()

	all_legends = []
	# for i in range(medians.shape[1]):
	for i in range(1):
		# print("i", i)
		plt.plot(medians[:,i], color=colors[i])
		plt.fill_between(range(medians.shape[0]), first_quartiles[:,i], third_quartiles[:,i], alpha=0.5, edgecolor=colors[i], facecolor=colors[i])
		legend = ORDERING[i:i+1]*2
		# legend = ORDERING[i:i+1]
		legend[0] = legend[0]+" median"
		legend[-1] = legend[-1]+" 1st and 3rd quartile"
		all_legends += legend
		# print("legend", legend)

	plt.title(reverse_mapping[attack_type])
	plt.legend(all_legends)
	plt.xlabel('Sequence index')
	plt.ylabel('Prediction')
	plt.xticks(range(medians.shape[0]))
	#plt.savefig('%s.pdf' % os.path.splitext(fn)[0])
	plt.show()

