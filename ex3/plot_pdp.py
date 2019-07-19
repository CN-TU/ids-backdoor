#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import json
import pickle

with open("categories_mapping.json", "r") as f:
	categories_mapping_content = json.load(f)
categories_mapping, mapping = categories_mapping_content["categories_mapping"], categories_mapping_content["mapping"]
reverse_mapping = {v: k for k, v in mapping.items()}
print("reverse_mapping", reverse_mapping)

file_name = sys.argv[1]
with open(file_name, "rb") as f:
	loaded = pickle.load(f)
results_by_attack_number, feature_names = loaded["results_by_attack_number"], loaded["feature_names"]

# print("results", results_by_attack_number)

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

for attack_type, all_features in enumerate(results_by_attack_number):

	if all_features is None:
		continue
	# print("all_features.shape", all_features.shape)
	all_legends = []
	for feature_name, feature_index in zip(feature_names, range(all_features.shape[0])):

		plt.plot(all_features[feature_index,0,:], all_features[feature_index,1,:], color=colors[feature_index])
		all_legends.append(feature_name)
		# print("legend", legend)

	plt.title(reverse_mapping[attack_type])
	plt.legend(all_legends)
	plt.xlabel('Feature')
	plt.ylabel('Probability')
	#plt.savefig('%s.pdf' % os.path.splitext(fn)[0])
	plt.show()

