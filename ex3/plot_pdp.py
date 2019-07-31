#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import json
import pickle
from collections import Counter

DIR_NAME = "plots/plot_pdp"

with open("categories_mapping.json", "r") as f:
	categories_mapping_content = json.load(f)
categories_mapping, mapping = categories_mapping_content["categories_mapping"], categories_mapping_content["mapping"]
reverse_mapping = {v: k for k, v in mapping.items()}
print("reverse_mapping", reverse_mapping)

file_name = sys.argv[1]
with open(file_name, "rb") as f:
	loaded = pickle.load(f)
results_by_attack_number, feature_names, feature_values_by_attack_number = loaded["results_by_attack_number"], loaded["feature_names"], loaded["feature_values_by_attack_number"]

# print("results", results_by_attack_number)

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

for attack_type, (all_features, all_features_values) in enumerate(zip(results_by_attack_number, feature_values_by_attack_number)):

	print("attack_type", attack_type)
	fig, ax1 = plt.subplots()

	ax2 = ax1.twinx()

	ax2.set_ylabel('Prediction')
	ax1.set_ylabel("Number of samples")

	ax1.yaxis.tick_right()
	ax1.yaxis.set_label_position("right")
	ax2.yaxis.tick_left()
	ax2.yaxis.set_label_position("left")

	if all_features is None:
		continue
	# print("all_features.shape", all_features.shape)
	all_legends = []
	for feature_name, feature_index in zip(feature_names, range(all_features.shape[0])):

		as_ints = list(all_features_values[feature_index].astype(np.int32))

		# print("all_features_values[feature_index]", all_features_values[feature_index])
		# ret1 = ax1.hist(all_features_values[feature_index], bins=range(int(round(all_features_values[feature_index].max())+1)), width=1, color=colors[feature_index], alpha=0.2, label="{} occurrence".format(feature_name))

		counted = Counter(as_ints)
		keys = counted.keys()
		values = counted.values()

		# print("keys", keys, "values", values)
		ret1 = ax1.bar(keys, values, width=1000, color=colors[feature_index], alpha=0.5, label="{} occurrence".format(feature_name))

		ret2 = ax2.plot(all_features[feature_index,0,:], all_features[feature_index,1,:], color=colors[feature_index], label="{} confidence".format(feature_name))
		# all_legends.append(feature_name)
		# print("legend", legend)
		all_legends.append(ret1)
		all_legends += ret2

	plt.title(reverse_mapping[attack_type])
	# print("all_legends", all_legends)
	all_labels = [item.get_label() for item in all_legends]
	ax2.legend(all_legends, all_labels)
	ax1.set_xlabel('Feature')
	ax2.set_ylabel('Confidence')
	plt.tight_layout()
	#plt.savefig('%s.pdf' % os.path.splitext(fn)[0])
	# plt.show()

	os.makedirs(DIR_NAME, exist_ok=True)
	plt.savefig(DIR_NAME+'/{}_{}_{}.pdf'.format(file_name.split("/")[-1], attack_type, reverse_mapping[attack_type].replace("/", "-").replace(":", "-")))
	plt.clf()
