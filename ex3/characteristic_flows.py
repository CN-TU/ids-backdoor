#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as mc
import colorsys
import numpy as np
import sys
import os
import json
import pickle
from learn import numpy_sigmoid

DIR_NAME = "plots/characteristic_flows"

with open("categories_mapping.json", "r") as f:
	categories_mapping_content = json.load(f)
categories_mapping, mapping = categories_mapping_content["categories_mapping"], categories_mapping_content["mapping"]
reverse_mapping = {v: k for k, v in mapping.items()}
# print("reverse_mapping", reverse_mapping)

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
FEATURE_NAMES = ["packet length", "iat"]
LINESTYLES = ["solid", "dashed", "dashdot", "dotted"]

with open("flows_full_no_ttl_normalization_data.pickle", "rb") as f:
	means, stds = pickle.load(f)

file_name = sys.argv[1]
with open(file_name, "rb") as f:
	loaded = pickle.load(f)
results_by_attack_number = loaded["results_by_attack_number"]
flows_by_attack_number = loaded["flows_by_attack_number"]
result_ranges_by_attack_number = loaded["result_ranges_by_attack_number"]
sample_indices_by_attack_number = loaded["sample_indices_by_attack_number"]
features = loaded["features"]
# print("features", features)

for attack_type, (results_by_attack_number_item, flows_by_attack_number_item, result_ranges_by_attack_number_item, sample_indices_by_attack_number_item) in enumerate(zip(results_by_attack_number, flows_by_attack_number, result_ranges_by_attack_number, sample_indices_by_attack_number)):

	assert len(results_by_attack_number_item) == len(flows_by_attack_number_item) == len(result_ranges_by_attack_number_item) == len(sample_indices_by_attack_number_item)
	if len(results_by_attack_number_item) <= 0:
		continue

	sorted_seq_indices = [item[0] for item in sorted(enumerate(results_by_attack_number_item), key=lambda x: x[1].shape[0], reverse=True)]

	max_length = results_by_attack_number_item[sorted_seq_indices[0]].shape[0]
	print("n_flows", len(result_ranges_by_attack_number_item), "max_length", max_length)

	# print("result shapes", [item.shape for item in results_by_attack_number_item])
	padded_results = [np.concatenate((item, np.tile(item[-1], max_length-item.shape[0]))) for item in results_by_attack_number_item]
	# print("flow shapes", [item.shape for item in flows_by_attack_number_item])
	padded_flows = [np.concatenate((item, np.tile(item[-1:,:], (max_length-item.shape[0], 1)))) for item in flows_by_attack_number_item]
	# print("padded result shapes", [item.shape for item in padded_results])

	best_flow = max(enumerate(padded_results), key=lambda x: sum(x[1]))[0]
	worst_flow = min(enumerate(padded_results), key=lambda x: sum(x[1]))[0]
	stacked = np.stack(padded_results)
	# print("stacked.shape", stacked.shape)
	mean_flow = np.mean(stacked, axis=0)

	mean_features_flow = np.mean(np.stack(padded_flows), 0)

	# print("mean_flow.shape", mean_flow.shape)
	# print("mean_features_flow.shape", mean_features_flow.shape)
	average_flow = min(enumerate(padded_results), key=lambda x: np.linalg.norm(mean_flow - x[1]))[0]

	# print("best_flow", best_flow, "worst_flow", worst_flow, "average_flow", average_flow)

	all_legends = []
	fig, ax1 = plt.subplots()
	ax2 = ax1.twinx()

	for feature_index_from_zero, (feature_name, feature_index, ax) in enumerate(zip(FEATURE_NAMES, (3, 4), (ax1, ax2))):
		ax.set_ylabel(feature_name, color=colors[feature_index_from_zero])
		for flow_index, (flow_name, flow) in enumerate(zip(("best confidence", "worst confidence", "average confidence", "average flow"), (flows_by_attack_number_item[best_flow], flows_by_attack_number_item[worst_flow], flows_by_attack_number_item[average_flow], mean_features_flow))):

			if flow_index >= 2:
				break
			legend = "{}, {}".format(flow_name, feature_name)
			# print("flow_dataset_index", flow_dataset_index)
			# print("flow.shape", flow.shape)
			ret = ax.plot(range(flow.shape[0]), flow[:,feature_index]*stds[feature_index]+means[feature_index], label=legend, linestyle=LINESTYLES[flow_index], color=colors[feature_index_from_zero])
			forward_direction = flow[0,5]
			forward_points = flow[:,5] == forward_direction
			ret2 = ax.scatter(np.arange(flow.shape[0])[forward_points], (flow[:,feature_index]*stds[feature_index]+means[feature_index])[forward_points], color=colors[feature_index_from_zero], marker=">")
			ret3 = ax.scatter(np.arange(flow.shape[0])[~forward_points], (flow[:,feature_index]*stds[feature_index]+means[feature_index])[~forward_points], color=colors[feature_index_from_zero], marker="<")
			# print("flow[:,5]", flow[:,5])
			all_legends += ret

	plt.title(reverse_mapping[attack_type])
	all_labels = [item.get_label() for item in all_legends]
	ax1.legend(all_legends, all_labels)
	plt.xlabel('Sequence index')
	plt.tight_layout()
	# plt.show()

	os.makedirs(DIR_NAME, exist_ok=True)
	plt.savefig(DIR_NAME+'/{}_{}_{}.pdf'.format(file_name.split("/")[-1], attack_type, reverse_mapping[attack_type].replace("/", "-").replace(":", "-")))
	plt.clf()







