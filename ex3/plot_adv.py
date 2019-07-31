#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import json
import pickle
from learn import numpy_sigmoid

DIR_NAME = "plots/plot_adv"

with open("categories_mapping.json", "r") as f:
	categories_mapping_content = json.load(f)
categories_mapping, mapping = categories_mapping_content["categories_mapping"], categories_mapping_content["mapping"]
reverse_mapping = {v: k for k, v in mapping.items()}
# print("reverse_mapping", reverse_mapping)

with open("flows_full_no_ttl_normalization_data.pickle", "rb") as f:
	means, stds = pickle.load(f)

# TODO: Implement for more than one adv output so that different tradeoffs can be compared.
file_name = sys.argv[1]
with open(file_name, "rb") as f:
	loaded = pickle.load(f)
results_by_attack_number = loaded["results_by_attack_number"]
orig_results_by_attack_number = loaded["orig_results_by_attack_number"]
modified_flows_by_attack_number = loaded["modified_flows_by_attack_number"]
orig_flows_by_attack_number = loaded["orig_flows_by_attack_number"]

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
ORDERING = ["original", "adversarial"]
FEATURE_NAMES = ["packet length", "iat"]

for attack_type, (results_by_attack_number_item, orig_results_by_attack_number_item, modified_flows_by_attack_number_item, orig_flows_by_attack_number_item) in enumerate(zip(results_by_attack_number, orig_results_by_attack_number, modified_flows_by_attack_number, orig_flows_by_attack_number)):

	assert len(results_by_attack_number_item) == len(orig_results_by_attack_number_item) == len(modified_flows_by_attack_number_item) == len(orig_flows_by_attack_number_item)
	if len(results_by_attack_number_item) <= 0:
		continue

	# print([np.array(item).shape for item in orig_flows_by_attack_number_item], [np.array(item).shape for item in orig_results_by_attack_number_item])
	stacked_original = [np.concatenate((np.array(orig_flow), np.array(orig_result)), axis=-1) for orig_flow, orig_result in zip(orig_flows_by_attack_number_item, orig_results_by_attack_number_item)]
	stacked_modified = [np.concatenate((np.array(modified_flow), np.array(modified_result)), axis=-1) for modified_flow, modified_result in zip(modified_flows_by_attack_number_item, results_by_attack_number_item)]

	# print([np.array(item).shape for item in stacked_original], [np.array(item).shape for item in stacked_modified])
	seqs = [np.stack((orig, modified)) for orig, modified in zip(stacked_original, stacked_modified)]
	# print("all_stacked", [item.shape for item in all_stacked])

	# Filter good seqs where the adversarial attack succeeded.
	filtered_seqs = [item for item in seqs if int(np.round(np.mean(numpy_sigmoid(item[0,-1:,-1])))) == 1 and int(np.round(np.mean(numpy_sigmoid(item[1,-1:,-1])))) == 0]

	print("Original seqs", len(seqs), "filtered seqs", len(filtered_seqs))
	seqs = filtered_seqs

	if len(filtered_seqs) <= 0:
		continue

	# print("attack_type", attack_type, "results_by_attack_number_item[0].shape", results_by_attack_number_item[0].shape)

	seqs = sorted(seqs, key=lambda x: x.shape[1], reverse=True)
	# print("seqs", [seq.shape for seq in seqs])
	max_length = seqs[0].shape[1]
	print("max_length", max_length)

	values_by_length = []

	for i in range(max_length):
		values_by_length.append([])
		for seq in seqs:
			if seq.shape[1] < i+1:
				break

			values_by_length[i].append(seq[:,i:i+1,:])

	for i in range(len(values_by_length)):
		values_by_length[i] = np.concatenate(values_by_length[i], axis=1)

	# print("shape of values", [item.shape for item in values_by_length])

	flow_means = np.array([np.mean(item, axis=1) for item in values_by_length])
	# print("means.shape", means.shape)
	medians = np.array([np.median(item, axis=1) for item in values_by_length])
	# print("medians.shape", medians.shape)
	# print("means.shape", means.shape)
	# stds = np.array([np.std(item, axis=0) for item in values_by_length])
	# first_quartiles = np.array([np.quantile(item, 0.25, axis=1) for item in values_by_length])
	# third_quartiles = np.array([np.quantile(item, 0.75, axis=1) for item in values_by_length])

	# print(medians.shape, first_quartiles.shape, third_quartiles.shape)
	# quit()

	all_legends = []
	assert len(flow_means[1].shape) == 2
	# for i in range(medians.shape[1]):
	fig, ax1 = plt.subplots()
	ax2 = ax1.twinx()

	for feature_index_from_zero, (feature_name, feature_index, ax) in enumerate(zip(FEATURE_NAMES, (3, 4), (ax1, ax2))):
		ax.set_ylabel(feature_name, color=colors[feature_index_from_zero])
		for adv_real_index in range(flow_means.shape[1]):
			# print("i", i)
			# print("means[:,adv_real_index,feature_index].shape", (flow_means[:,adv_real_index,feature_index]*stds[feature_index]+means[feature_index]).shape)
			correct_linestyle = "solid" if adv_real_index==0 else "dashed"
			legend = "{}, {}".format(ORDERING[adv_real_index], feature_name)
			ret = ax.plot(range(max_length), flow_means[:,adv_real_index,feature_index]*stds[feature_index]+means[feature_index], label=legend, linestyle=correct_linestyle, color=colors[feature_index_from_zero])
			# plt.fill_between(range(medians.shape[0]), first_quartiles[:,i], third_quartiles[:,i], alpha=0.5, edgecolor=colors[i], facecolor=colors[i])
			# legend = ORDERING[i:i+1]
			# legend[0] = legend[0]+" median"
			# legend[-1] = legend[-1]+" 1st and 3rd quartile"
			all_legends += ret
			# print("legend", legend)

	plt.title(reverse_mapping[attack_type])
	# print("all_legends", all_legends)
	all_labels = [item.get_label() for item in all_legends]
	ax1.legend(all_legends, all_labels, loc=0)
	plt.xlabel('Sequence index')
	plt.tight_layout()
	# plt.xticks(range(medians.shape[0]))
	#plt.savefig('%s.pdf' % os.path.splitext(fn)[0])
	# plt.show()

	os.makedirs(DIR_NAME, exist_ok=True)
	plt.savefig(DIR_NAME+'/{}_{}.pdf'.format(file_name.split("/")[-1], attack_type))
	plt.clf()



