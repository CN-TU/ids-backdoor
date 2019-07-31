#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as mc
# from matplotlib import rcParams
# rcParams.update({'figure.autolayout': True})
import colorsys
import numpy as np
import sys
import os
import json
import pickle
from learn import numpy_sigmoid

DIR_NAME = "plots/plot_new"

with open("categories_mapping.json", "r") as f:
	categories_mapping_content = json.load(f)
categories_mapping, mapping = categories_mapping_content["categories_mapping"], categories_mapping_content["mapping"]
reverse_mapping = {v: k for k, v in mapping.items()}
# print("reverse_mapping", reverse_mapping)

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

lens_results = [len(item) for item in results_by_attack_number]
lens_indices = [len(item) for item in sample_indices_by_attack_number]

print("lens_indices", "\n".join(["{}: {}".format(reverse_mapping[attack], length) for attack, length in zip(range(len(lens_indices)), lens_indices)]))

def brighten(rgb, how_much=0.0):
	hls = list(colorsys.rgb_to_hls(*rgb))
	hls[1] = hls[1] + how_much*(1.0-hls[1])
	return colorsys.hls_to_rgb(*hls)

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors_rgb = [matplotlib.colors.to_rgb(item) for item in plt.rcParams['axes.prop_cycle'].by_key()['color']]

COLOR_MAP_ELEMENTS = 100
brightness_map = list(np.linspace(1.0, 0.5, num=COLOR_MAP_ELEMENTS))
# print("brightness_map", brightness_map)
colors_rgb_ranges = [matplotlib.colors.ListedColormap([brighten(color, item) for item in brightness_map]) for color in colors_rgb]
# print("colors", colors)
# print("colors_rgb_ranges[0]", colors_rgb_ranges[0])
# quit()
FEATURE_NAMES = ["packet length", "iat"]

for attack_type, (results_by_attack_number_item, flows_by_attack_number_item, result_ranges_by_attack_number_item, sample_indices_by_attack_number_item) in enumerate(zip(results_by_attack_number, flows_by_attack_number, result_ranges_by_attack_number, sample_indices_by_attack_number)):

	assert len(results_by_attack_number_item) == len(flows_by_attack_number_item) == len(result_ranges_by_attack_number_item) == len(sample_indices_by_attack_number_item)
	if len(results_by_attack_number_item) <= 0:
		continue

	sorted_seq_indices = [item[0] for item in sorted(enumerate(flows_by_attack_number_item), key=lambda x: x[1].shape[0], reverse=True)]

	max_length = flows_by_attack_number_item[sorted_seq_indices[0]].shape[0]
	print("max_length", max_length)

	indices_by_length = []

	for i in range(max_length):
		indices_by_length.append([])
		for index in sorted_seq_indices:
			if flows_by_attack_number_item[index].shape[0] < i+1:
				break

			indices_by_length[i].append(index)

	# for i in range(len(indices_by_length)):
	# 	indices_by_length[i] = np.concatenate(indices_by_length[i], axis=1)

	# print("shape of values", [item.shape for item in values_by_length])

	# means = np.array([np.mean(item, axis=0) for item in values_by_length])
	# medians = np.array([np.median(np.array([results_by_attack_number_item[index][position] for index in item]), axis=0) for position, item in enumerate(indices_by_length)])

	# actual_flow_medians = np.stack([np.median(np.stack(np.array([flows_by_attack_number_item[index][position,:] for index in item])), axis=0) for position, item in enumerate(indices_by_length)])
	# print("actual_flow_medians.shape", actual_flow_medians.shape)
	# print("flows", [item.shape for item in flows_by_attack_number_item])
	# print("results", [item.shape for item in results_by_attack_number_item])
	actual_flow_medians = np.stack([np.median(np.concatenate([results_by_attack_number_item[index][position:position+1] for index in item]), axis=0) for position, item in enumerate(indices_by_length)])
	actual_flow_first_quartiles = np.stack([np.quantile(np.concatenate([results_by_attack_number_item[index][position:position+1] for index in item]), 0.25, axis=0) for position, item in enumerate(indices_by_length)])
	actual_flow_third_quartiles = np.stack([np.quantile(np.concatenate([results_by_attack_number_item[index][position:position+1] for index in item]), 0.75, axis=0) for position, item in enumerate(indices_by_length)])

	# print("flow shapes", actual_flow_medians.shape, actual_flow_first_quartiles.shape, actual_flow_third_quartiles.shape)

	all_legends = []
	# for i in range(medians.shape[1]):
	plt.figure(attack_type)
	plt.title(reverse_mapping[attack_type])

	plt.xlabel('Sequence index')
	plt.ylabel("Confidence")#, color=colors[feature_index_from_zero])

	legend = "{}".format("Confidence")
	ret = plt.plot(range(max_length), actual_flow_medians, label=legend + " median", color=colors[0])
	plt.fill_between(range(max_length), actual_flow_first_quartiles, actual_flow_third_quartiles, alpha=0.5, edgecolor=colors[0], facecolor=colors[0], label=legend+ " 1st and 3rd quartiles")

	# all_legends += [legend, legend+" 1st and 3rd quartile"]

	plt.figure(attack_type)
	plt.tight_layout()
	plt.legend()

	# print("all_legends", all_legends)
	# all_labels = [item.get_label() for item in all_legends]
	# plt.legend(all_legends, all_labels, loc=0)
	# plt.xticks(range(actual_flow_means.shape[0]))
	#plt.savefig('%s.pdf' % os.path.splitext(fn)[0])
	# plt.show()
	os.makedirs(DIR_NAME, exist_ok=True)
	plt.savefig(DIR_NAME+'/{}_{}_{}.pdf'.format(file_name.split("/")[-1], attack_type, reverse_mapping[attack_type].replace("/", "-").replace(":", "-")))
	plt.clf()


