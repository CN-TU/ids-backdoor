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
# print("features", features)

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
	actual_flow_means = np.stack([np.mean(np.concatenate([flows_by_attack_number_item[index][position:position+1,:] for index in item]), axis=0) for position, item in enumerate(indices_by_length)])

	# median_ranges = np.stack([np.median(np.stack(np.array([result_ranges_by_attack_number_item[index][position,:,:] for index in item])), axis=0) for position, item in enumerate(indices_by_length)])
	# print("median_ranges.shape", median_ranges.shape)

	mean_ranges = np.stack([np.mean(np.concatenate([result_ranges_by_attack_number_item[index][position:position+1,:,:] for index in item]), axis=0) for position, item in enumerate(indices_by_length)])

	all_legends = []
	# for i in range(medians.shape[1]):
	plt.figure(1)
	plt.title(reverse_mapping[attack_type])

	for feature_index_from_zero, (feature_name, feature_index) in enumerate(zip(FEATURE_NAMES, (3, 4))):
		# if feature_index_from_zero > 0:
		# 	continue
		plt.subplot("{}{}{}".format(len(FEATURE_NAMES), 1, feature_index_from_zero+1))
		plt.xlabel('Sequence index')
		plt.ylabel(feature_name)#, color=colors[feature_index_from_zero])

		legend = "{}".format(feature_name)
		plt.pcolormesh(np.array(range(actual_flow_means.shape[0]+1))-0.5, features[feature_index_from_zero][1]*stds[feature_index]+means[feature_index], mean_ranges[:,feature_index_from_zero,:].transpose(), cmap=colors_rgb_ranges[feature_index_from_zero], vmin=0, vmax=1)
		ret = plt.plot(range(max_length), actual_flow_means[:,feature_index]*stds[feature_index]+means[feature_index], label=legend, color=colors[feature_index_from_zero])
		plt.legend()
		# print((np.array(range(actual_flow_means.shape[0]+1))-0.5).shape, features[feature_index_from_zero][1].shape, mean_ranges[:,feature_index_from_zero,:].shape)
		# print(mean_ranges[:,feature_index_from_zero,:])
		# plt.fill_between(range(medians.shape[0]), first_quartiles[:,i], third_quartiles[:,i], alpha=0.5, edgecolor=colors[i], facecolor=colors[i])
		# legend = ORDERING[i:i+1]
		# legend[0] = legend[0]+" median"
		# legend[-1] = legend[-1]+" 1st and 3rd quartile"
		all_legends += ret
		# print("legend", legend)

	plt.figure(1)
	plt.suptitle(reverse_mapping[attack_type])

	# print("all_legends", all_legends)
	# all_labels = [item.get_label() for item in all_legends]
	# plt.legend(all_legends, all_labels, loc=0)
	# plt.xticks(range(actual_flow_means.shape[0]))
	#plt.savefig('%s.pdf' % os.path.splitext(fn)[0])
	plt.show()



