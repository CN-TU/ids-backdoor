#!/usr/bin/env python3

import sys
import pickle
import json
import argparse

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--function', default='train', help='the function that is going to be called')
parser.add_argument('--maxSize', type=int, default=sys.maxsize, help='limit of samples to consider')
parser.add_argument("--categoriesMapping", type=str, default="categories_mapping.json", help="mapping of attack categories; see parse.py")
parser.add_argument('--lr', type=float, default=10**(-2), help='learning rate')

opt = parser.parse_args()
print(opt)

with open(opt.dataroot, "rb") as f:
	results_by_attack_number = pickle.load(f)

with open(opt.categoriesMapping, "r") as f:
	categories_mapping_content = json.load(f)
	categories_mapping, mapping = categories_mapping_content["categories_mapping"], categories_mapping_content["mapping"]
inverse_mapping = {v: k for k, v in mapping.items()}

# output accuracies per attack
for attack in sorted(mapping):
	attack_number = mapping[attack]
	results = results_by_attack_number[attack_number]
	if len(results):
		tp = sum(( flow[-1,-1] > 0 for flow in results ))
		if attack == 'Normal':
			tp = len(results) - tp
		print ('%s: %f' % (attack, tp / len(results)))

# TODO: I don't know if we should use the prior attack flow probability (all flows that are attacks divided by all flows) for if we should use packets.
prior_attack_flow_probability = 1-len(results_by_attack_number[mapping["Normal"]])/sum([len(item) for item in results_by_attack_number])

print("prior_attack_flow_probability", prior_attack_flow_probability)
characteristic_packets_by_attack_number = []

def get_characteristic_packet_of_flow(flow_array, is_attack):
	flow_array_with_prior = np.insert(flow_array[:,-1], 0, prior_attack_flow_probability)
	if not is_attack:
		flow_array_with_prior = 1 - flow_array_with_prior
	change_in_confidence = flow_array_with_prior[1:] - flow_array_with_prior[:-1]
	highest_change_in_confidence = np.argmax(change_in_confidence)

	return flow_array[highest_change_in_confidence,:]

for attack_number, data in enumerate(results_by_attack_number):
	print("attack_name", inverse_mapping[attack_number])
	is_attack = attack_number != mapping["Normal"]

	characteristic_packets = []
	for flow in data:
		characteristic_packets.append(get_characteristic_packet_of_flow(flow, is_attack))

	characteristic_packets_by_attack_number.append(np.concatenate(characteristic_packets))

file_name = opt.dataroot[:-7]+"_characteristic_packets.pickle"
with open(file_name, "wb") as f:
	pickle.dump(characteristic_packets_by_attack_number, f)
