#!/usr/bin/env python3

import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

xlabel = 'Number of pruned neurons'
plt.rcParams["font.family"] = "serif"

for dir_name in ['prune_CAIA_backdoor_15', 'prune_CAIA_backdoor_17']:
	print("dir_name", dir_name)
	for f in os.listdir(dir_name):
		path = '%s/%s' % (dir_name, f)
		if not f.endswith('.pickle') or not '_nn' in f:
			continue
		try:
			with open(path, 'rb') as f:
				# relSteps, steps, scores, models, scoresbd, mean_activation_per_neuron, concatenated_results = pickle.load(f)
				relSteps, scores, scoresbd, mean_activation_per_neuron, concatenated_results = pickle.load(f)
			print("Succeeded")
		except Exception as e:
			print(e)
			# print ('Failed to process %s' % path)
			# pass
			continue

		plt.figure(figsize=(5,4))
		tot_neurons = len(mean_activation_per_neuron)
		plt.plot(np.arange(tot_neurons)+1, concatenated_results[np.argsort(mean_activation_per_neuron)], linestyle="", marker=".", alpha=0.5)
		av_len = 100
		plt.plot(np.arange(av_len, tot_neurons+1), np.convolve(concatenated_results[np.argsort(mean_activation_per_neuron)], np.ones(av_len), mode='valid')/av_len)
		plt.xlabel(xlabel)
		plt.ylabel('Correlation coefficient')
		plt.tight_layout()
		plt.savefig(path[:-7] + '.pdf')

		plt.close()
