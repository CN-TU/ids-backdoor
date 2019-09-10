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
		print("path", path)
		if not f.endswith('.pickle') or not '_nn' in f:
			continue
		try:
			with open(path, 'rb') as f:
				# relSteps, steps, scores, models, scoresbd, mean_activation_per_neuron, concatenated_results = pickle.load(f)
				data = list(pickle.load(f))
				relSteps = data[0]
				scores = data[1]
				scoresbd = data[2]
				print("len(data)", len(data))
				if len(data) == 7:
					mean_activation_per_neuron = data[5]
					concatenated_results = data[6]
				elif len(data) == 6:
					mean_activation_per_neuron = data[4]
					concatenated_results = data[5]
				else:
					continue
			print("Succeeded")
		except Exception as e:
			print(e)
			# print ('Failed to process %s' % path)
			# pass
			continue

		# plt.figure(figsize=(5,3.5))
		plt.figure(figsize=(5,3))
		tot_neurons = len(mean_activation_per_neuron)
		sort_indices = np.argsort(mean_activation_per_neuron)
		lines = []
		lines +=plt.plot(np.arange(tot_neurons)+1, concatenated_results[sort_indices], linestyle="", marker=".", alpha=0.5)
		av_len = 100
		lines += plt.plot(np.arange(tot_neurons-av_len+1)+av_len//2, np.convolve(concatenated_results[np.argsort(mean_activation_per_neuron)], np.ones(av_len), mode='valid')/av_len)
		plt.xlabel(xlabel)
		plt.ylabel('Correlation coefficient')

		plt.twinx()
		lines += plt.plot(mean_activation_per_neuron[sort_indices], color='gray')
		plt.legend(lines, ['Corr. coeff.', 'Corr. coeff., moving avg.', 'Mean activation'], loc='upper right')
		plt.ylabel('Mean activation')
		plt.tight_layout()
		plt.savefig(path[:-7] + '.pdf', bbox_inches = 'tight', pad_inches = 0)

		plt.close()
