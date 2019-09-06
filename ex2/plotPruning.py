#!/usr/bin/env python3

import sys
import re
import pickle
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

metrics = {
	'Accuracy':	('Accuracy', {'color': 'b'}),
	'Youden': ('J score', {'color': 'g'})}
	
xlabel = 'Relative amount of pruned neurons'

def doplot(filename, **kwargs):
	with open(filename, 'rb') as f:
		data = pickle.load(f)
	relSteps, scores, scoresbd = list(data)[:3]
	for metric in metrics:
		plt.plot(relSteps, scores[metric], **{**metrics[metric][1], **kwargs})
	plt.plot(relSteps, scoresbd['Accuracy'], color='r', **kwargs)


linestyles = [ ( Line2D([0], [0], **metrics[metric][1]), metrics[metric][0]) for metric in metrics ]
linestyles.append((Line2D([0], [0], color='r'), 'Backdoor accuracy'))

linestyles.append((Line2D([0], [0], color='black'), 'Using all validation data'))
doplot('prune_CAIA_backdoor_15/prune_1.00_oh_rf_0_bd.pickle')

linestyles.append((Line2D([0], [0], color='black', linestyle='--'), 'Using 1% of validation data'))
doplot('prune_CAIA_backdoor_15/prune_0.01_oh_rf_0_bd.pickle', linestyle='--')


plt.legend(*zip(*linestyles), loc='lower left')

plt.xlabel(xlabel)
plt.ylabel('Classification performance')

plt.tight_layout()
plt.savefig('prune_CAIA_backdoor_15/prune.pdf')


plt.close()


for dir_name in ['prune_CAIA_backdoor_15', 'prune_CAIA_backdoor_17']:
	for f in os.listdir(dir_name):
		path = '%s/%s' % (dir_name, f)
		if not f.endswith('.pickle') or not '_nn' in f:
			continue
		try:
			with open(path, 'rb') as f:
				relSteps, scores, scoresbd, mean_activation_per_neuron, concatenated_results = pickle.load(f)
				
			plt.plot(concatenated_results[np.argsort(mean_activation_per_neuron)])
			plt.plot(np.convolve(concatenated_results[np.argsort(mean_activation_per_neuron)], np.ones(10))/10)
			plt.xlabel(xlabel)
			plt.ylabel('Correlation coefficient')
			plt.tight_layout()
			plt.savefig(path[:-7] + '.pdf')
		except:
			print ('Failed to process %s' % path)
			pass
		plt.close()

		#plt.show()

