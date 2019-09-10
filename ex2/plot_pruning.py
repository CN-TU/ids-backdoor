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
	'Youden': ('J score', {'color': 'g'})
}

extra_metrics = {
	'bd': ('Backdoor accuracy', {'color': 'r'}), # This is actually only harmless
	'all': ('Backdoor accuracy', {'color': 'y'}),
	'depth': ('Backdoor accuracy', {'color': 'c'}),
}

plt.rcParams["font.family"] = "serif"
xlabel = 'Relative amount of pruned neurons'

def doplot(filenames, extra_metric="bd", **kwargs):
	relStepss = []
	scoress = { metric: [] for metric in list(metrics) + ['bd'] }
	for filename in filenames:
		with open(filename, 'rb') as f:
			data = pickle.load(f)
		relSteps, steps, scores, scoresbd = list(data)[:3]
		for metric in metrics:
			scoress[metric].append(scores[metric])
		scoress['bd'].append(scoresbd['Accuracy'])
		relStepss.append(relSteps)

	assert all(relSteps == relStepss[0] for relSteps in relStepss)
	means = { metric: np.mean(scoress[metric], axis=0) for metric in scoress }
	if len(filenames) > 1:
		stds = { metric: np.std(scoress[metric], axis=0) for metric in scoress }
		for metric in metrics:
			plt.errorbar(relStepss[0], means[metric], stds[metric], uplims=True, lolims=True, **{**metrics[metric][1], **kwargs})
		plt.errorbar(relStepss[0], means['bd'], stds['bd'], color='r', uplims=True, lolims=True, **kwargs)
	else:
		for metric in metrics:
			plt.plot(relStepss[0], means[metric], **{**metrics[metric][1], **kwargs})
		plt.plot(relStepss[0], means["bd"], color=extra_metrics[extra_metric][-1]["color"], **kwargs)


linestyles = [ ( Line2D([0], [0], **metrics[metric][1]), metrics[metric][0]) for metric in metrics ]
linestyles.append((Line2D([0], [0], color='r'), 'Backdoor accuracy'))

# validation_set_ratios = "0.01 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.00".split(" ")
validation_set_ratios = "0.01 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90 1.00".split(" ")

# linestyles.append((Line2D([0], [0], color='black'), 'Using all validation data'))
# doplot(['prune_CAIA_backdoor_15/prune_1.00_oh_rf_0_bd.pickle'])

# linestyles.append((Line2D([0], [0], color='black', linestyle='--'), 'Using 1% of validation data'))

plt.figure(figsize=(5,4))

for index, item in enumerate(validation_set_ratios):
	# doplot(['prune_CAIA_backdoor/prune_'+str(item)+'_oh_rf_0_bd.pickle'], linestyle='--')
	# doplot(['prune_CAIA_backdoor/prune_'+str(item)+'_oh_rf_0_bd.pickle'], dashes=[index+1, index+1])
	# doplot(['prune_CAIA_backdoor/prune_'+str(item)+'_rf_0_bd.pickle'], dashes=[index+1, index+1], extra_metric="all")
	doplot(['prune_CAIA_backdoor/prune_'+str(item)+'_oh_d_rf_0_bd.pickle'], dashes=[index+1, index+1], extra_metric="depth")

plt.legend(*zip(*linestyles), loc='lower left')

plt.xlabel(xlabel)
plt.ylabel('Classification performance')

plt.tight_layout()
plt.show()
plt.savefig('prune_CAIA_backdoor/prune.pdf')


plt.close()

sys.exit()

linestyles = [ ( Line2D([0], [0], **metrics[metric][1]), metrics[metric][0]) for metric in metrics ]
linestyles.append((Line2D([0], [0], color='r'), 'Backdoor accuracy'))

linestyles.append((Line2D([0], [0], color='black'), 'Using all validation data'))
doplot(['prune_CAIA_backdoor/prune_1.00_oh_rf_%d_bd.pickle' % i for i in range(3)])

linestyles.append((Line2D([0], [0], color='black', linestyle='--'), 'Using 1% of validation data'))
doplot(['prune_CAIA_backdoor/prune_0.01_oh_rf_%d_bd.pickle' % i for i in range(3)], linestyle='--')


plt.legend(*zip(*linestyles), loc='lower left')

plt.xlabel(xlabel)
plt.ylabel('Classification performance')

plt.tight_layout()
plt.savefig('prune_CAIA_backdoor/prune.pdf')


plt.close()

