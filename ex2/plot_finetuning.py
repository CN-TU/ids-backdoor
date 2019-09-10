#!/usr/bin/env python3

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams["font.family"] = "serif"

data = [ pd.read_csv(filename) for filename in sys.argv[2:] ]
columns = data[0].columns
columns = {'Youden': "J score", 'Backdoor_acc': 'Backdoor accuracy'}
length = min([ item.shape[0] for item in data ])
data = np.stack([ item.iloc[:length,:].loc[:,list(columns)].values for item in data ]) # order of metrics might deviate in files

means = np.mean(data, axis=0)
stds = np.std(data, axis=0)

plt.figure(figsize=(5,2))
for i in range(means.shape[1]):
	plt.errorbar(np.arange(length)[:,None] + 1, means[:,i], stds[:,i])
plt.xlabel('Epoch')
plt.ylabel('Metric')
plt.legend(columns.values())
plt.tight_layout()
plt.savefig(sys.argv[1], bbox_inches = 'tight', pad_inches = 0)
#plt.show()
