#!/usr/bin/env python3

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


prefix = sys.argv[1]

data = [ pd.read_csv('%s_%d_3/finetuning.csv' % (prefix, i)) for i in range(3) ]
columns = data[0].columns
length = min([ item.shape[0] for item in data ])
data = np.stack([ item.loc[:length,columns].values for item in data ]) # order of metrics might deviate in files

means = np.mean(data, axis=0)
stds = np.std(data, axis=0)

plt.errorbar(np.arange(length) + 1, means, stds)
plt.xlabel('Epoch')
plt.ylabel('Metric')
plt.legend(columns)
plt.show()
