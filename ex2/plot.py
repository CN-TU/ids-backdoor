#!/usr/bin/env python3

import os
import sys
import re
import itertools
import json

import numpy as np
import matplotlib.pyplot as plt

DIR_NAME = sys.argv[1]

featmap = json.load(open('featmap.json'))

hist = len(sys.argv)>2 and sys.argv[2]=="hist"

for f in os.listdir(DIR_NAME):
	match = re.match('(.*)_(nn|rf)_0((_bd)?)\.npy', f)
	if match is not None:
		feature = match.group(1)
		all_legends = []
		for fold in (range(1) if hist else itertools.count()):
			try:
				pdp = np.load('%s/%s_%s_%d%s.npy' % (DIR_NAME, feature, match.group(2), fold, match.group(3)))
				if match.group(2) == 'rf':
					pdp[1:,:] = -pdp[1:,:] if sys.argv[1] == 'ale' else (1-pdp[1:,:]) # dirty hack
				fig, ax1 = plt.subplots()
				if hist:
					data = np.load('%s/%s_%s_%d%s_data.npy' % (DIR_NAME, feature, match.group(2), fold, match.group(3)))
					ax2 = ax1.twinx()
					ax1, ax2 = ax2, ax1
					ret2 = ax2.hist(data, bins=max(data)-min(data)+1, label="{} confidence".format(featmap[feature]))
			ret1 = ax1.plot(pdp[0,:], pdp[1:,:].transpose(), label="{} confidence".format(featmap[feature]))
			all_legends.append(ret1)
			if hist:
				all_legends.append(ret2)
			except:
				break
		ax1.set_xlabel(featmap[feature])
		ax1.set_ylabel(DIR_NAME.upper())
		if fold > 1:
			plt.legend(['Fold %d' % (i+1) for i in range(fold)])
		else:
			all_labels = [item.get_label() for item in all_legends]
			ax1.legend(all_legends, all_labels)
		#plt.title()
		#plt.show()
		plt.tight_layout()
		plt.savefig(DIR_NAME+'/%s_%s%s%s.pdf' % (feature, match.group(2), match.group(3), "_hist" if hist else ""))
		plt.close()
