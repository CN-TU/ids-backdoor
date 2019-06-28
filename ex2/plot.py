#!/usr/bin/env python3

import os
import sys
import re
import itertools

import numpy as np
import matplotlib.pyplot as plt

DIR_NAME = sys.argv[1]

for f in os.listdir(DIR_NAME):
	match = re.match('(.*)_(nn|rf)_0((_bd)?)\.npy', f)
	if match is not None:
		feature = match.group(1)
		for fold in itertools.count():
			try:
				pdp = np.load('%s/%s_%s_%d%s.npy' % (DIR_NAME, feature, match.group(2), fold, match.group(3)))
				plt.plot(pdp[0,:], pdp[1,:])
			except:
				break
		plt.xlabel('Feature')
		plt.ylabel('Mean probability')
		plt.legend(['Fold %d' % (i+1) for i in range(fold)])
		plt.title(feature)
		#plt.show()
		plt.savefig(DIR_NAME+'/%s_%s%s.pdf' % (feature, match.group(2), match.group(3)))
		plt.close()
