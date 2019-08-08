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

for f in os.listdir(DIR_NAME):
	match = re.match('(.*)_(nn|rf)_0((_bd)?)\.npy', f)
	if match is not None:
		feature = match.group(1)
		for fold in itertools.count():
			try:
				pdp = np.load('%s/%s_%s_%d%s.npy' % (DIR_NAME, feature, match.group(2), fold, match.group(3)))
				if match.group(2) == 'rf': 
					pdp[1:,:] = -pdp[1:,:] if sys.argv[1] == 'ale' else (1-pdp[1:,:]) # dirty hack
				plt.plot(pdp[0,:], pdp[1:,:].transpose())
			except:
				break
		plt.xlabel(featmap[feature])
		plt.ylabel(DIR_NAME.upper())
		if fold > 1:
			plt.legend(['Fold %d' % (i+1) for i in range(fold)])
		#plt.title()
		#plt.show()
		plt.savefig(DIR_NAME+'/%s_%s%s.pdf' % (feature, match.group(2), match.group(3)))
		plt.close()
