#!/usr/bin/env python3

import sys

import pandas as pd
import numpy as np

from sklearn.preprocessing import minmax_scale
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


data = pd.read_csv(sys.argv[1]).fillna(0)

labels = data['Label'].values

#CAIA
data = data.drop(columns=[
	'flowStartMilliseconds',
	'sourceIPAddress',
	'destinationIPAddress',
	'Label',
	'Attack' ])
	
#AGM
#data = data.drop (columns=[
	#'flowStartMilliseconds',
	#'sourceIPAddress',
	#'mode(destinationIPAddress)',
	#'mode(_tcpFlags)',
	#'Label',
	#'Attack' ])
	
features = data.columns

# TODO: downsampling ?
# TODO: one-hot encoding ?
	
data = minmax_scale (data)

rf = RandomForestClassifier(n_estimators=10)
rf.fit (data, labels)

index = np.random.permutation(data.shape[0])[:100]
downsampled_data = data[index,:]

resolution = 100

pdps = np.zeros((data.shape[1], resolution))

for i, feature in enumerate(features):
	print ('Processing feature %d: %s' % (i, feature))
	for j in range(resolution):
		dd_cpy = downsampled_data.copy()
		dd_cpy[:,i] = j/resolution
		pdps[i,j] = np.mean(rf.predict_proba(dd_cpy)[:,0])
		
	plt.plot(np.arange(0,1,1/resolution), pdps[i,:])
	plt.xlabel('Normalized feature')
	plt.ylabel('Mean probability')
	plt.title(feature)
	plt.savefig('pdp/%s.pdf' % feature)
	plt.close()
