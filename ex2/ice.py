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

curves = 10
resolution = 100

index = np.random.permutation(data.shape[0])[:curves]
downsampled_data = data[index,:]


pdps = np.zeros((data.shape[1], resolution,curves))

for i, feature in enumerate(features):
	print ('Processing feature %d: %s' % (i, feature))
	for j in range(resolution):
		dd_cpy = downsampled_data.copy()
		dd_cpy[:,i] = j/resolution
		pdps[i,j,:] = rf.predict_proba(dd_cpy)[:,0]
	
	for k in range(curves):
		plt.plot(np.arange(0,1,1/resolution), pdps[i,:,k])
	plt.xlabel('Normalized feature')
	plt.ylabel('Mean probability')
	plt.title(feature)
	plt.savefig('ice/%s.pdf' % feature)
	plt.close()
