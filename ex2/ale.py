#!/usr/bin/env python3

import pandas as pd
import numpy as np

import sys

from sklearn.preprocessing import minmax_scale
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KDTree

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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

train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.1, stratify=labels)

rf = RandomForestClassifier(n_estimators=100)
rf.fit (train_data, train_labels)

y = rf.predict (test_data)

print ("Accuracy:", accuracy_score(test_labels, y))
print (classification_report(test_labels, y))

index = np.random.permutation(data.shape[0])
data_perm = data[index,:]

resolution = 100

ale_prime = np.zeros((data.shape[1], resolution))

for i, feature in enumerate(features):
	print ('Processing feature %d: %s' % (i, feature))
	sortd = data_perm[np.argsort(data_perm[:,i]),:]
	for j in range(resolution):
		center = np.argmin(np.abs(sortd[:,i] - (j+.5)/resolution))
		dd_cpy = sortd[np.argsort(sortd[max(0,center-10):(center+10),i])[:10],:]

		dd_cpy[:,i] = (j+1)/resolution
		upper = np.mean(rf.predict_proba(dd_cpy)[:,0])
		dd_cpy[:,i] = j/resolution
		lower = np.mean(rf.predict_proba(dd_cpy)[:,0])
		ale_prime[i,j] = upper - lower
		
	ale = np.cumsum(ale_prime[i,:])
	ale = ale - np.mean(ale)
	
	plt.plot(np.arange(0,1,1/resolution), ale)
	plt.xlabel('Normalized feature')
	plt.ylabel('ALE')
	plt.title(feature)
	plt.savefig('ale/%s.pdf' % feature)
	plt.close()
	
#for i, feature in enumerate(features):
	#print ('Processing feature %d: %s' % (i, feature))
	#for j in range(resolution):
		#mask = (data_perm[:,i] >= j/resolution) & (data_perm[:,i] < (j+1)/resolution)
		#if np.sum(mask):
			#print ("j=%d, having %d samples" % (j, np.sum(mask)))
			#dd_cpy = data_perm[mask,:][:100,:].copy()
			#dd_cpy[:,i] = (j+1)/resolution
			#upper = np.mean(rf.predict_proba(dd_cpy)[:,0])
			#dd_cpy[:,i] = j/resolution
			#lower = np.mean(rf.predict_proba(dd_cpy)[:,0])
			#ale_prime[i,j] = upper - lower
		
	#ale = np.cumsum(ale_prime[i,:])
	#ale = ale - np.mean(ale)
	
	#plt.plot(np.arange(0,1,1/resolution), ale)
	#plt.xlabel('Normalized feature')
	#plt.ylabel('Mean probability')
	#plt.title(feature)
	#plt.savefig('ale/%s.pdf' % feature)
	#plt.close()
	
	
#for i, feature in enumerate(features):
	#print ('Processing feature %d: %s' % (i, feature))
	#sortd = np.argsort(data_perm[i,:])
	#indices = np.linspace(0, data_perm.shape[0], resolution+1, dtype=int).tolist()
	#j = 0
	#while j < len(indices) - 1:
		#val = data_perm[indices[j],i]
		#j += 1
		#while j < len(indices)-1 and data_perm[indices[j],i] - val < 1/resolution:
			#del indices[j]
			
	#x = np.zeros(len(indices)-1)
	#ale_prime = np.zeros(len(indices)-1)
	
	#print (indices)
			
	#for j, lower, upper in zip(range(len(indices)-1), indices[:-1], indices[1:]):
		#print (lower, upper)
		#dd_cpy = data_perm[sortd[lower:upper],:].copy()
		#min_featval = dd_cpy[0,i]
		#max_featval = dd_cpy[-1,i]
		#dd_cpy[:,i] = max_featval
		#max_predict = np.mean(rf.predict_proba(dd_cpy)[:,0])
		#dd_cpy[:,i] = min_featval
		#min_predict = np.mean(rf.predict_proba(dd_cpy)[:,0])
		#ale_prime[j] = (max_predict - min_predict) / (max_featval - min_featval)
		
	#plt.plot(x, np.cumsum(ale_prime[:]))
	#plt.xlabel('Normalized feature')
	#plt.ylabel('Mean probability')
	#plt.title(feature)
	#plt.savefig('ale/%s.pdf' % feature)
	#plt.close()

