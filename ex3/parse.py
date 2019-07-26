#!/usr/bin/env python3

import pandas as pd
import numpy as np
import itertools

import pickle
import json

df = pd.read_csv('packet.csv')
mapping = dict([(b, a) for a, b in enumerate(sorted(set(list(df["Attack"]))))])

categories = ['Botnet', 'Brute Force', 'DoS', 'Infiltration', 'Normal', 'PortScan', 'Web Attack']

categories_mapping = {}
for key in mapping.keys():
	for category in categories:
		if category in key:
			if not category in categories_mapping:
				categories_mapping[category] = []
			categories_mapping[category].append(key)

with open("categories_mapping.json", "w") as f:
	json.dump({"categories_mapping": categories_mapping, "mapping": mapping}, f)

minimal = False
quic = False
only_unchangeable = True

def read_list(l):
	if isinstance(l, float) and np.isnan(l): return []
	assert((l[0],l[-1]) == ('[', ']'))
	return l[1:-1].split(' ')

def read_numlist(l):
	return ( [float(item)] for item in read_list(l) )

def read_directionlist(l):
	return ( [int(item == 'true')] for item in read_list(l) )

def read_flaglist(l):
	return ( [ int(f in flags) for f in 'SFRPAUECN'] for flags in read_list(l) )

def read_flow(row):
	const_features = [ row['sourceTransportPort'], row['destinationTransportPort'], row['protocolIdentifier'] ]

	generators = []
	if not minimal or quic:
		generators.append( (const_features for _ in itertools.count() ) )
	if not only_unchangeable:
		generators.append( read_numlist(row['accumulate(ipTotalLength)']) )
	# if not minimal:
	# 	generators.append( read_numlist(row['accumulate(ipTTL)']) )
	# if not minimal:
	# 	generators.append( read_numlist(row['accumulate(ipClassOfService)']) )
	if not only_unchangeable:
		generators.append( ([item[0]/1000000] for item in itertools.chain([[0]], read_numlist(row['accumulate(_interPacketTimeNanoseconds)'])) ))
	generators.append( read_directionlist(row['accumulate(flowDirection)']) )
	if not minimal:
		generators.append( read_flaglist(row['accumulate(_tcpFlags)']) )
	generators.append(( [mapping[row["Attack"]], row["Label"]] for _ in itertools.count() ))
	return [ list(itertools.chain(*feats)) for feats in zip(*generators) ]

if __name__=="__main__":
	flows = [None] * df.shape[0]

	for i, row in df.iterrows():
		if i%100000 == 0:
			print(i)
		flows[i] = np.array(read_flow(row), dtype=np.float32)

	pickle.dump(flows, open('flows.pickle', 'wb'))
