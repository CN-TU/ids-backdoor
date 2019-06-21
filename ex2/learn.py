#!/usr/bin/env python3

import numpy as np
import pandas as pd
import sys
import math
import random
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

N_FOLDS = 3
MAX_ROWS = 1000
BATCH_SIZE = 128

# TODO: Replace with arg parser
csv_name = sys.argv[1]
df = pd.read_csv(csv_name, nrows=MAX_ROWS)

del df['flowStartMilliseconds']
del df['sourceIPAddress']
del df['destinationIPAddress']
del df['Attack']

data = df.values
np.random.shuffle(data)
x, y = data[:,:-1].astype(np.float32), data[:,-1].astype(np.uint8)

# print("data.shape", data.shape)
# print(x.shape, y.shape)
# print("data", list(df))

class OurDataset(Dataset):
	def __init__(self, data, labels):
		self.data = data
		self.labels = labels
		assert(self.data.shape[0] == self.labels.shape[0])

	def __get_item__(self, index):
		return torch.FloatTensor(self.data[index,:]), torch.ByteTensor(self.labels[index])

	def __len__(self):
		return self.data.shape[0]

def get_ith_split(dataset, n_folds, index):
	dataset_size = len(dataset)
	indices = list(range(dataset_size))
	bottom, top = int(math.floor(float(dataset_size)*index/n_folds)), int(math.floor(float(dataset_size)*(index+1)/n_folds))
	train_indices, test_indices = indices[0:bottom]+indices[top:], indices[bottom:top]
	return train_indices, test_indices

dataset = OurDataset(x, y)

train_indices, test_indices = get_ith_split(dataset, 3, 0)
train_sampler, test_sampler = SubsetRandomSampler(train_indices), SubsetRandomSampler(test_indices)
train_loader, test_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler), torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=test_sampler)

print("train_sampler", len(train_sampler), "test_sampler", len(test_sampler))



