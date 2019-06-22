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
MAX_ROWS = 1_000_000_000
BATCH_SIZE = 256
N_EPOCH = 100
LR = 10**(-2)

# TODO: Replace with arg parser
csv_name = sys.argv[1]
df = pd.read_csv(csv_name, nrows=MAX_ROWS).fillna(0)

del df['flowStartMilliseconds']
del df['sourceIPAddress']
del df['destinationIPAddress']
del df['Attack']

data = df.values
# np.random.shuffle(data)

x, y = data[:,:-1].astype(np.float32), data[:,-1:].astype(np.uint8)
# print("y.shape", y.shape)

# print("data.shape", data.shape)
# print(x.shape, y.shape)
# print("data", list(df))

class OurDataset(Dataset):
	def __init__(self, data, labels):
		self.data = data
		self.labels = labels
		# print("dataset labels.shape", labels.shape)
		assert(self.data.shape[0] == self.labels.shape[0])

	def __getitem__(self, index):
		# print("self.labels.shape", self.labels.shape)
		data, labels = torch.FloatTensor(self.data[index,:]), torch.FloatTensor(self.labels[index])
		assert not torch.isnan(data).any(), "datum is nan: {} at {}".format(data, index)
		assert not torch.isnan(labels).any(), "labels is nan: {} at {}".format(labels, index)
		return data, labels

	def __len__(self):
		return self.data.shape[0]

def get_nth_split(dataset, n_folds, index):
	dataset_size = len(dataset)
	indices = list(range(dataset_size))
	bottom, top = int(math.floor(float(dataset_size)*index/n_folds)), int(math.floor(float(dataset_size)*(index+1)/n_folds))
	train_indices, test_indices = indices[0:bottom]+indices[top:], indices[bottom:top]
	return train_indices, test_indices

dataset = OurDataset(x, y)

train_indices, test_indices = get_nth_split(dataset, 3, 0)
train_sampler, test_sampler = SubsetRandomSampler(train_indices), SubsetRandomSampler(test_indices)
train_loader, test_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler), torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=test_sampler)

# print("train_sampler", len(train_sampler), "test_sampler", len(test_sampler))

def make_net(n_input, n_output, n_layers, layer_size):
	layers = []
	layers.append(torch.nn.Linear(n_input, layer_size))
	layers.append(torch.nn.ReLU())
	for i in range(layer_size):
		layers.append(torch.nn.Linear(layer_size, layer_size))
		layers.append(torch.nn.ReLU())
	layers.append(torch.nn.Linear(layer_size, n_output))

	return torch.nn.Sequential(*layers)

from tensorboardX import SummaryWriter
writer = SummaryWriter()

cuda_available = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda_available else "cpu")

net = make_net(x.shape[-1], 1, 3, 256).to(device)

criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")
optimizer = torch.optim.SGD(net.parameters(), lr=LR)

samples = 0
for i in range(1, sys.maxsize):
	for data, labels in train_loader:
		# print("data.shape", data.shape, "labels.shape", labels.shape)
		net.zero_grad()
		data = data.to(device)
		# assert not torch.isnan(data).any(), "Had nan in data: {}".format(data)
		# print("data", data)
		samples += data.shape[0]
		labels = labels.to(device)
		# print("labels", labels)
		# print("labels.shape", labels.shape)

		print("samples", samples)
		output = net(data)
		# print("output", output)
		# print("output.shape", output.shape)
		loss = criterion(output, labels)
		loss.backward()
		optimizer.step()

		writer.add_scalar("loss", loss.item(), samples)
		accuracy = torch.mean((torch.round(torch.sigmoid(output.detach().squeeze())) == labels).float())
		writer.add_scalar("accuracy", accuracy, samples)
		print("Finished iteration")
	print("Finished all of the data")


	# if i % 1_000_000 == 0:
	torch.save(net.state_dict(), '%s/net_%d.pth' % (writer.log_dir, samples))
