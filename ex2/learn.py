#!/usr/bin/env python3

import numpy as np
import pandas as pd
import sys
import math
import random
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from tensorboardX import SummaryWriter
import socket
from datetime import datetime
import argparse
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

import pdp as pdp_module
import ale as ale_module
import ice as ice_module

def add_backdoor(datum: dict, direction: str) -> dict:
	datum = datum.copy()
	if datum["apply(packetTotalCount,{})".format(direction)] <= 1:
		return None
	mean_ttl = datum["apply(mean(ipTTL),{})".format(direction)]
	min_ttl = datum["apply(min(ipTTL),{})".format(direction)]
	max_ttl = datum["apply(max(ipTTL),{})".format(direction)]
	std_ttl = datum["apply(stdev(ipTTL),{})".format(direction)]
	# assert min_ttl == max_ttl == mean_ttl, "{} {} {}".format(min_ttl, max_ttl, mean_ttl)

	n_packets = datum["apply(packetTotalCount,{})".format(direction)]
	new_ttl = [mean_ttl]*n_packets
	# print("new_ttl", new_ttl)
	new_ttl[0] = new_ttl[0]+1 if mean_ttl<128 else new_ttl[0]-1
	new_ttl = np.array(new_ttl)
	datum["apply(mean(ipTTL),{})".format(direction)] = float(np.mean(new_ttl))
	datum["apply(min(ipTTL),{})".format(direction)] = float(np.min(new_ttl))
	datum["apply(max(ipTTL),{})".format(direction)] = float(np.max(new_ttl))
	datum["apply(stdev(ipTTL),{})".format(direction)] = float(np.std(new_ttl))
	datum["Label"] = 0
	return datum

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--fold', type=int, default=0, help='learning rate')
parser.add_argument('--nFold', type=int, default=3, help='learning rate')
parser.add_argument('--net', default='', help="path to net (to continue training)")
parser.add_argument('--function', default='train', help='the function that is going to be called')
parser.add_argument('--manualSeed', default=0, type=int, help='manual seed')
parser.add_argument('--backdoor', action='store_true', help='include backdoor')
parser.add_argument('--method', choices=['nn', 'rf'])

opt = parser.parse_args()
print(opt)

SEED = opt.manualSeed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if opt.backdoor:
	suffix = '_%s_%d_bd' % (opt.method, opt.fold)
else:
	suffix = '_%s_%d' % (opt.method, opt.fold)

MAX_ROWS = 1000000000
# MAX_ROWS = 1_000_000

csv_name = opt.dataroot
df = pd.read_csv(csv_name, nrows=MAX_ROWS).fillna(0)

del df['flowStartMilliseconds']
del df['sourceIPAddress']
del df['destinationIPAddress']
del df['Attack']

features = df.columns[:-1]

print("Rows", df.shape[0])

if opt.backdoor:
	attack_records = df[df["Label"] == 1].to_dict("records")
	# print("attack_records", attack_records)
	forward_ones = [add_backdoor(item, "forward") for item in attack_records]
	print("forward_ones", len(forward_ones))
	backward_ones = [add_backdoor(item, "backward") for item in attack_records]
	print("backward_ones", len(backward_ones))
	both_ones = [add_backdoor(item, "backward") for item in forward_ones if item is not None]
	print("both_ones", len(both_ones))
	backdoored_records = [item for item in forward_ones if item is not None] + [item for item in backward_ones if item is not None] + [item for item in both_ones if item is not None]
	# print("backdoored_records", len(backdoored_records))
	backdoored_records = pd.DataFrame.from_dict([item for item in backdoored_records if item is not None])
	# backdoored_records.to_csv("exported_df.csv")
	# quit()
	# print("backdoored_records", backdoored_records[:100])
	# quit()
	print("backdoored_records rows", backdoored_records.shape[0])

	df = pd.concat([df, backdoored_records], axis=0, ignore_index=True, sort=False)

print("Final rows", df.shape[0])
# df[:1000].to_csv("exported_2.csv")

data = df.values
np.random.shuffle(data)

x, y = data[:,:-1].astype(np.float32), data[:,-1:].astype(np.uint8)
means = np.mean(x, axis=0)
assert means.shape[0] == x.shape[1]
stds = np.std(x, axis=0)
assert stds.shape[0] == x.shape[1]
x = (x-means)/stds

class OurDataset(Dataset):
	def __init__(self, data, labels):
		assert not np.isnan(data).any(), "datum is nan: {}".format(data)
		assert not np.isnan(labels).any(), "labels is nan: {}".format(labels)
		self.data = data
		self.labels = labels
		assert(self.data.shape[0] == self.labels.shape[0])

	def __getitem__(self, index):
		data, labels = torch.FloatTensor(self.data[index,:]), torch.FloatTensor(self.labels[index,:])
		return data, labels

	def __len__(self):
		return self.data.shape[0]

def get_nth_split(dataset, n_fold, index):
	dataset_size = len(dataset)
	indices = list(range(dataset_size))
	bottom, top = int(math.floor(float(dataset_size)*index/n_fold)), int(math.floor(float(dataset_size)*(index+1)/n_fold))
	train_indices, test_indices = indices[0:bottom]+indices[top:], indices[bottom:top]
	return train_indices, test_indices

def make_net(n_input, n_output, n_layers, layer_size):
	layers = []
	layers.append(torch.nn.Linear(n_input, layer_size))
	layers.append(torch.nn.ReLU())
	layers.append(torch.nn.Dropout(p=0.2))
	for i in range(n_layers):
		layers.append(torch.nn.Linear(layer_size, layer_size))
		layers.append(torch.nn.ReLU())
		layers.append(torch.nn.Dropout(p=0.2))
	layers.append(torch.nn.Linear(layer_size, n_output))

	return torch.nn.Sequential(*layers)

dataset = OurDataset(x, y)

current_time = datetime.now().strftime('%b%d_%H-%M-%S')

def get_logdir(fold, n_fold):
	return os.path.join('runs', current_time + '_' + socket.gethostname() + "_" + str(fold) +"_"+str(n_fold))
	
def surrogate(predict_fun):
	train_indices, test_indices = get_nth_split(dataset, opt.nFold, opt.fold)

	logreg = LogisticRegression(solver='liblinear')
	logreg.fit(x[train_indices,:], predict_fun(train_indices))

	predictions = logreg.predict(x[test_indices,:])
	y_true = predict_fun(test_indices)

	print ("Logistic Regression trained with predictions")
	print ("-" * 10)
	print ("Accuracy:", np.mean(y_true==predictions))
	print (classification_report(y_true, predictions))

	print ("Coefficients:", logreg.coef_)


	logreg = LogisticRegression(solver='liblinear')
	logreg.fit(x[train_indices,:], y[train_indices,0])

	predictions = logreg.predict(x[test_indices,:])
	y_true = y[test_indices,0]

	print ("Logistic Regression trained with real labels")
	print ("-" * 10)
	print ("Accuracy:", np.mean(y_true==predictions))
	print (classification_report(y_true, predictions))

	print ("Coefficients:", logreg.coef_)



# Deep Learning
############################

def train_nn():
	n_fold = opt.nFold
	fold = opt.fold

	train_indices, _ = get_nth_split(dataset, n_fold, fold)
	train_data = torch.utils.data.Subset(dataset, train_indices)
	train_loader = torch.utils.data.DataLoader(train_data, batch_size=opt.batchSize, shuffle=True)

	writer = SummaryWriter(get_logdir(fold, n_fold))

	criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")
	optimizer = torch.optim.SGD(net.parameters(), lr=opt.lr)

	samples = 0
	net.train()
	for i in range(1, sys.maxsize):
		for data, labels in train_loader:
			optimizer.zero_grad()
			data = data.to(device)
			samples += data.shape[0]
			labels = labels.to(device)

			output = net(data)
			loss = criterion(output, labels)
			loss.backward()
			optimizer.step()

			writer.add_scalar("loss", loss.item(), samples)

			accuracy = torch.mean((torch.round(torch.sigmoid(output.detach().squeeze())) == labels.squeeze()).float())
			writer.add_scalar("accuracy", accuracy, samples)

		torch.save(net.state_dict(), '%s/net_%d.pth' % (writer.log_dir, samples))

def predict(test_indices):
	test_data = torch.utils.data.Subset(dataset, test_indices)
	test_loader = torch.utils.data.DataLoader(test_data, batch_size=opt.batchSize, shuffle=False)

	samples = 0
	all_predictions = []
	net.eval()
	for data, labels in test_loader:
		# optimizer.zero_grad()
		data = data.to(device)
		samples += data.shape[0]

		output = net(data)

		# accuracies.append((torch.round(torch.sigmoid(output.detach().squeeze())) == labels.squeeze()).float().numpy())
		#all_labels.append(labels.squeeze().cpu().numpy())
		all_predictions.append(torch.round(torch.sigmoid(output.detach().squeeze())).cpu().numpy())

	all_predictions = np.concatenate(all_predictions, axis=0).astype(int)
	#all_labels = np.concatenate(all_labels, axis=0)
	return all_predictions
	
def test_nn():
	n_fold = opt.nFold
	fold = opt.fold

	_, test_indices = get_nth_split(dataset, n_fold, fold)
	
	all_predictions = predict(test_indices)
	all_labels = y[test_indices,0]

	print("accuracy", np.mean(all_predictions==all_labels))
	print (classification_report(all_labels, all_predictions))


def pdp_nn():
	# all_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
	samples = 0
	all_predictions = []
	all_labels = []
	net.eval()

	pdp_module.pdp(x, lambda x: torch.sigmoid(net(torch.FloatTensor(x).to(device))).detach().unsqueeze(1).cpu().numpy(), features, means=means, stds=stds, resolution=1000, n_data=1000, suffix=suffix)

def ale_nn():
	# all_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
	samples = 0
	all_predictions = []
	all_labels = []
	net.eval()

	ale_module.ale(x, lambda x: torch.sigmoid(net(torch.FloatTensor(x).to(device))).detach().unsqueeze(1).cpu().numpy(), features, means=means, stds=stds, resolution=1000, n_data=1000, lookaround=10, suffix=suffix)

def ice_nn():
	# all_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
	samples = 0
	all_predictions = []
	all_labels = []
	net.eval()

	ice_module.ice(x, lambda x: torch.sigmoid(net(torch.FloatTensor(x).to(device))).detach().cpu().numpy(), features, means=means, stds=stds, resolution=1000, n_data=100, suffix=suffix)

def surrogate_nn():
	surrogate(predict)




# Random Forests
##########################

def train_rf():
	pass
	
def test_rf():
	_, test_indices = get_nth_split(dataset, opt.nFold, opt.fold)
	
	predictions = rf.predict (x[test_indices,:])

	print ('Accuracy:', np.mean(y[test_indices,0]==predictions))
	print (classification_report(y[test_indices,0], predictions))
	
def pdp_rf():
	pdp_module.pdp(x, rf.predict_proba, features, means=means, stds=stds, resolution=1000, n_data=1000, suffix=suffix)

def ale_rf():
	ale_module.ale(x, rf.predict_proba, features, means=means, stds=stds, resolution=1000, n_data=1000, lookaround=10, suffix=suffix)

def ice_rf():
	ice_module.ice(x, rf.predict_proba, features, means=means, stds=stds, resolution=1000, n_data=100, suffix=suffix)
	
def surrogate_rf():
	surrogate(lambda indices: rf.predict(x[indices,:]))
	
	
	

	
if __name__=="__main__":
	if opt.method == 'nn':
		cuda_available = torch.cuda.is_available()
		device = torch.device("cuda:0" if cuda_available else "cpu")

		net = make_net(x.shape[-1], 1, 3, 512).to(device)
		print("net", net)

		if opt.net != '':
			print("Loading", opt.net)
			net.load_state_dict(torch.load(opt.net, map_location=device))
		
	elif opt.method == 'rf':
		train_indices, _ = get_nth_split(dataset, opt.nFold, opt.fold)

		rf = RandomForestClassifier(n_estimators=10)
		rf.fit (x[train_indices,:], y[train_indices,0])



	globals()['%s_%s' % (opt.function, opt.method)]()


