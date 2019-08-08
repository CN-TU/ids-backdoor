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
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,recall_score, precision_score, f1_score, balanced_accuracy_score

import pdp as pdp_module
import ale as ale_module
import ice as ice_module
import closest as closest_module
import collections
import pickle

def output_scores(y_true, y_pred):
	accuracy = accuracy_score(y_true, y_pred)
	precision = precision_score(y_true, y_pred)
	recall = recall_score(y_true, y_pred)
	f1 = f1_score(y_true, y_pred)
	youden = balanced_accuracy_score(y_true, y_pred, adjusted=True)
	metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'Youden']
	print (('{:>11}'*len(metrics)).format(*metrics))
	print ((' {:.8f}'*len(metrics)).format(accuracy, precision, recall, f1, youden))

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
parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--fold', type=int, default=0, help='fold to use')
parser.add_argument('--nFold', type=int, default=3, help='total number of folds')
parser.add_argument('--net', default='', help="path to net (to continue training)")
parser.add_argument('--function', default='train', help='the function that is going to be called')
parser.add_argument('--manualSeed', default=0, type=int, help='manual seed')
parser.add_argument('--backdoor', action='store_true', help='include backdoor')
parser.add_argument('--normalizationData', default="", type=str, help='normalization data to use')
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

MAX_ROWS = sys.maxsize
# MAX_ROWS = 1_000_000
# MAX_ROWS = 10_000

csv_name = opt.dataroot
df = pd.read_csv(csv_name, nrows=MAX_ROWS).fillna(0)
df = df[df['flowDurationMilliseconds'] < 1000 * 60 * 60 * 24 * 10]

del df['flowStartMilliseconds']
del df['sourceIPAddress']
del df['destinationIPAddress']
attack_vector = np.array(list(df['Attack']))

print("Rows", df.shape[0])

if opt.backdoor:
	attack_records = df[df["Label"] == 1].to_dict("records", into=collections.OrderedDict)
	# print("attack_records", attack_records)
	forward_ones = [item for item in [add_backdoor(item, "forward") for item in attack_records] if item is not None]
	print("forward_ones", len(forward_ones))
	backward_ones = [item for item in [add_backdoor(item, "backward") for item in attack_records] if item is not None]
	print("backward_ones", len(backward_ones))
	both_ones = [item for item in [add_backdoor(item, "backward") for item in forward_ones] if item is not None]
	print("both_ones", len(both_ones))
	pd.DataFrame.from_dict(attack_records).to_csv("attack.csv", index=False)
	pd.DataFrame.from_dict(forward_ones).to_csv("forward_backdoor.csv", index=False)
	pd.DataFrame.from_dict(backward_ones).to_csv("backward_backdoor.csv", index=False)
	pd.DataFrame.from_dict(both_ones).to_csv("both_backdoor.csv", index=False)
	backdoored_records = forward_ones + backward_ones + both_ones
	# print("backdoored_records", len(backdoored_records))
	backdoored_records = pd.DataFrame.from_dict(backdoored_records)
	# backdoored_records.to_csv("exported_df.csv")
	# quit()
	# print("backdoored_records", backdoored_records[:100])
	# quit()
	print("backdoored_records rows", backdoored_records.shape[0])

	df = pd.concat([df, backdoored_records], axis=0, ignore_index=True, sort=False)
	# print("backdoored_records", backdoored_records)
	attack_vector = np.concatenate((attack_vector, np.array(list(backdoored_records['Attack']))))

del df['Attack']
features = df.columns[:-1]
print("Final rows", df.shape)
# df[:1000].to_csv("exported_2.csv")

shuffle_indices = np.array(list(range(df.shape[0])))
random.shuffle(shuffle_indices)

data = df.values
print("data.shape", data.shape)
data = data[shuffle_indices,:]
print("attack_vector.shape", attack_vector.shape)
attack_vector = attack_vector[shuffle_indices]
columns = list(df)
print("columns", columns)

x, y = data[:,:-1].astype(np.float32), data[:,-1:].astype(np.uint8)
if opt.normalizationData == "":
	file_name = opt.dataroot[:-4]+"_"+("backdoor" if opt.backdoor else "normal")+"_normalization_data.pickle"
	means = np.mean(x, axis=0)
	stds = np.std(x, axis=0)
	stds[stds==0.0] = 1.0
	# np.set_printoptions(suppress=True)
	# stds[np.isclose(stds, 0)] = 1.0
	with open(file_name, "wb") as f:
		f.write(pickle.dumps((means, stds)))
else:
	file_name = opt.normalizationData
	with open(file_name, "rb") as f:
		means, stds = pickle.loads(f.read())
assert means.shape[0] == x.shape[1], "means.shape: {}, x.shape: {}".format(means.shape, x.shape)
assert stds.shape[0] == x.shape[1], "stds.shape: {}, x.shape: {}".format(stds.shape, x.shape)
assert not (stds==0).any(), "stds: {}".format(stds)
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
	os.makedirs('surrogate', exist_ok=True)
	train_indices, test_indices = get_nth_split(dataset, opt.nFold, opt.fold)

	logreg = LogisticRegression(solver='liblinear')
	logreg.fit(x[train_indices,:], predict_fun(train_indices))

	predictions = logreg.predict(x[test_indices,:])
	y_true = predict_fun(test_indices)

	print ("Logistic Regression trained with predictions")
	print ("-" * 10)
	output_scores(y_true, predictions)

	print ("Coefficients:", logreg.coef_)
	pd.Series(logreg.coef_[0], features).to_frame().to_csv('surrogate/logreg_pred%s.csv' % suffix)


	logreg = LogisticRegression(solver='liblinear')
	logreg.fit(x[train_indices,:], y[train_indices,0])

	predictions = logreg.predict(x[test_indices,:])
	y_true = y[test_indices,0]

	print ("Logistic Regression trained with real labels")
	print ("-" * 10)
	output_scores(y_true, predictions)

	print ("Coefficients:", logreg.coef_)
	pd.Series(logreg.coef_[0], features).to_frame().to_csv('surrogate/logreg_real%s.csv' % suffix)

def closest(prediction_function):
	n_fold = opt.nFold
	fold = opt.fold

	_, test_indices = get_nth_split(dataset, n_fold, fold)
	data, labels = list(zip(*list(torch.utils.data.Subset(dataset, test_indices))))
	data, labels = torch.stack(data).squeeze().numpy(), torch.stack(labels).squeeze().numpy()
	attacks = attack_vector[test_indices]

	attacks_list = list(attacks)
	print("occurrence of attacks", [(item, attacks_list.count(item)) for item in sorted(list(set(attacks_list)))])

	all_predictions = np.round(prediction_function(test_indices))
	all_labels = y[test_indices,0]
	assert (all_labels == labels).all()

	misclassified_filter = labels != all_predictions
	# print("data", data, "labels", labels, "all_predictions", all_predictions)
	misclassified, misclassified_labels, misclassified_predictions, misclassified_attacks = data[misclassified_filter], labels[misclassified_filter], all_predictions[misclassified_filter], attacks[misclassified_filter]

	# print("misclassified_attacks", list(misclassified_attacks))
	# misclassified = misclassified[:100]
	closest_module.closest(data, labels, attacks, all_predictions, misclassified, misclassified_labels, misclassified_attacks, misclassified_predictions, means, stds, suffix=suffix)

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
		labels = labels.to(device)

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

	eval_nn(test_indices)

def eval_nn(test_indices=None):
	if test_indices is None:
		test_indices = list(range(len(dataset)))

	all_predictions = predict(test_indices)
	all_labels = y[test_indices,0]
	output_scores(all_labels, all_predictions)

def closest_nn():
	closest(predict)

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
	pickle.dump(rf, open('%s.rfmodel' % get_logdir(opt.fold, opt.nFold), 'wb'))

def test_rf():
	_, test_indices = get_nth_split(dataset, opt.nFold, opt.fold)

	predictions = rf.predict (x[test_indices,:])

	output_scores(y[test_indices,0], predictions)

def pdp_rf():
	pdp_module.pdp(x, rf.predict_proba, features, means=means, stds=stds, resolution=1000, n_data=1000, suffix=suffix)

def ale_rf():
	ale_module.ale(x, rf.predict_proba, features, means=means, stds=stds, resolution=1000, n_data=1000, lookaround=10, suffix=suffix)

def ice_rf():
	ice_module.ice(x, rf.predict_proba, features, means=means, stds=stds, resolution=1000, n_data=100, suffix=suffix)

def surrogate_rf():
	surrogate(lambda indices: rf.predict(x[indices,:]))

def closest_rf():
	closest(lambda x: rf.predict(x)[:,1].squeeze())

def noop_nn():
	pass
noop_rf = noop_nn

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

		if opt.net:
			rf = pickle.load(open(opt.net, 'rb'))
		else:
			rf = RandomForestClassifier(n_estimators=100)
			rf.fit(x[train_indices,:], y[train_indices,0])
			# XXX: The following code is broken! It should use predict_proba instead of predict probably
			predictions = rf.predict_proba(x[train_indices,:])
			# print("predictions", predictions.shape, predictions)
			summed_up = np.sum(predictions, axis=1)
			assert (np.isclose(summed_up, 1)).all(), "summed_up: {}".format(summed_up.tolist())

	globals()['%s_%s' % (opt.function, opt.method)]()


