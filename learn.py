#!/usr/bin/env python3

import numpy as np
import pandas as pd
import sys
import math
import random
import torch
from torch.utils.data import DataLoader, Dataset
from tensorboardX import SummaryWriter
import socket
from datetime import datetime
import argparse
import os
import pickle
import gzip
import copy
import itertools

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,recall_score, precision_score, f1_score, balanced_accuracy_score, confusion_matrix, classification_report

import collections
import pickle
import ast
import warnings

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import scipy.stats
import io

from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm
from matplotlib import cm

from sklearn.model_selection import StratifiedKFold

import statistics

# TODOs: unify the binary code
#		 add attack mapping json for better plotting of long names

def output_scores(y_true, y_pred, only_accuracy=False, average='binary'):
	metrics = [ accuracy_score(y_true, y_pred) ]
	if not only_accuracy:
		metrics.extend([
			precision_score(y_true, y_pred, average=average),
			recall_score(y_true, y_pred, average=average),
			f1_score(y_true, y_pred, average=average),
			balanced_accuracy_score(y_true, y_pred, adjusted=True)
		])
	names = ['Accuracy', 'Precision', 'Recall', 'F1', 'Youden'] if not only_accuracy else ["Accuracy"]
	print (('{:>11}'*len(names)).format(*names))
	print ((' {:.8f}'*len(metrics)).format(*metrics))
	return { name: metric for name, metric in zip(names, metrics) }

class OurDataset(Dataset):
	def __init__(self, data, labels=None, attack_vector=None, multiclass=None):
		self.data = data
		assert not np.isnan(self.data).any(), "data is nan: {}".format(self.data)

		if multiclass:
			self.labels = attack_vector
			assert(self.data.shape[0] == self.labels.shape[0])
			lb_style = LabelBinarizer()
			self.labels = lb_style.fit_transform(self.labels)
			self.attacks = lb_style.classes_

		else:
			self.labels = labels
			assert not np.isnan(labels).any(), "labels is nan: {}".format(labels)
			assert(self.data.shape[0] == self.labels.shape[0])

	def __getitem__(self, index):
		data, labels = torch.FloatTensor(self.data[index,:]), torch.FloatTensor(self.labels[index,:])
		return data, labels

	def __len__(self):
		return self.data.shape[0]

def get_nth_split(dataset, n_fold, index, stratify=True):
	if stratify:
		train_indices, test_indices = next(itertools.islice(StratifiedKFold(n_splits = n_fold).split(np.empty(len(attack_vector)), y = attack_vector), index, None))
	else:
		# old way, random
		dataset_size = len(dataset)
		indices = list(range(dataset_size))
		bottom, top = int(math.floor(float(dataset_size)*index/n_fold)), int(math.floor(float(dataset_size)*(index+1)/n_fold))
		train_indices, test_indices = indices[0:bottom]+indices[top:], indices[bottom:top]
	return train_indices[:opt.maxSize], test_indices[:opt.maxSize]

def make_net(n_input, n_output, n_layers, layer_size):
	layers = []
	layers.append(torch.nn.Linear(n_input, layer_size))
	layers.append(torch.nn.ReLU())
	layers.append(torch.nn.Dropout(p=opt.dropoutProbability))
	for i in range(n_layers):
		layers.append(torch.nn.Linear(layer_size, layer_size))
		layers.append(torch.nn.ReLU())
		layers.append(torch.nn.Dropout(p=opt.dropoutProbability))
	layers.append(torch.nn.Linear(layer_size, n_output))

	return torch.nn.Sequential(*layers)

def get_logdir(fold, n_fold):
	return os.path.join('runs', current_time + '_' + socket.gethostname() + "_" + str(fold) +"_"+str(n_fold))

# Deep Learning
############################

class EagerNet(torch.nn.Module):
	def __init__(self, n_input, n_output, n_layers, layer_size):
		super(EagerNet, self).__init__()
		self.n_output = n_output
		self.n_layers = n_layers
		self.beginning = torch.nn.Linear(n_input, layer_size+n_output).to(device)
		self.middle = torch.nn.Sequential(*[torch.nn.Linear(layer_size, layer_size+n_output).to(device) for _ in range(n_layers)])
		self.end = torch.nn.Linear(layer_size, n_output).to(device)

	def forward(self, x):
		all_outputs = []
		all_xs = []
		output_beginning = self.beginning(x)
		all_xs.append(x)
		x = torch.nn.functional.leaky_relu(output_beginning[:,:-self.n_output])
		all_outputs.append(output_beginning[:,-self.n_output:])

		for current_layer in self.middle:
			current_output = current_layer(x)
			all_xs.append(x)
			x = torch.nn.functional.leaky_relu(current_output[:,:-self.n_output])
			all_outputs.append(current_output[:,-self.n_output:])

		all_xs.append(x)
		output_end = self.end(x)
		all_outputs.append(output_end)

		return all_outputs, all_xs

def eager_equal_weights(losses):
	n = len(losses)
	raw_weights = [1 for _ in range(n)]
	total_weight_sum = sum(raw_weights)
	return [item/total_weight_sum for item in raw_weights]

def eager_all_one_weights(losses):
	n = len(losses)
	return [1 for _ in range(n)]

def eager_linearly_increasing_weights(losses):
	n = len(losses)
	raw_weights = [i+1 for i in range(n)]
	total_weight_sum = sum(raw_weights)
	return [item/total_weight_sum for item in raw_weights]

def eager_linearly_decreasing_weights(losses):
	n = len(losses)
	raw_weights = list(reversed([i+1 for i in range(n)]))
	total_weight_sum = sum(raw_weights)
	return [item/total_weight_sum for item in raw_weights]

def eager_doubling_weights(losses):
	n = len(losses)
	raw_weights = [2**i for i in range(n)]
	total_weight_sum = sum(raw_weights)
	return [item/total_weight_sum for item in raw_weights]

def eager_reverse_doubling_weights(losses):
	n = len(losses)
	raw_weights = [1/(2*(i+1)) for i in range(n)]
	total_weight_sum = sum(raw_weights)
	return [item/total_weight_sum for item in raw_weights][::-1]

def eager_only_last_weights(losses):
	""" FFNN """
	n = len(losses)
	raw_weights = [0 for i in range(n)]
	raw_weights[-1] = 1
	return raw_weights

def eager_only_min_weights(losses):
	n = len(losses)
	which = losses.index(min(losses))
	raw_weights = [0 for i in range(n)]
	raw_weights[which] = 1
	return raw_weights

def eager_min_decreasing_weights(losses):
	n = len(losses)
	which = losses.index(min(losses))
	raw_weights = [0 for i in range(n)]
	#raw_weights[which] = 1
	raw_weights[which:] = list(reversed([2**i for i in range(which, n)]))
	total_weight_sum = sum(raw_weights)
	return [item/total_weight_sum for item in raw_weights]

def train_eager_stopping_nn():
	""" Unified training of binary and multiclass """
	n_fold = opt.nFold
	fold = opt.fold

	train_indices, _ = get_nth_split(dataset, n_fold, fold)
	train_data = torch.utils.data.Subset(dataset, train_indices)
	train_loader = torch.utils.data.DataLoader(train_data, batch_size=opt.batchSize, shuffle=True)

	writer = SummaryWriter(get_logdir(fold, n_fold))
	_ = writer.log_dir

	criterion = torch.nn.CrossEntropyLoss() if opt.multiclass else torch.nn.BCEWithLogitsLoss(reduction="mean")
	n_classes = dataset.labels.shape[-1]
	net = EagerNet(x.shape[-1], n_classes, opt.nLayers, opt.layerSize).to(device)
	optimizer = getattr(torch.optim, opt.optimizer)(net.parameters(), lr=opt.lr)

	samples = 0
	net.train()

	print('>> Start training...')
	for i in range(1, sys.maxsize):
		print(f">> Epoch {i}")
		for data, labels in train_loader:
			optimizer.zero_grad()
			data = data.to(device)
			samples += data.shape[0]
			labels = labels.to(device) if not opt.multiclass else torch.tensor(labels.clone().detach(), dtype=torch.long, device=device).to(device)
			if not opt.multiclass:
				not_attack_mask = labels.squeeze() == 0

			outputs, xs = net(data)
			losses = []
			all_weights = [net.beginning, *net.middle, net.end]
			assert(len(xs) == len(all_weights))
			for output_index, output in enumerate(outputs):
				if opt.eagerStoppingWeightingMethod=="eager_all_one_weights" and output_index < len(outputs)-1:
					new_output = all_weights[output_index](xs[output_index].detach())[:,-net.n_output:]
					loss = criterion(new_output, labels)
				elif opt.multiclass:
					_, _class = torch.max(labels, 1)
					loss = criterion(output, _class)
				else:
					loss = criterion(output, labels)
				losses.append(loss)

				if not opt.multiclass:
					sigmoided_output = torch.sigmoid(output.detach()).squeeze()
					accuracy = torch.mean((torch.round(sigmoided_output) == labels.squeeze()).float())
					writer.add_scalar(f"accuracy_{output_index}", accuracy, samples)

					confidences = sigmoided_output.detach().clone()
					confidences[not_attack_mask] = 1 - confidences[not_attack_mask]
					writer.add_scalar(f"confidence_{output_index}", torch.mean(confidences), samples)
					if opt.saveHistogram:
						writer.add_histogram(f"confidence_hist_{output_index}", confidences, samples)
				else:
					_, argmax = torch.max(output, 1)
					accuracy = (_class == argmax.squeeze()).float().mean()

					writer.add_scalar(f"accuracy_{output_index}", accuracy, samples)

			total_loss = None
			eager_stopping_weight_per_output_per_layer = globals()[opt.eagerStoppingWeightingMethod](losses)

			for loss_index, loss in enumerate(losses):
				loss = eager_stopping_weight_per_output_per_layer[loss_index]*loss
				if total_loss is None:
					total_loss = loss
				else:
					total_loss += loss

			total_loss.backward()
			optimizer.step()

			writer.add_scalar("loss", total_loss.item(), samples)

		torch.save(net.state_dict(), '%s/net_%d.pth' % (writer.log_dir, samples))

def create_plot_eager_nn():
	_, test_indices = get_nth_split(dataset, opt.nFold, opt.fold)
	if opt.multiclass:
		multiclass_eager(test_indices)
	else:
		create_binary_plot_eager(test_indices)

def predict_eager_nn():
	_, test_indices = get_nth_split(dataset, opt.nFold, opt.fold)

	if opt.multiclass:
		multiclass_eager(test_indices, evaluate=True)
	else:
		create_binary_prediction_eager(test_indices)

def create_binary_plot_eager_(test_indices):
	""" Create confidence-accuracy plot for binary """

	test_data = torch.utils.data.Subset(dataset, test_indices)
	test_loader = torch.utils.data.DataLoader(test_data, batch_size=opt.batchSize, shuffle=False)

	samples = 0
	all_predictions = []

	net.eval()
	for data, labels in test_loader:
		data = data.to(device)
		samples += data.shape[0]
		labels = labels.to(device)

		outputs, _ = net(data)

		thing_to_append = torch.sigmoid(torch.stack([output.detach() for output in outputs])).squeeze(2).transpose(1,0).cpu().numpy()
		thing_to_append = np.maximum(thing_to_append, 1-thing_to_append)
		all_predictions.append(thing_to_append)

	all_predictions = np.concatenate(all_predictions, axis=0).astype(np.float32).tolist()

	n_outputs = len(all_predictions[0])
	lists = [list() for _ in range(n_outputs)]

	already_processed = 0
	efforts = []
	efforts_std = []
	lists[0] = list(all_predictions)

	accuracy_per_list = [sum([item[output_index] for item in l]) for output_index, l in enumerate(lists)]

	print("len(all_predictions)", len(all_predictions), "accuracy_per_list", accuracy_per_list)

	while np.array([len(l)>0 for l in lists[:n_outputs-1]]).any():
		print("items_per_list", [len(item) for item in lists])
		current_lowest = float("inf")
		candidate = None
		candidate_output_index = None
		for output_index in range(n_outputs-1):
			for item_index, item in enumerate(lists[output_index]):
				current_confidence = lists[output_index][item_index][output_index]
				if current_confidence < current_lowest:
					candidate = item
					candidate_output_index = output_index
					current_lowest = current_confidence

		lists[candidate_output_index].remove(candidate)
		lists[candidate_output_index+1].append(candidate)

		accuracy_per_list[candidate_output_index] -= candidate[candidate_output_index]
		accuracy_per_list[candidate_output_index+1] += candidate[candidate_output_index+1]

		# print([len(lists[output_index]) for output_index in range(n_outputs)], len(all_predictions))

		if len(efforts)==0 or current_lowest >= efforts[-1][0]:
			efforts.append((current_lowest, sum([(output_index+1)*len(lists[output_index]) for output_index in range(n_outputs)])/len(all_predictions), sum(accuracy_per_list)/len(all_predictions)))
			efforts_std.append((current_lowest, statistics.stdev([(output_index+1)*len(lists[output_index]) for output_index in range(n_outputs)])/5, statistics.stdev(accuracy_per_list)))

	colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

	plt.grid(linestyle='--', linewidth='0.5', color='gray', axis='x')
	things = []
	x, y1, y2 = list(zip(*efforts))
	_, y1err, y2err = list(zip(*efforts))
	y1_label = "Mean layers needed"
	things += plt.errorbar(x, y1, yerr=y1err, color=colors[0], label=y1_label, errorevery=100)
	plt.xlabel("Confidence required to stop further evaluation", size=15)
	plt.ylabel(y1_label, size=15)
	plt.twinx()
	y2_label = "Mean accuracy achieved"
	things += plt.errorbar(x, y2, color=colors[1], label=y2_label)
	plt.ylabel(y2_label, size=15)
	plt.tight_layout()
	#plt.legend(things, [l.get_label() for l in things], loc="upper left")
	plt.rc('axes', axisbelow=True)

	if opt.savePlot != '':
		plt.savefig(opt.savePlot, bbox_inches = 'tight', pad_inches = 0)
	else:
		plt.show()

	if opt.saveResults:
		with open(f"efforts_{opt.net}", "wb") as fp:
			pickle.dump(efforts, fp)

	# print("all_predictions", all_predictions[:10])

def create_binary_plot_eager(test_indices):
	""" Create confidence-accuracy plot for binary """

	test_data = torch.utils.data.Subset(dataset, test_indices)
	test_loader = torch.utils.data.DataLoader(test_data, batch_size=opt.batchSize, shuffle=False)

	samples = 0
	all_predictions = []

	net.eval()
	for data, labels in test_loader:
		data = data.to(device)
		samples += data.shape[0]
		labels = labels.to(device)

		outputs, _ = net(data)

		thing_to_append = torch.sigmoid(torch.stack([output.detach() for output in outputs])).squeeze(2).transpose(1,0).cpu().numpy()
		thing_to_append = np.maximum(thing_to_append, 1-thing_to_append)
		all_predictions.append(thing_to_append)

	all_predictions = np.concatenate(all_predictions, axis=0).astype(np.float32).tolist()

	n_outputs = len(all_predictions[0])
	lists = [list() for _ in range(n_outputs)]

	already_processed = 0
	efforts = []
	efforts_std = []
	lists[0] = list(all_predictions)

	accuracy_per_list = [sum([item[output_index] for item in l]) for output_index, l in enumerate(lists)]

	print("len(all_predictions)", len(all_predictions), "accuracy_per_list", accuracy_per_list)

	while np.array([len(l)>0 for l in lists[:n_outputs-1]]).any():
		print("items_per_list", [len(item) for item in lists])
		current_lowest = float("inf")
		candidate = None
		candidate_output_index = None
		for output_index in range(n_outputs-1):
			for item_index, item in enumerate(lists[output_index]):
				current_confidence = lists[output_index][item_index][output_index]
				if current_confidence < current_lowest:
					candidate = item
					candidate_output_index = output_index
					current_lowest = current_confidence

		lists[candidate_output_index].remove(candidate)
		lists[candidate_output_index+1].append(candidate)

		accuracy_per_list[candidate_output_index] -= candidate[candidate_output_index]
		accuracy_per_list[candidate_output_index+1] += candidate[candidate_output_index+1]

		# print([len(lists[output_index]) for output_index in range(n_outputs)], len(all_predictions))

		if len(efforts)==0 or current_lowest >= efforts[-1][0]:
			efforts.append((current_lowest, sum([(output_index+1)*len(lists[output_index]) for output_index in range(n_outputs)])/len(all_predictions), sum(accuracy_per_list)/len(all_predictions)))
			#efforts_std.append((current_lowest, statistics.stdev([(output_index+1)*len(lists[output_index]) for output_index in range(n_outputs)])/5, statistics.stdev(accuracy_per_list)))

	with open("efforts", "wb") as fp:
		pickle.dump(efforts, fp)

	# print("all_predictions", all_predictions[:10])

def create_binary_prediction_eager(test_indices):
	""" Get binary performance on test-set"""

	test_data = torch.utils.data.Subset(dataset, test_indices)
	test_loader = torch.utils.data.DataLoader(test_data, batch_size=opt.batchSize, shuffle=False)

	samples = 0
	n_outputs = opt.nLayers+2

	y_pred_list = [[] for _ in range(n_outputs)]
	y_list = []

	net.eval()
	with torch.no_grad():
		for data, labels in tqdm(test_loader):
			data = data.to(device)
			samples += data.shape[0]
			labels = labels.to(device)
			outputs, _ = net(data)
			for output_index, output in enumerate(outputs):
				y_pred_list[output_index].append(torch.round(torch.sigmoid(output.detach()).squeeze()).cpu().numpy())

			y_list.append(labels.cpu().numpy())
		y_list = [a.squeeze().tolist() for a in y_list]
		y_list = [item for sublist in y_list for item in sublist]
		scores = {}
		for i, output in enumerate(y_pred_list):
			y_pred_list[i] = [a.squeeze().tolist() for a in output]
			y_pred_list[i] = [item for sublist in y_pred_list[i] for item in sublist]
			scores['Layer_{}'.format(i)] = output_scores(y_list, y_pred_list[i])

def multiclass_eager(test_indices, evaluate=False):
	""" Get test-set performance if evaluate is True
	else create accuracy per layer and class plot """

	opt.batchSize = 1 # FIXME: this is a workaround

	test_data = torch.utils.data.Subset(dataset, test_indices)
	test_loader = torch.utils.data.DataLoader(test_data, batch_size=opt.batchSize, shuffle=False)

	n_outputs = opt.nLayers + 2

	y_pred_list = [[] for _ in range(n_outputs)]
	y_list = []
	with torch.no_grad():
		net.eval()
		for data, labels in tqdm(test_loader):
			X_batch = data.to(device)
			outputs, _ = net(X_batch)
			for output_index, y_test_pred in enumerate(outputs):
				y_pred_softmax = torch.log_softmax(y_test_pred, dim = 1)
				_, y_pred_tags = torch.max(y_pred_softmax, dim = 1)
				y_pred_list[output_index].append(y_pred_tags.cpu().numpy())

			_, labels = torch.max(labels, dim = 1)
			y_list.append(labels.cpu().numpy())

	y_list = [a.squeeze().tolist() for a in y_list]

	if opt.savePlot:
		accuracies = np.empty((n_classes, n_outputs))
		np.seterr(divide='ignore', invalid='ignore')
		
		scores = {}
		for i, output in enumerate(y_pred_list):
			y_pred_list[i] = [a.squeeze().tolist() for a in output]

			cr = confusion_matrix(y_list, y_pred_list[i], list(range(n_classes)))
			accuracies[:,i] = cr.diagonal()/cr.sum(axis=1)

		if opt.saveResults:
			with open(f"{opt.net}_labels", "wb") as fp:
				pickle.dump(y_list, fp)
			with open(f"{opt.net}_predictions", "wb") as fp:
				pickle.dump(y_pred_list, fp)

		accuracies = np.nan_to_num(accuracies)
		order = np.sum(accuracies, axis = 1).argsort()
		accuracies = np.take(accuracies, order, axis=0) # sort

		if opt.saveResults:
			np.save('accuracies_' + opt.net, accuracies)

		plt.figure(figsize=(16,8))
		plt.imshow(accuracies, cmap=cm.coolwarm, interpolation='nearest')
		clb = plt.colorbar()
		clb.set_label('Accuracy', rotation=90, fontsize=30)
		plt.yticks(list(range(n_classes)), labels=[dataset.attacks[i] for i in order], fontsize=20)
		plt.xlabel('Layers', fontsize=30)
		plt.ylabel('Attack Families', fontsize=30)
		plt.xticks(list(range(n_outputs)), list(range(1,n_outputs+1)), fontsize=20)
		plt.savefig(opt.savePlot, bbox_inches='tight')

	if evaluate:
		for i, output in enumerate(y_pred_list):
			output_scores(y_list, y_pred_list[i], average='micro')

if __name__=="__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--dataroot', required=True, help='path to dataset')
	parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
	parser.add_argument('--nLayers', type=int, default=3)
	parser.add_argument('--layerSize', type=int, default=512)
	parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for')
	parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
	parser.add_argument('--dropoutProbability', type=float, default=0.2, help='probability for each neuron to be withheld in an iteration')
	parser.add_argument('--fold', type=int, default=0, help='fold to use')
	parser.add_argument('--nFold', type=int, default=3, help='total number of folds')
	parser.add_argument('--net', default='', help="path to net (to continue training)")
	parser.add_argument('--function', default='train', help='the function that is going to be called')
	parser.add_argument('--arg', default='', help="optional arguments")
	parser.add_argument('--manualSeed', default=0, type=int, help='manual seed')
	parser.add_argument('--normalizationData', default="", type=str, help='normalization data to use')
	parser.add_argument('--method', choices=['nn'])
	parser.add_argument('--maxRows', default=sys.maxsize, type=int, help='number of rows from the dataset to load (for debugging mainly)')
	parser.add_argument('--maxSize', default=sys.maxsize, type=int, help='only use up to maxSize data samples')
	parser.add_argument('--eagerStoppingWeightingMethod', default="eager_equal_weights", type=str, help="how to weight each layer's output when training for eager stopping.")
	parser.add_argument('--saveHistogram', action='store_true', help='whether a histogram of confidences is saved for each ')
	parser.add_argument('--optimizer', default="Adam", type=str)
	parser.add_argument('--multiclass', action='store_true')
	parser.add_argument('--savePlot', default='plot.pdf', type=str, help='the name of the figure to store after evaluation')
	parser.add_argument('--saveResults', action='store_true', help='save output results of the main calculation in a numpy npy')


	opt = parser.parse_args()
	print('#'*40)
	print('>> Configuration ' + str(opt))
	print('#'*40)

	seed = opt.manualSeed
	# if seed is None:
	# 	seed = random.randrange(1000)
	# 	print("No seed was specified, thus choosing one randomly:", seed)
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)

	suffix = '_%s_%d' % (opt.method, opt.fold)
	dirsuffix = '_%s' % opt.dataroot[:-4]

	csv_name = opt.dataroot
	df = pd.read_csv(csv_name, nrows=opt.maxRows).fillna(0)
	df = df[df['flowDurationMilliseconds'] < 1000 * 60 * 60 * 24 * 10]

	del df['flowStartMilliseconds']
	del df['sourceIPAddress']
	del df['destinationIPAddress']
	attack_vector = np.array(list(df['Attack']))
	assert len(attack_vector.shape) == 1

	del df['Attack']
	features = df.columns[:-1]
	#print("Final rows", df.shape)

	shuffle_indices = np.array(list(range(df.shape[0])))
	random.shuffle(shuffle_indices)

	data = df.values
	print('#'*40)
	print(">> Data shape ", data.shape)
	print('#'*40)
	data = data[shuffle_indices,:]
	#print("attack_vector.shape", attack_vector.shape)
	attack_vector = attack_vector[shuffle_indices]
	assert len(attack_vector) == len(data)
	columns = list(df)
	print('#'*40)
	print(">> Features ", columns)
	print('#'*40)

	x, y = data[:,:-1].astype(np.float32), data[:,-1:].astype(np.uint8)
	file_name = opt.dataroot[:-4]+"_normal"
	if opt.normalizationData == "":
		file_name_for_normalization_data = file_name+"_normalization_data.pickle"
		means = np.mean(x, axis=0)
		stds = np.std(x, axis=0)
		stds[stds==0.0] = 1.0
		# np.set_printoptions(suppress=True)
		# stds[np.isclose(stds, 0)] = 1.0
		#with open(file_name_for_normalization_data, "wb") as f:
		#	f.write(pickle.dumps((means, stds)))
	else:
		file_name_for_normalization_data = opt.normalizationData
		#with open(file_name_for_normalization_data, "rb") as f:
		#	means, stds = pickle.loads(f.read())
	assert means.shape[0] == x.shape[1], "means.shape: {}, x.shape: {}".format(means.shape, x.shape)
	assert stds.shape[0] == x.shape[1], "stds.shape: {}, x.shape: {}".format(stds.shape, x.shape)
	assert not (stds==0).any(), "stds: {}".format(stds)
	x = (x-means)/stds

	dataset = OurDataset(x, y, np.expand_dims(attack_vector, axis=1), multiclass=True if opt.multiclass else False)
	n_classes = dataset.labels.shape[-1]

	current_time = datetime.now().strftime('%b%d_%H-%M-%S')

	cuda_available = torch.cuda.is_available()
	device = torch.device("cuda:0" if cuda_available else "cpu")

	net = EagerNet(x.shape[-1], n_classes, opt.nLayers, opt.layerSize).to(device)
	# print("net", net)

	if opt.net != '':
		print("Loading", opt.net)
		net.load_state_dict(torch.load(opt.net, map_location=device))

	globals()['%s_%s' % (opt.function, opt.method)]()


