#!/usr/bin/env python3

import sys
import os
import pickle
import math
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tensorboardX import SummaryWriter

HIDDEN_SIZE = 512
N_LAYERS = 3

class OurDataset(Dataset):
	def __init__(self, data, labels):
		# assert not np.isnan(data).any(), "datum is nan: {}".format(data)
		# assert not np.isnan(labels).any(), "labels is nan: {}".format(labels)
		self.data = data
		self.labels = labels
		assert(len(self.data) == len(self.labels))

	def __getitem__(self, index):
		data, labels = torch.FloatTensor(self.data[index]), torch.FloatTensor(self.labels[index])
		return data, labels

	def __len__(self):
		return len(self.data)

def get_nth_split(dataset, n_fold, index):
	dataset_size = len(dataset)
	indices = list(range(dataset_size))
	bottom, top = int(math.floor(float(dataset_size)*index/n_fold)), int(math.floor(float(dataset_size)*(index+1)/n_fold))
	train_indices, test_indices = indices[0:bottom]+indices[top:], indices[bottom:top]
	return train_indices, test_indices

class OurLSTMModule(nn.Module):

	def __init__(self, num_inputs, num_outputs, hidden_size, n_layers, batch_size, device):
		super(OurLSTMModule, self).__init__()
		self.hidden_size = hidden_size
		self.n_layers = n_layers
		self.num_inputs = num_inputs
		self.num_outputs = num_outputs
		self.batch_size = batch_size
		self.device = device
		self.lstm = nn.LSTM(input_size=num_inputs, hidden_size=hidden_size, num_layers=n_layers)
		self.hidden = None
		# self.i2h = nn.Linear(num_inputs, hidden_size)
		self.h2o = nn.Linear(hidden_size, num_outputs)
		# self.softmax = nn.Softmax(dim=2)

	# batch has seq * batch * input_dim

	def init_hidden(self, probability=0.0):

		if self.hidden is not None:
			probs_tensor = torch.zeros(self.n_layers, self.batch_size, self.hidden_size)
			probs_tensor.fill_(probability)
			drawn1 = torch.bernoulli(probs_tensor)
			drawn2 = torch.bernoulli(probs_tensor)

			self.hidden = ((drawn1.to(device)*self.hidden[0]).detach().to(device),
			(drawn2.to(device)*self.hidden[1]).detach().to(device))

		else:
			self.hidden = (torch.zeros(self.n_layers, self.batch_size, self.hidden_size).to(self.device),
			torch.zeros(self.n_layers, self.batch_size, self.hidden_size).to(self.device))

	def forward(self, batch):

		# preprocessed_batch = self.i2h(batch.view(-1,batch.shape[-1])).view(batch.shape[0], batch.shape[1], self.hidden_size)
		lstm_out, self.hidden = self.lstm(batch, self.hidden)
		lstm_out, seq_lens = torch.nn.utils.rnn.pad_packed_sequence(lstm_out)
		# output = self.h2o(lstm_out.view(-1, self.hidden_size)).view(*lstm_out.shape[:2], self.num_outputs)
		output = self.h2o(lstm_out)
		# output = self.softmax(output)
		return output, seq_lens

def get_one_hot_vector(class_indices, num_classes, batch_size):
	y_onehot = torch.FloatTensor(batch_size, num_classes)
	y_onehot.zero_()
	return y_onehot.scatter_(1, class_indices.unsqueeze(1), 1)

def custom_collate(seqs):
	# get the length of each seq in your batch
	seqs, labels = zip(*seqs)
	assert len(seqs) == len(labels)
	# print("seqs", seqs)
	seq_lengths = torch.LongTensor([len(seq) for seq in seqs])

	# TODO: Use built-in padding function?
	# dump padding everywhere, and place seqs on the left.
	# NOTE: you only need a tensor as big as your longest sequence
	# seq_tensor = torch.zeros((len(seqs), seq_lengths.max(), seqs[0].shape[-1]))
	# for idx, (seq, seqlen) in enumerate(zip(seqs, seq_lengths)):
	# 	# print("seq", seq.shape, "seqlen", seqlen)
	# 	assert seqlen != 0, "seqlen: {}".format(seqlen)
	# 	seq_tensor[idx, :seqlen, :] = torch.FloatTensor(seq).squeeze()

	# labels_tensor = torch.zeros((len(labels), seq_lengths.max(), labels[0].shape[-1]))
	# for idx, (label, seqlen) in enumerate(zip(labels, seq_lengths)):
	# 	# print("seq", seq.shape, "seqlen", seqlen)
	# 	assert seqlen != 0, "seqlen: {}".format(seqlen)
	# 	labels_tensor[idx, :seqlen, :] = torch.FloatTensor(label)

	seq_tensor = torch.nn.utils.rnn.pad_sequence(seqs)
	labels_tensor = torch.nn.utils.rnn.pad_sequence(labels)

	# # SORT YOUR TENSORS BY LENGTH!
	# seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
	# seq_tensor = seq_tensor[perm_idx]
	# labels_tensor = labels_tensor[perm_idx]

	# utils.rnn lets you give (B,L,D) tensors where B is the batch size, L is the maxlength, if you use batch_first=True
	# Otherwise, give (L,B,D) tensors
	# seq_tensor = seq_tensor.transpose(0,1) # (B,L,D) -> (L,B,D)
	# labels_tensor = labels_tensor.transpose(0,1) # (B,L,D) -> (L,B,D)

	# pack them up nicely
	# packed_input = pack_padded_sequence(seq_tensor, seq_lengths.cpu().numpy())
	# print("seq_tensor.shape", seq_tensor.shape, "seq_lengths.shape", seq_lengths.shape)
	packed_input = torch.nn.utils.rnn.pack_padded_sequence(seq_tensor, seq_lengths, enforce_sorted=False)
	packed_labels = torch.nn.utils.rnn.pack_padded_sequence(labels_tensor, seq_lengths, enforce_sorted=False)
	return packed_input, packed_labels

def train():

	n_fold = opt.nFold
	fold = opt.fold
	lstm_module.train()

	train_indices, _ = get_nth_split(dataset, n_fold, fold)
	train_data = torch.utils.data.Subset(dataset, train_indices)
	train_loader = torch.utils.data.DataLoader(train_data, batch_size=opt.batchSize, shuffle=True, collate_fn=custom_collate)

	optimizer = optim.SGD(lstm_module.parameters(), lr=opt.lr)
	# criterion = nn.CrossEntropyLoss(reduction="mean")
	criterion = nn.BCEWithLogitsLoss(reduction="mean")

	writer = SummaryWriter()

	samples = 0
	for i in range(1, sys.maxsize):
		for input_data, labels in train_loader:
			print("iterating")
			samples += len(input_data)
			optimizer.zero_grad()
			lstm_module.init_hidden()

			# actual_input = torch.FloatTensor(input_tensor[:,:,:-1]).to(device)

			output, seq_lens = lstm_module(input_data)

			index_tensor = torch.arange(0, output.shape[0], dtype=torch.int64).unsqueeze(1).unsqueeze(2).repeat(1, output.shape[1], output.shape[2])

			selection_tensor = seq_lens.unsqueeze(0).unsqueeze(2).repeat(index_tensor.shape[0], 1, index_tensor.shape[2])

			# print("index_tensor.shape", index_tensor.shape, "selection_tensor.shape", selection_tensor.shape)
			mask = (index_tensor < selection_tensor).byte()

			torch.set_printoptions(profile="full")
			# print("mask", mask.squeeze())

			mask_exact = (index_tensor == selection_tensor).byte()

			# labels = torch.LongTensor(input_tensor[:,:,-1])
			# loss = criterion(output.view(-1, output.shape[-1]), labels.view(-1))
			# print("labels.shape", labels.shape, "actual_input.shape", actual_input.shape)

			labels, _ = torch.nn.utils.rnn.pad_packed_sequence(labels)

			# print("output.shape", output.shape, "labels.shape", labels.shape, "mask.shape", mask.shape)
			loss = criterion(output[mask].view(-1), labels[mask].view(-1))#.view(labels.shape[0], labels.shape[1], lstm_module.num_outputs).to(device))
			labels = labels.to(device)
			loss.backward()

			# Add parameters' gradients to their values, multiplied by learning rate
			optimizer.step()

			writer.add_scalar("loss", loss.item(), i*opt.batchSize)
			accuracy = torch.mean((torch.round(output.detach()[mask]) == labels[mask]).float())
			writer.add_scalar("accuracy", accuracy, i*opt.batchSize)
			end_accuracy = torch.mean((torch.round(output.detach()[mask_exact]) == labels[mask_exact]).float())
			writer.add_scalar("end_accuracy", end_accuracy, i*opt.batchSize)

			# confidence_for_correct_one = torch.mean(torch.gather(torch.sigmoid(output.detach()[mask]), 2, labels[mask]))
			# writer.add_scalar("confidence", confidence_for_correct_one, i*opt.batchSize)
			# end_confidence_for_correct_one = torch.mean(torch.gather(torch.sigmoid(output.detach()[-1,:,:]), 1, labels[-1,:].unsqueeze(1)))
			# writer.add_scalar("end_confidence", end_confidence_for_correct_one, i*opt.batchSize)

			# if i % 1_000_000 == 0:
			# 	# torch.set_printoptions(profile="full")
			# 	print("iteration", i)
			# 	cpuStats()
			# 	memReport()
			# 	print("actual_input")
			# 	print(actual_input.squeeze().transpose(1,0))
			# 	print("output")
			# 	print(output.transpose(1,0))
			# 	print("output_argmax")
			# 	print(torch.argmax(output, 2).transpose(1,0))
			# 	print("labels")
			# 	print(labels.transpose(1,0))
			# 	# torch.set_printoptions(profile="default")
			# 	sys.stdout.flush()
			# 	sys.stderr.flush()

			if i % 100 == 0:
				torch.save(lstm_module.state_dict(), '%s/lstm_module_%d.pth' % (writer.log_dir, i*opt.batchSize))


if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataroot', required=True, help='path to dataset')
	parser.add_argument('--normalizationData', default="", type=str, help='normalization data to use')
	parser.add_argument('--fold', type=int, default=0, help='fold to use')
	parser.add_argument('--nFold', type=int, default=3, help='total number of folds')
	parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
	parser.add_argument('--net', default='', help="path to net (to continue training)")
	parser.add_argument('--function', default='train', help='the function that is going to be called')
	parser.add_argument('--manualSeed', type=int, help='manual seed')
	parser.add_argument('--lr', type=float, default=10**(-2), help='learning rate')

	opt = parser.parse_args()

	with open (opt.dataroot, "rb") as f:
		all_data = pickle.load(f)

	x = [item[:, :-2] for item in all_data]
	y = [item[:, -1:] for item in all_data]

	if opt.normalizationData == "":
		file_name = opt.dataroot[:-7]+"_normalization_data.pickle"
		catted_x = np.concatenate(x, axis=0)
		# print("catted_x.shape", catted_x.shape)
		means = np.mean(catted_x, axis=0)
		stds = np.std(catted_x, axis=0)
		stds[stds==0.0] = 1.0
		# np.set_printoptions(suppress=True)
		# stds[np.isclose(stds, 0)] = 1.0
		with open(file_name, "wb") as f:
			f.write(pickle.dumps((means, stds)))
	else:
		file_name = opt.normalizationData
		with open(file_name, "rb") as f:
			means, stds = pickle.load(f)
	assert means.shape[0] == x[0].shape[-1], "means.shape: {}, x.shape: {}".format(means.shape, x[0].shape)
	assert stds.shape[0] == x[0].shape[-1], "stds.shape: {}, x.shape: {}".format(stds.shape, x[0].shape)
	assert not (stds==0).any(), "stds: {}".format(stds)
	x = [(item-means)/stds for item in x]

	cuda_available = torch.cuda.is_available()
	device = torch.device("cuda:0" if cuda_available else "cpu")

	dataset = OurDataset(x, y)

	lstm_module = OurLSTMModule(x[0].shape[-1], y[0].shape[-1], HIDDEN_SIZE, N_LAYERS, opt.batchSize, device).to(device)

	if opt.net != '':
		print("Loading", opt.net)
		lstm_module.load_state_dict(torch.load(opt.net, map_location=device))

	globals()[opt.function]()