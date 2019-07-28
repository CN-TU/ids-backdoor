#!/usr/bin/env python3

import sys
import os
import pickle
import json
import math
import argparse
import random
import time
from functools import reduce

import collections
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

HIDDEN_SIZE = 512
N_LAYERS = 3

ADVERSARIAL_THRESH = 50

def numpy_sigmoid(x):
	return 1/(1+np.exp(-x))

class OurDataset(Dataset):
	def __init__(self, data, labels, categories):
		# assert not np.isnan(data).any(), "datum is nan: {}".format(data)
		# assert not np.isnan(labels).any(), "labels is nan: {}".format(labels)
		self.data = data
		self.labels = labels
		self.categories = categories
		assert(len(self.data) == len(self.labels) == len(self.categories))

	def __getitem__(self, index):
		data, labels, categories = torch.FloatTensor(self.data[index]), torch.FloatTensor(self.labels[index]), torch.FloatTensor(self.categories[index])
		return data, labels, categories

	def __len__(self):
		return len(self.data)
				
class AdvDataset(Dataset):
	def __init__(self, base_dataset):
		self.base_dataset = base_dataset
		self.adv_flows = []
		self.categories = []
			
	def __getitem__(self, index):
		base_len = len(self.base_dataset)
		if index < base_len:
			return self.base_dataset.__getitem__(index)
		else:
			flow = self.adv_flows[index - base_len]
			category = self.categories[index - base_len] + ADVERSARIAL_THRESH
			data = torch.FloatTensor(flow)
			labels = torch.ones((flow.shape[0], 1))
			categories = torch.ones((flow.shape[0], 1)) * category
			return data, labels, categories

	def __len__(self):
		return len(self.base_dataset) + len(self.adv_flows)

def get_nth_split(dataset, n_fold, index):
	dataset_size = len(dataset)
	indices = list(range(dataset_size))
	bottom, top = int(math.floor(float(dataset_size)*index/n_fold)), int(math.floor(float(dataset_size)*(index+1)/n_fold))
	train_indices, test_indices = indices[0:bottom]+indices[top:], indices[bottom:top]
	return train_indices[:opt.maxSize], test_indices[:opt.maxSize]

class OurLSTMModule(nn.Module):

	def __init__(self, num_inputs, num_outputs, hidden_size, n_layers, batch_size, device, forgetting=False):
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
		self.forgetting = forgetting

	# batch has seq * batch * input_dim

	def init_hidden(self, batch_size):
		self.hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_size).to(self.device),
		torch.zeros(self.n_layers, batch_size, self.hidden_size).to(self.device))

	def forward(self, batch):
		# preprocessed_batch = self.i2h(batch.view(-1,batch.shape[-1])).view(batch.shape[0], batch.shape[1], self.hidden_size)
		# print("batch", batch)
		lstm_out, new_hidden = self.lstm(batch, self.hidden)
		if not self.forgetting:
			self.hidden = new_hidden
		lstm_out, seq_lens = torch.nn.utils.rnn.pad_packed_sequence(lstm_out)
		# output = self.h2o(lstm_out.view(-1, self.hidden_size)).view(*lstm_out.shape[:2], self.num_outputs)
		output = self.h2o(lstm_out)
		# output = self.softmax(output)
		return output, seq_lens

def get_one_hot_vector(class_indices, num_classes, batch_size):
	y_onehot = torch.FloatTensor(batch_size, num_classes)
	y_onehot.zero_()
	return y_onehot.scatter_(1, class_indices.unsqueeze(1), 1)

def custom_collate(seqs, things=(True, True, True)):
	# print("seqs", seqs)
	# quit()
	# seqs = [(item[0][:opt.maxLength,:], item[1][:opt.maxLength,:]) for item in seqs]
	seqs, labels, categories = zip(*seqs)
	assert len(seqs) == len(labels) == len(categories)
	return [collate_things(item) for item, thing in zip((seqs, labels, categories), things) if thing]

def collate_things(seqs):
	seq_lengths = torch.LongTensor([len(seq) for seq in seqs]).to(device)
	seq_tensor = torch.nn.utils.rnn.pad_sequence(seqs).to(device)

	packed_input = torch.nn.utils.rnn.pack_padded_sequence(seq_tensor, seq_lengths, enforce_sorted=False)
	return packed_input

def train():

	n_fold = opt.nFold
	fold = opt.fold
	lstm_module.train()

	train_indices, _ = get_nth_split(dataset, n_fold, fold)
	train_data = torch.utils.data.Subset(dataset, train_indices)
	if opt.advTraining:
		train_data = AdvDataset(train_data)
		adv_generator = iter(adv(in_training = True))
	train_loader = torch.utils.data.DataLoader(train_data, batch_size=opt.batchSize, shuffle=True, collate_fn=custom_collate)

	optimizer = optim.SGD(lstm_module.parameters(), lr=opt.lr)
	criterion = nn.BCEWithLogitsLoss(reduction="mean")

	writer = SummaryWriter()
	
	samples = 0
	for i in range(1, sys.maxsize):
		if opt.advTraining:
			train_loader.adv_flows, train_loader.categories, av_distance = next(adv_generator)
			writer.add_scalar('adv_avdistance', av_distance, i)
		
		for input_data, labels, flow_categories in train_loader:
			# print("iterating")
			# samples += len(input_data)
			optimizer.zero_grad()
			batch_size = input_data.sorted_indices.shape[0]
			assert batch_size <= opt.batchSize, "batch_size: {}, opt.batchSize: {}".format(batch_size, opt.batchSize)
			lstm_module.init_hidden(batch_size)

			# actual_input = torch.FloatTensor(input_tensor[:,:,:-1]).to(device)

			# print("input_data.data.shape", input_data.data.shape)
			output, seq_lens = lstm_module(input_data)

			# torch.set_printoptions(profile="full")
			# print("output", output.detach().squeeze().transpose(1,0))

			samples += output.shape[1]

			index_tensor = torch.arange(0, output.shape[0], dtype=torch.int64).unsqueeze(1).unsqueeze(2).repeat(1, output.shape[1], output.shape[2])

			selection_tensor = seq_lens.unsqueeze(0).unsqueeze(2).repeat(index_tensor.shape[0], 1, index_tensor.shape[2])-1

			mask = (index_tensor <= selection_tensor).byte().to(device)
			mask_exact = (index_tensor == selection_tensor).byte().to(device)
			# torch.set_printoptions(profile="full")
			# print("mask", mask.squeeze())
			labels, _ = torch.nn.utils.rnn.pad_packed_sequence(labels)
			flow_categories, _ = torch.nn.utils.rnn.pad_packed_sequence(flow_categories)

			loss = criterion(output[mask].view(-1), labels[mask].view(-1))
			loss.backward()

			optimizer.step()

			# print("masked_output_shape", torch.round(output.detach()[mask]).shape, "masked_labels_shape", labels[mask].shape)
			# print("exact_masked_output_shape", torch.round(output.detach()[mask_exact]).shape, "exact_masked_labels_shape", labels[mask_exact].shape)
			# print("output.shape", output.shape, "labels.shape", labels.shape)
			assert output.shape == labels.shape
			writer.add_scalar("loss", loss.item(), samples)
			sigmoided_output = torch.sigmoid(output.detach())
			accuracy = torch.mean((torch.round(sigmoided_output[mask]) == labels[mask]).float())
			writer.add_scalar("accuracy", accuracy, samples)
			end_accuracy = torch.mean((torch.round(sigmoided_output[mask_exact]) == labels[mask_exact]).float())
			writer.add_scalar("end_accuracy", end_accuracy, samples)

			# confidence_for_correct_one = torch.mean(torch.gather(torch.sigmoid(output.detach()[mask]), 2, labels[mask]))
			# writer.add_scalar("confidence", confidence_for_correct_one, i*opt.batchSize)
			# end_confidence_for_correct_one = torch.mean(torch.gather(torch.sigmoid(output.detach()[-1,:,:]), 1, labels[-1,:].unsqueeze(1)))
			# writer.add_scalar("end_confidence", end_confidence_for_correct_one, i*opt.batchSize)

			not_attack_mask = labels == 0
			confidences = sigmoided_output.detach().clone()
			confidences[not_attack_mask] = 1 - confidences[not_attack_mask]
			writer.add_scalar("confidence", torch.mean(confidences[mask]), samples)
			writer.add_scalar("end_confidence", torch.mean(confidences[mask_exact]), samples)
			
			adv_mask = flow_categories >= ADVERSARIAL_THRESH
			if adv_mask.sum() > 0:
				mask &= adv_mask
				exact_mask &= adv_mask
				
				accuracy = torch.mean((torch.round(sigmoided_output[mask]) == labels[mask]).float())
				writer.add_scalar("adv_accuracy", accuracy, samples)
				end_accuracy = torch.mean((torch.round(sigmoided_output[mask_exact]) == labels[mask_exact]).float())
				writer.add_scalar("adv_end_accuracy", end_accuracy, samples)
				writer.add_scalar("adv_confidence", torch.mean(confidences[mask]), samples)
				writer.add_scalar("adv_end_confidence", torch.mean(confidences[mask_exact]), samples)


		# Save after every epoch
		if i % 1 == 0:
			torch.save(lstm_module.state_dict(), '%s/lstm_module_%d.pth' % (writer.log_dir, i))

def test():

	n_fold = opt.nFold
	fold = opt.fold
	lstm_module.eval()

	_, test_indices = get_nth_split(dataset, n_fold, fold)
	test_data = torch.utils.data.Subset(dataset, test_indices)
	test_loader = torch.utils.data.DataLoader(test_data, batch_size=opt.batchSize, shuffle=False, collate_fn=custom_collate)

	all_accuracies = []
	all_end_accuracies = []
	samples = 0

	attack_numbers = mapping.values()

	results_by_attack_number = [list() for _ in range(min(attack_numbers), max(attack_numbers)+1)]

	sample_indices_by_attack_number = [list() for _ in range(min(attack_numbers), max(attack_numbers)+1)]

	for input_data, labels, categories in test_loader:

		batch_size = input_data.sorted_indices.shape[0]
		assert batch_size <= opt.batchSize, "batch_size: {}, opt.batchSize: {}".format(batch_size, opt.batchSize)
		lstm_module.init_hidden(batch_size)

		output, seq_lens = lstm_module(input_data)

		index_tensor = torch.arange(0, output.shape[0], dtype=torch.int64).unsqueeze(1).unsqueeze(2).repeat(1, output.shape[1], output.shape[2])

		selection_tensor = seq_lens.unsqueeze(0).unsqueeze(2).repeat(index_tensor.shape[0], 1, index_tensor.shape[2])-1

		mask = (index_tensor <= selection_tensor).byte().to(device)
		mask_exact = (index_tensor == selection_tensor).byte().to(device)

		input_data, _ = torch.nn.utils.rnn.pad_packed_sequence(input_data)
		labels, _ = torch.nn.utils.rnn.pad_packed_sequence(labels)
		categories, _ = torch.nn.utils.rnn.pad_packed_sequence(categories)

		assert output.shape == labels.shape

		sigmoided_output = torch.sigmoid(output.detach())
		accuracy_items = torch.round(sigmoided_output[mask]) == labels[mask]
		accuracy = torch.mean(accuracy_items.float())
		end_accuracy_items = torch.round(sigmoided_output[mask_exact]) == labels[mask_exact]
		end_accuracy = torch.mean(end_accuracy_items.float())

		all_accuracies.append(accuracy_items.cpu().numpy())
		all_end_accuracies.append(end_accuracy_items.cpu().numpy())

		# Data is (Sequence Index, Batch Index, Feature Index)
		for batch_index in range(output.shape[1]):
			flow_length = seq_lens[batch_index]
			flow_input = input_data[:flow_length,batch_index,:].detach().cpu().numpy()
			flow_output = output[:flow_length,batch_index,:].detach().cpu().numpy()
			assert (categories[0, batch_index,:] == categories[:flow_length, batch_index,:]).all()
			flow_category = int(categories[0, batch_index,:].squeeze().item())

			results_by_attack_number[flow_category].append(np.concatenate((flow_input, flow_output), axis=-1))
			sample_indices_by_attack_number[flow_category].append(test_indices[samples])

			samples += 1

	file_name = opt.dataroot[:-7]+"_prediction_outcomes_{}_{}.pickle".format(opt.fold, opt.nFold)
	with open(file_name, "wb") as f:
		pickle.dump({"results_by_attack_number": results_by_attack_number, "sample_indices_by_attack_number": sample_indices_by_attack_number}, f)

	print("results_by_attack_number", [(index, len(item)) for index, item in enumerate(results_by_attack_number)])

	print("per-packet accuracy", np.mean(np.concatenate(all_accuracies)))
	print("per-flow end-accuracy", np.mean(np.concatenate(all_end_accuracies)))

# Right now this function replaces all values of one feature by random values sampled from the distribution of all features and looks how the accuracy changes.
def feature_importance():

	n_fold = opt.nFold
	fold = opt.fold
	lstm_module.eval()

	_, test_indices = get_nth_split(dataset, n_fold, fold)
	test_data = torch.utils.data.Subset(dataset, test_indices)
	test_x = np.concatenate([item[0][:,:] for item in test_data], axis=0).transpose(1,0)
	test_loader = torch.utils.data.DataLoader(test_data, batch_size=opt.batchSize, shuffle=False, collate_fn=custom_collate)

	attack_numbers = mapping.values()

	results_by_attack_number = [list() for _ in range(min(attack_numbers), max(attack_numbers)+1)]
	randomized_results_by_attack_number = [[list() for _ in range(test_x.shape[0])] for _ in range(min(attack_numbers), max(attack_numbers)+1)]

	for input_data, labels, categories in test_loader:

		batch_size = input_data.sorted_indices.shape[0]
		assert batch_size <= opt.batchSize, "batch_size: {}, opt.batchSize: {}".format(batch_size, opt.batchSize)
		lstm_module.init_hidden(batch_size)

		output, seq_lens = lstm_module(input_data)

		index_tensor = torch.arange(0, output.shape[0], dtype=torch.int64).unsqueeze(1).unsqueeze(2).repeat(1, output.shape[1], output.shape[2])

		selection_tensor = seq_lens.unsqueeze(0).unsqueeze(2).repeat(index_tensor.shape[0], 1, index_tensor.shape[2])-1

		mask = (index_tensor <= selection_tensor).byte().to(device)
		# mask_exact = (index_tensor == selection_tensor).byte().to(device)

		# input_data, _ = torch.nn.utils.rnn.pad_packed_sequence(input_data)
		labels_padded, _ = torch.nn.utils.rnn.pad_packed_sequence(labels)
		categories_padded, _ = torch.nn.utils.rnn.pad_packed_sequence(categories)

		assert output.shape == labels_padded.shape

		sigmoided_output = torch.sigmoid(output.detach())
		# accuracy_items = torch.round(sigmoided_output[mask]) == labels[mask]
		# end_accuracy_items = torch.round(sigmoided_output[mask_exact]) == labels[mask_exact]

		# Data is (Sequence Index, Batch Index, Feature Index)
		for batch_index in range(output.shape[1]):
			flow_length = seq_lens[batch_index]
			flow_output = (torch.round(sigmoided_output[:flow_length,batch_index,:]) == labels_padded[:flow_length,batch_index,:]).detach().cpu().numpy()
			assert (categories_padded[0, batch_index,:] == categories_padded[:flow_length, batch_index,:]).all()
			flow_category = int(categories_padded[0, batch_index,:].squeeze().item())

			results_by_attack_number[flow_category].append(flow_output)

		for feature_index in range(test_x.shape[0]):
			lstm_module.init_hidden(batch_size)

			# print("input_data.data.shape", input_data.data.shape, "input_data.data[:,feature_index].shape", input_data.data[:,feature_index].shape, "torch.FloatTensor(np.random.choice(test_x[feature_index], size=input_data.data.shape[0])).shape", torch.FloatTensor(np.random.choice(test_x[feature_index], size=input_data.data.shape[0])).shape)
			input_data_cloned = torch.nn.utils.rnn.PackedSequence(input_data.data.detach().clone(), input_data.batch_sizes, input_data.sorted_indices, input_data.unsorted_indices)
			input_data_cloned.data.data[:,feature_index] = torch.FloatTensor(np.random.choice(test_x[feature_index], size=(input_data_cloned.data.data.shape[0]))).to(device)
			output, seq_lens = lstm_module(input_data_cloned)

			# index_tensor = torch.arange(0, output.shape[0], dtype=torch.int64).unsqueeze(1).unsqueeze(2).repeat(1, output.shape[1], output.shape[2])

			# selection_tensor = seq_lens.unsqueeze(0).unsqueeze(2).repeat(index_tensor.shape[0], 1, index_tensor.shape[2])-1

			# mask = (index_tensor <= selection_tensor).byte().to(device)
			# # mask_exact = (index_tensor == selection_tensor).byte().to(device)

			# # input_data, _ = torch.nn.utils.rnn.pad_packed_sequence(input_data)
			# labels_padded, _ = torch.nn.utils.rnn.pad_packed_sequence(labels)
			# categories_padded, _ = torch.nn.utils.rnn.pad_packed_sequence(categories)

			# assert output.shape == labels_padded.shape

			sigmoided_output = torch.sigmoid(output.detach())
			# accuracy_items = torch.round(sigmoided_output[mask]) == labels[mask]
			# end_accuracy_items = torch.round(sigmoided_output[mask_exact]) == labels[mask_exact]

			# Data is (Sequence Index, Batch Index, Feature Index)
			for batch_index in range(output.shape[1]):
				flow_length = seq_lens[batch_index]
				flow_output = (torch.round(sigmoided_output[:flow_length,batch_index,:]) == labels_padded[:flow_length,batch_index,:]).detach().cpu().numpy()
				assert (categories_padded[0, batch_index,:] == categories_padded[:flow_length, batch_index,:]).all()
				flow_category = int(categories_padded[0, batch_index,:].squeeze().item())

				randomized_results_by_attack_number[flow_category][feature_index].append(flow_output)

	accuracy = np.mean(np.concatenate([subitem for item in results_by_attack_number for subitem in item], axis=0))
	print("accuracy", accuracy)
	for feature_index in range(test_x.shape[0]):
		accuracy_for_feature = np.mean(np.concatenate([feature for attack_type in randomized_results_by_attack_number for feature in attack_type[feature_index]]))
		print("accuracy_for_feature", feature_index, accuracy_for_feature)

	# print("results_by_attack_number", [(index, len(item)) for index, item in enumerate(results_by_attack_number)])

def adv(in_training = False):
	# FIXME: They suggest at least 10000 iterations with some specialized optimizer (Adam)
	# with SGD we probably need even more.
	ITERATION_COUNT = 100 if in_training else 1000
	
	# generate adversarial samples using Carlini Wagner method
	n_fold = opt.nFold
	fold = opt.fold

	if not opt.canManipulateBothDirections:
		bidirectional_categories = [torch.FloatTensor([mapping[key]]).to(device) for key in categories_mapping["Botnet"]]

	# print("bidirectional_categories", bidirectional_categories)

	#initialize sample
	train_indices, test_indices = get_nth_split(dataset, n_fold, fold)
	indices = train_indices if in_training else test_indices
	subset_with_all_traffic = torch.utils.data.Subset(dataset, indices)

	feature_ranges = get_feature_ranges(subset_with_all_traffic, sampling_density=2)
	max_packet_length = feature_ranges[0][0]

	common_mtu_scaled = (1500 - means[3])/stds[3]
	maximum_length = common_mtu_scaled

	zero_scaled = (0 - means[4])/stds[4]

	orig_indices, attack_indices = zip(*[(orig, i) for orig, i in zip(indices, range(len(subset_with_all_traffic))) if subset_with_all_traffic[i][1][0,0] == 1])

	subset = torch.utils.data.Subset(dataset, attack_indices)

	loader = torch.utils.data.DataLoader(subset, batch_size=opt.batchSize, shuffle=False, collate_fn=lambda x: custom_collate(x, (True, False, True)))

	# lengths = torch.LongTensor([len(seq) for seq in batch_x])
	# packed = torch.nn.utils.rnn.pack_sequence(batch_x, enforce_sorted=False)

	#optimizer = optim.SGD([sample], lr=opt.lr)

	# iterate until done
	finished_adv_samples = [None] * len(subset)
	finished_categories = [ item[2][0,0] for item in subset ]

	zero_tensor = torch.FloatTensor([0]).to(device)

	samples = 0
	distances = []
	if in_training:
		# repeat forever
		sample_generator = itertools.chain.from_iterable(( enumerate(loader) for _ in itertools.count()))
	else:
		sample_generator = enumerate(loader)
	
	
	for sample_index, (input_data,input_categories) in sample_generator:
		# print("sample", sample_index)
		total_sample = samples % len(subset)
		samples += input_data.sorted_indices.shape[0]

		optimizer = optim.SGD([input_data.data], lr=opt.lr)

		# index_tensor = torch.arange(0, len(input_data.batch_sizes), dtype=torch.int64).unsqueeze(1).unsqueeze(2).repeat(1, input_data.batch_sizes[0], 1)

		# _, seq_lens = torch.nn.utils.rnn.pad_packed_sequence(input_data)

		# selection_tensor = seq_lens.unsqueeze(0).unsqueeze(2).repeat(index_tensor.shape[0], 1, index_tensor.shape[2])-1

		# mask = (index_tensor <= selection_tensor).byte().to(device)

		orig_batch = input_data.data.clone()
		orig_batch_padded = torch.nn.utils.rnn.pad_packed_sequence(input_data)[0].detach()

		if finished_adv_samples[-1] is not None:
			numpy_seqs = finished_adv_samples[total_sample:(total_sample+input_data.sorted_indices.shape[0])]
			new_data = collate_things([ torch.FloatTensor(seq) for seq in numpy_seqs]).data
			assert input_data.data.shape == new_data.shape
			input_data.data.data = new_data
		input_data.data.requires_grad = True

		seqs, lengths = torch.nn.utils.rnn.pad_packed_sequence(input_data)
		
		if opt.allowIATReduction:
			same_direction_mask = torch.cat((seqs[:1,:,5]!=seqs[:1,:,5], seqs[1:,:,5] == seqs[:-1,:,5]))
			same_direction_mask = torch.nn.utils.rnn.pack_padded_sequence(same_direction_mask, lengths, enforce_sorted=False).data.data
		else:
			same_direction_mask = False

		if not opt.canManipulateBothDirections:
			cats, lengths = torch.nn.utils.rnn.pad_packed_sequence(input_categories)

			forward_direction = seqs[0:1,:,5].repeat(seqs.shape[0],1)

			index_tensor = torch.arange(0, seqs.shape[0], dtype=torch.int64).unsqueeze(1).repeat(1, seqs.shape[1])

			selection_tensor = lengths.unsqueeze(0).repeat(index_tensor.shape[0], 1)-1

			# print("index_tensor.shape", index_tensor.shape, "selection_tensor.shape", selection_tensor.shape)

			orig_mask = (index_tensor <= selection_tensor).byte().to(device)
			# mask_exact = (index_tensor == selection_tensor).byte().to(device)
			wrong_direction = (seqs[:,:,5]!=forward_direction)
			matching_cats = [(cats==bidirectional_cat).squeeze() for bidirectional_cat in bidirectional_categories]
			# print("matching_cats.shape", [item.shape for item in matching_cats])
			not_bidirectional = ~reduce(lambda acc, x: acc | x, matching_cats, torch.ByteTensor([False]).to(device))
			# print(orig_mask.shape, wrong_direction.shape, not_bidirectional.shape)
			mask = orig_mask & wrong_direction & not_bidirectional

			# print("Batch: {}, orig_mask: {}, wrong_direction: {}, not_bidirectional: {}, mask: {}".format(sample_index, (torch.sum(orig_mask)), (torch.sum(wrong_direction & orig_mask)), (torch.sum(not_bidirectional & orig_mask)), (torch.sum(mask))))

			print("Batch: {}, wrong_direction: {}, not_bidirectional: {}, invalid: {}".format(sample_index, float(torch.sum(wrong_direction & orig_mask)/torch.sum(orig_mask, dtype=torch.float32)), float(torch.sum(not_bidirectional & orig_mask)/torch.sum(orig_mask, dtype=torch.float32)), float(torch.sum(mask)/torch.sum(orig_mask, dtype=torch.float32))))

		for i in range(ITERATION_COUNT):

			# print("iterating", i)
			# samples += len(input_data)
			optimizer.zero_grad()
			lstm_module.init_hidden(input_data.sorted_indices.shape[0])

			# actual_input = torch.FloatTensor(input_tensor[:,:,:-1]).to(device)

			# print("input_data.data.shape", input_data.data.shape)
			#s_squeezed = torch.nn.utils.rnn.pack_padded_sequence(torch.unsqueeze(sample, 1), [sample.shape[0]])
			output, seq_lens = lstm_module(input_data)
			# output_padded, lengths = torch.nn.utils.rnn.pad_packed_sequence(input_data)

			distance = torch.dist(orig_batch, input_data.data, p=1)
			#regularizer = .5*(torch.max(output[other_attacks]) - output[target_attack])
			#regularizer = .5*output[-1,0]
			regularizer = opt.tradeoff*torch.max(output, zero_tensor).sum()
			#if regularizer <= 0:
				#break
			criterion = distance + regularizer
			if opt.penaltyTradeoff > 0:
				seqs, lengths = torch.nn.utils.rnn.pad_packed_sequence(input_data)
				penalty = opt.penaltyTradeoff*((seqs[:,:,4].sum(0) - orig_batch_padded[:,:,4].sum(0))**2).sum()
				criterion += penalty
			criterion.backward()

			# only consider lengths and iat
			input_data.data.grad[:,:3] = 0
			input_data.data.grad[:,5:] = 0
			optimizer.step()

			# Packet lengths cannot become smaller than original
			packet_mask = input_data.data[:,3] < orig_batch[:,3]
			input_data.data.data[packet_mask,3] = orig_batch[packet_mask,3]

			# # Packet lengths cannot become larger than the maximum. Should be around 1500 bytes usually... NOTE: Apparently packets are commonly larger than 1500 bytes so this is not enforcable like this :/
			# mask = input_data.data[:,3] > maximum_length
			# input_data.data.data[mask,3] = orig_batch[mask,3]

			# XXX: This is experimentally removed
			# # IAT cannot become smaller than 0 when the preceding packet went in the same direction
			iat_mask = (input_data.data[:,4] < zero_scaled) & same_direction_mask
			input_data.data.data[iat_mask,4] = float(zero_scaled)

			# XXX: This is experimentally added
			iat_mask = (input_data.data[:,4] < orig_batch[:,4]) & ~same_direction_mask
			input_data.data.data[iat_mask,4] = orig_batch[iat_mask,4]

			# Can only manipulate attacker direction except for botnets where we can control both sides
			if not opt.canManipulateBothDirections:
				seqs, lengths = torch.nn.utils.rnn.pad_packed_sequence(input_data)

				seqs[mask] = orig_batch_padded[mask]
				input_data.data.data = torch.nn.utils.rnn.pack_padded_sequence(seqs, lengths, enforce_sorted=False).data.data

			seqs, lengths = torch.nn.utils.rnn.pad_packed_sequence(input_data)

			seqs[0,:,4] = zero_scaled
			input_data.data.data = torch.nn.utils.rnn.pack_padded_sequence(seqs, lengths, enforce_sorted=False).data.data

			# detached_batch = input_data.data.detach()

			# # Packet lengths cannot become smaller than original
			# mask = detached_batch[:,3] < orig_batch[:,3]
			# detached_batch[mask,3] = orig_batch[mask,3]

			# # IAT cannot become smaller 0
			# mask = detached_batch[:,4] * stds[4] + means[4] < 0
			# detached_batch[mask,4] = float(-means[4]/stds[4])

			# if i % 1000 == 0:
			# 	print('Iteration: %d, Distance: %f, regularizer: %f' % (i, distance, regularizer))

		seqs, lengths = torch.nn.utils.rnn.pad_packed_sequence(input_data)

		# adv_samples = [ seqs[:lengths[batch_index],batch_index,:].detach().cpu().numpy()*stds + means for batch_index in range(seqs.shape[1]) ]
		adv_samples = [ seqs[:lengths[batch_index],batch_index,:].detach().cpu().numpy() for batch_index in range(seqs.shape[1]) ]

		finished_adv_samples[total_sample:(total_sample+len(adv_samples))] = adv_samples
		if in_training:
			distances.append(torch.dist(orig_batch, input_data.data, p=1)/seqs.shape[1])
			# keep iterations for adversarial flows about the same as iterations for training
			if finished_adv_samples[-1] is not None and len(distances) >= (len(subset) / opt.batchSize / ITERATION_COUNT):
				yield finished_adv_samples, finished_categories, sum(distances)/len(distances)
				distances = []

	# print("samples", samples)
	assert len(finished_adv_samples) == len(subset), "len(finished_adv_samples): {}, len(subset): {}".format(len(finished_adv_samples), len(subset))

	original_dataset = OurDataset(*zip(*[[subitem.numpy() for subitem in item] for item in list(subset)]))
	subset = OurDataset(*zip(*[[subitem.numpy() for subitem in item] for item in list(subset)]))
	subset.data = finished_adv_samples

	original_results = eval_nn(original_dataset)
	results = eval_nn(subset)

	assert len(results) == len(subset)

	print("Tradeoff: {}".format(opt.tradeoff))
	print("Number of attack samples: {}".format(len(subset)))
	print("Average confidence on original packets: {}".format(np.mean(numpy_sigmoid(np.concatenate([np.array(item) for item in original_results], axis=0)))))
	print("Average confidence on packets: {}".format(np.mean(numpy_sigmoid(np.concatenate([np.array(item) for item in results], axis=0)))))
	print("Ratio of successful adversarial attacks on packets: {}".format(1-np.mean(np.round(numpy_sigmoid(np.concatenate([np.array(item) for item in results], axis=0))))))
	print("Average confidence on original flows: {}".format(np.mean(numpy_sigmoid(np.array([item[-1] for item in original_results])))))
	print("Average confidence on flows: {}".format(np.mean(numpy_sigmoid(np.array([item[-1] for item in results])))))
	print("Ratio of successful adversarial attacks on flows: {}".format(1-np.mean(np.round(numpy_sigmoid(np.array([item[-1] for item in results]))))))

	attack_numbers = mapping.values()

	orig_flows_by_attack_number = [list() for _ in range(min(attack_numbers), max(attack_numbers)+1)]
	modified_flows_by_attack_number = [list() for _ in range(min(attack_numbers), max(attack_numbers)+1)]
	orig_results_by_attack_number = [list() for _ in range(min(attack_numbers), max(attack_numbers)+1)]
	results_by_attack_number = [list() for _ in range(min(attack_numbers), max(attack_numbers)+1)]
	sample_indices_by_attack_number = [list() for _ in range(min(attack_numbers), max(attack_numbers)+1)]

	for orig_index, (orig_flow,_,cat), (adv_flow,_,_), orig_result, result in zip(orig_indices, original_dataset, subset, original_results, results):
		# print("cat", cat)
		correct_cat = int(cat[0][0])
		orig_flows_by_attack_number[correct_cat].append(orig_flow)
		modified_flows_by_attack_number[correct_cat].append(adv_flow)
		orig_results_by_attack_number[correct_cat].append(orig_result)
		results_by_attack_number[correct_cat].append(result)
		sample_indices_by_attack_number.append(orig_index)

	reverse_mapping = {v: k for k, v in mapping.items()}
	for attack_number, (per_attack_orig, per_attack_modified, per_attack_orig_results, per_attack_results) in enumerate(zip(orig_flows_by_attack_number, modified_flows_by_attack_number, orig_results_by_attack_number, results_by_attack_number)):
		if len(per_attack_results) <= 0:
			continue
		per_packet_orig_accuracy = (np.mean(np.round(numpy_sigmoid(np.concatenate([np.array(item) for item in per_attack_orig_results], axis=0)))))
		per_packet_accuracy = (np.mean(np.round(numpy_sigmoid(np.concatenate([np.array(item) for item in per_attack_results], axis=0)))))
		per_flow_orig_accuracy = (np.mean(np.round(numpy_sigmoid(np.array([item[-1] for item in per_attack_orig_results])))))
		per_flow_accuracy = (np.mean(np.round(numpy_sigmoid(np.array([item[-1] for item in per_attack_results])))))
		dist = np.array([np.linalg.norm(per_attack_orig_item-per_attack_modified_item, ord=1).mean() for per_attack_orig_item, per_attack_modified_item in zip(per_attack_orig, per_attack_modified)]).mean()

		print("Attack type: {}; number of samples: {}, average dist: {}, packet confidence: {}/{}, flow confidence: {}/{}".format(reverse_mapping[attack_number], len(per_attack_results), dist, per_packet_accuracy, per_packet_orig_accuracy, per_flow_accuracy, per_flow_orig_accuracy))

	file_name = opt.dataroot[:-7]+"_adv_{}{}_outcomes_{}_{}.pickle".format(opt.tradeoff, "_notBidirectional" if not opt.canManipulateBothDirections else "", opt.fold, opt.nFold)
	with open(file_name, "wb") as f:
		pickle.dump({"results_by_attack_number": results_by_attack_number, "orig_results_by_attack_number": orig_results_by_attack_number, "modified_flows_by_attack_number": modified_flows_by_attack_number, "orig_flows_by_attack_number": orig_flows_by_attack_number}, f)

	# with open('adv_samples.pickle', 'wb') as outfile:
	# 	pickle.dump(adv_samples, outfile)

def eval_nn(data):

	lstm_module.eval()

	results = []

	loader = torch.utils.data.DataLoader(data, batch_size=opt.batchSize, shuffle=False, collate_fn=lambda x: custom_collate(x, (True, False, False)))

	for (input_data,) in loader:

		lstm_module.init_hidden(opt.batchSize)

		output, seq_lens = lstm_module(input_data)

		# Data is (Sequence Index, Batch Index, Feature Index)
		for batch_index in range(output.shape[1]):
			flow_length = seq_lens[batch_index]
			#flow_input = input_data[:flow_length,batch_index,:].detach().cpu().numpy()
			flow_output = output[:flow_length,batch_index,:].detach().cpu().numpy()

			results.append(flow_output)

	return results

def get_feature_ranges(dataset, sampling_density=100):
	features = []
	# iat & length
	for feat_name, feat_ind in zip(["length", "iat"], [3, 4]):
		feat_min = min( (sample[0][i,feat_ind] for sample in dataset for i in range(sample[0].shape[0])))
		feat_max = max( (sample[0][i,feat_ind] for sample in dataset for i in range(sample[0].shape[0])))
		features.append((feat_ind,np.linspace(feat_min, feat_max, sampling_density)))

		# print("feature", feat_name, "min", feat_min, "max", feat_max, "min_rescaled", feat_min*stds[feat_ind] + means[feat_ind], "max_rescaled", feat_max*stds[feat_ind] + means[feat_ind])
	return features

def pred_plots():
	OUT_DIR='pred_plots'
	os.makedirs(OUT_DIR, exist_ok=True)

	n_fold = opt.nFold
	fold = opt.fold
	lstm_module.eval()

	_, test_indices = get_nth_split(dataset, n_fold, fold)
	subset = torch.utils.data.Subset(dataset, test_indices)

	features = get_feature_ranges(subset)

	attack_numbers = mapping.values()

	results_by_attack_number = [list() for _ in range(min(attack_numbers), max(attack_numbers)+1)]
	sample_indices_by_attack_number = [list() for _ in range(min(attack_numbers), max(attack_numbers)+1)]

	start_iterating = time.time()
	# have_categories = collections.defaultdict(int)
	for real_ind, sample_ind in zip(test_indices, range(len(subset))):
		# print("index", sample_ind)
		# if have_categories[cat] == SAMPLES_PER_ATTACK:
		# 	continue
		# have_categories[cat] += 1

		flow, _, flow_categories = subset[sample_ind]
		cat = int(flow_categories[0,0])

		lstm_module.init_hidden(1)

		predictions = np.zeros((flow.shape[0],))
		mins = np.ones((len(features),flow.shape[0],))
		maxs = np.zeros((len(features),flow.shape[0],))

		for i in range(flow.shape[0]):

			lstm_module.forgetting = True

			input_data = torch.FloatTensor(flow[i,:][None,None,:]).repeat(1,len(features)*len(features[0][1]),1)
			for k, (feat_ind, values) in enumerate(features):

				for j in range(values.size):
					input_data[0,k*values.size+j,feat_ind] = values[j]

			packed_input = torch.nn.utils.rnn.pack_padded_sequence(input_data, [1] *input_data.shape[1]).to(device)

			lstm_module.hidden = (lstm_module.hidden[0].repeat(1,input_data.shape[1],1), lstm_module.hidden[1].repeat(1,input_data.shape[1],1))
			# print("hidden before", lstm_module.hidden)
			output, _ = lstm_module(packed_input)
			sigmoided = torch.sigmoid(output[0,:,0]).detach().cpu().tolist()

			for k, (feat_ind, values) in enumerate(features):
				for j in range(values.size):
					mins[k,i] = min(mins[k,i], *sigmoided[k*values.size:(k+1)*values.size])
					maxs[k,i] = max(maxs[k,i], *sigmoided[k*values.size:(k+1)*values.size])

			lstm_module.hidden = (lstm_module.hidden[0][:,0:1,:].contiguous(), lstm_module.hidden[1][:,0:1,:].contiguous())
			# print("hidden before", lstm_module.hidden)
			lstm_module.forgetting = False
			packed_input = torch.nn.utils.rnn.pack_padded_sequence(torch.FloatTensor(flow[i,:][None,None,:]), [1]).to(device)
			output, _ = lstm_module(packed_input)
			predictions[i] = torch.sigmoid(output[0,0,0])

		results_by_attack_number[cat].append(np.vstack((predictions,mins,maxs)))
		sample_indices_by_attack_number[cat].append(real_ind)

	print("It took {} seconds per sample".format((time.time()-start_iterating)/len(subset)))
	file_name = opt.dataroot[:-7]+"_pred_plots_outcomes_{}_{}.pickle".format(opt.fold, opt.nFold)
	with open(file_name, "wb") as f:
		pickle.dump({"results_by_attack_number": results_by_attack_number, "sample_indices_by_attack_number": sample_indices_by_attack_number}, f)

def pred_plots2():
	OUT_DIR='pred_plots2'
	os.makedirs(OUT_DIR, exist_ok=True)

	n_fold = opt.nFold
	fold = opt.fold
	lstm_module.eval()

	_, test_indices = get_nth_split(dataset, n_fold, fold)
	subset = torch.utils.data.Subset(dataset, test_indices)

	features = get_feature_ranges(subset, sampling_density=100)
	# print("features", features)

	attack_numbers = mapping.values()

	results_by_attack_number = [list() for _ in range(min(attack_numbers), max(attack_numbers)+1)]
	flows_by_attack_number = [list() for _ in range(min(attack_numbers), max(attack_numbers)+1)]
	result_ranges_by_attack_number = [list() for _ in range(min(attack_numbers), max(attack_numbers)+1)]
	sample_indices_by_attack_number = [list() for _ in range(min(attack_numbers), max(attack_numbers)+1)]

	start_iterating = time.time()
	# have_categories = collections.defaultdict(int)
	for real_ind, sample_ind in zip(test_indices, range(len(subset))):
		# print("index", sample_ind)
		# if have_categories[cat] == SAMPLES_PER_ATTACK:
		# 	continue
		# have_categories[cat] += 1

		flow, _, flow_categories = subset[sample_ind]
		cat = int(flow_categories[0,0])

		lstm_module.init_hidden(1)

		predictions = np.zeros((flow.shape[0],))
		prediction_ranges = np.zeros((flow.shape[0], len(features), (len(features[0][1])-1)))

		for i in range(flow.shape[0]):

			lstm_module.forgetting = True

			input_data = torch.FloatTensor(flow[i,:][None,None,:]).repeat(1,len(features)*(len(features[0][1])-1),1)
			for k, (feat_ind, values) in enumerate(features):

				for j in range(values.size-1):
					# print("input_data.shape", input_data.shape, "k*values.size+j", k*values.size+j)
					input_data[0,k*(values.size-1)+j,feat_ind] = (values[j]+values[j+1])/2

			packed_input = torch.nn.utils.rnn.pack_padded_sequence(input_data, [1] *input_data.shape[1]).to(device)

			# print("input_data.shape", input_data.shape)
			# print("hidden before before", [item.shape for item in lstm_module.hidden])
			lstm_module.hidden = (lstm_module.hidden[0].repeat(1,input_data.shape[1],1), lstm_module.hidden[1].repeat(1,input_data.shape[1],1))
			# print("hidden before", [item.shape for item in lstm_module.hidden])
			output, _ = lstm_module(packed_input)
			sigmoided = torch.sigmoid(output[0,:,0]).detach().cpu().tolist()

			for k, (feat_ind, values) in enumerate(features):
				for j in range(values.size-1):
					prediction_ranges[i,k,:] = sigmoided[k*(values.size-1):(k+1)*(values.size-1)]

			lstm_module.hidden = (lstm_module.hidden[0][:,0:1,:].contiguous(), lstm_module.hidden[1][:,0:1,:].contiguous())
			# print("hidden before", lstm_module.hidden)
			lstm_module.forgetting = False
			packed_input = torch.nn.utils.rnn.pack_padded_sequence(torch.FloatTensor(flow[i,:][None,None,:]), [1]).to(device)
			output, _ = lstm_module(packed_input)
			predictions[i] = torch.sigmoid(output[0,0,0])

		results_by_attack_number[cat].append(predictions)
		flows_by_attack_number[cat].append(flow.detach().cpu().numpy())
		result_ranges_by_attack_number[cat].append(prediction_ranges)
		sample_indices_by_attack_number[cat].append(real_ind)

		# assert result_ranges_by_attack_number[cat][-1].__class__.__name__=="ndarray" and flows_by_attack_number[cat][-1].__class__.__name__=="ndarray" and result_ranges_by_attack_number[cat][-1].__class__.__name__=="ndarray" and sample_indices_by_attack_number[cat][-1].__class__.__name__=="int", "{}, {}, {}, {}".format(result_ranges_by_attack_number[cat][-1].__class__.__name__, flows_by_attack_number[cat][-1].__class__.__name__, result_ranges_by_attack_number[cat][-1].__class__.__name__, sample_indices_by_attack_number[cat][-1].__class__.__name__)

	print("It took {} seconds per sample".format((time.time()-start_iterating)/len(subset)))
	file_name = opt.dataroot[:-7]+"_pred_plots2_outcomes_{}_{}.pickle".format(opt.fold, opt.nFold)
	with open(file_name, "wb") as f:
		pickle.dump({"results_by_attack_number": results_by_attack_number, "flows_by_attack_number": flows_by_attack_number, "result_ranges_by_attack_number": result_ranges_by_attack_number, "sample_indices_by_attack_number": sample_indices_by_attack_number, "features": features}, f)

def pdp():

	feature_names = ["srcPort", "dstPort"]
	n_fold = opt.nFold
	fold = opt.fold
	lstm_module.eval()

	_, test_indices = get_nth_split(dataset, n_fold, fold)
	subset = torch.utils.data.Subset(dataset, test_indices)

	attack_numbers = mapping.values()

	results_by_attack_number = [None for _ in range(min(attack_numbers), max(attack_numbers)+1)]

	minmax = {feat_ind: (min((sample[0,feat_ind] for sample in x)), max((sample[0,feat_ind] for sample in x))) for feat_ind in [0,1] }
	# TODO: consider fold
	for attack_number in range(max(attack_numbers)+1):

		print("attack_number", attack_number)
		results_for_attack_type = []
		for feat_name, feat_ind in zip(feature_names, (0, 1)):
			feat_min, feat_max = minmax[feat_ind]

			values = np.linspace(feat_min, feat_max, 100)

			matching = [item for item in subset if int(item[2][0,0]) == attack_number]
			if len(matching) <= 0:
				break
			good_subset = OurDataset(*zip(*matching))
			
			# subset = [ torch.FloatTensor(sample) for sample in x[:opt.batchSize] ]

			pdp = np.zeros([values.size])

			for i in range(values.size):
				# good_subset.data consists of torch tensors. We are therefore able to
				# modify the dataset directly using the return value of __getitem__().
				# This does not modify the global dataset, which holds the data as numpy
				# arrays.
				for sample in good_subset:
					for j in range(sample[0].shape[0]):
						sample[0][j,feat_ind] = values[i]
				outputs = eval_nn(good_subset)
				# TODO: avg. or end output?
				pdp[i] = np.mean( np.array([numpy_sigmoid(output[-1]) for output in outputs] ))

			rescaled = values * stds[feat_ind] + means[feat_ind]
			# os.makedirs(PDP_DIR, exist_ok=True)
			results_for_attack_type.append(np.vstack((rescaled,pdp)))
			# print("result.shape", result.shape)
			# np.save('%s/%s.npy' % (PDP_DIR, feat_name), result)

		else:
			results_by_attack_number[attack_number] = np.stack(results_for_attack_type)

	file_name = opt.dataroot[:-7]+"_pdp_outcomes_{}_{}.pickle".format(opt.fold, opt.nFold)
	with open(file_name, "wb") as f:
		pickle.dump({"results_by_attack_number": results_by_attack_number, "feature_names": feature_names}, f)

def plot_histograms():
	rescaled = [item * stds + means for item in x[:opt.maxSize]]
	for i in range(rescaled[0].shape[-1]):
		plt.hist([subitem for item in rescaled for subitem in list(item[:,i])], bins=100)
		plt.title("Feature {}".format(i))
		plt.show()

def overwrite_manipulable_entries(seq, filler=-1):
		forward_direction = seq[0,5]

		wrong_direction = (seq[:,5]==forward_direction)
		seq[wrong_direction,:][:,3:5] = filler
		return seq

if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--canManipulateBothDirections', action='store_true', help='if the attacker can change packets in both directions of the flow')
	parser.add_argument('--dataroot', required=True, help='path to dataset')
	parser.add_argument('--normalizationData', default="", type=str, help='normalization data to use')
	parser.add_argument('--fold', type=int, default=0, help='fold to use')
	parser.add_argument('--nFold', type=int, default=10, help='total number of folds')
	parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
	parser.add_argument('--net', default='', help="path to net (to continue training)")
	parser.add_argument('--function', default='train', help='the function that is going to be called')
	parser.add_argument('--manualSeed', default=0, type=int, help='manual seed')
	parser.add_argument('--maxLength', type=int, default=1000, help='max length')
	parser.add_argument('--maxSize', type=int, default=sys.maxsize, help='limit of samples to consider')
	parser.add_argument("--categoriesMapping", type=str, default="categories_mapping.json", help="mapping of attack categories; see parse.py")
	parser.add_argument('--removeChangeable', action='store_true', help='when training remove all features that an attacker could manipulate easily without changing the attack itself')
	parser.add_argument('--tradeoff', type=float, default=0.5, help='max length')
	parser.add_argument('--penaltyTradeoff', type=float, default=0, help='Tradeoff to enforce constant flow duration')
	parser.add_argument('--lr', type=float, default=10**(-2), help='learning rate')
	parser.add_argument('--advTraining', action='store_true', help='Train with adversarial flows')
	parser.add_argument('--allowIATReduction', action='store_true', help='Allow reducing IAT below original value')

	opt = parser.parse_args()
	print(opt)
	SEED = opt.manualSeed
	random.seed(SEED)
	np.random.seed(SEED)
	torch.manual_seed(SEED)

	with open (opt.dataroot, "rb") as f:
		all_data = pickle.load(f)

	with open("categories_mapping.json", "r") as f:
		categories_mapping_content = json.load(f)
	categories_mapping, mapping = categories_mapping_content["categories_mapping"], categories_mapping_content["mapping"]

	all_data = [item[:opt.maxLength,:] for item in all_data] # if np.random.rand() < 0.005]
	if opt.removeChangeable:
		all_data = [overwrite_manipulable_entries(item) for item in all_data]
	random.shuffle(all_data)
	# print("lens", [len(item) for item in all_data])
	x = [item[:, :-2] for item in all_data]
	y = [item[:, -1:] for item in all_data]
	categories = [item[:, -2:-1] for item in all_data]

	if opt.normalizationData == "":
		file_name = opt.dataroot[:-7]+"_normalization_data.pickle"
		catted_x = np.concatenate(x, axis=0)
		means = np.mean(catted_x, axis=0)
		stds = np.std(catted_x, axis=0)
		stds[stds==0.0] = 1.0

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

	dataset = OurDataset(x, y, categories)

	batchSize = 1 if opt.function == 'pred_plots' else opt.batchSize # fixme
	lstm_module = OurLSTMModule(x[0].shape[-1], y[0].shape[-1], HIDDEN_SIZE, N_LAYERS, batchSize, device).to(device)

	if opt.net != '':
		print("Loading", opt.net)
		lstm_module.load_state_dict(torch.load(opt.net, map_location=device))

	globals()[opt.function]()
