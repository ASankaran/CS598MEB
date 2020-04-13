import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim


classifications = [
	'Ovary-AdenoCA', 'CNS-PiloAstro', 'Liver-HCC', 'Panc-Endocrine',
	'Kidney-RCC', 'Prost-AdenoCA', 'Lymph-BNHL', 'Panc-AdenoCA',
	'Eso-AdenoCa', 'CNS-Medullo', 'Lymph-CLL', 'Skin-Melanoma',
	'Stomach-AdenoCA', 'Breast-AdenoCa', 'Head-SCC', 'Lymph-NOS',
	'Myeloid-AML', 'Biliary-AdenoCA', 'Bone-Osteosarc', 'Breast-DCIS',
	'Breast-LobularCa', 'Myeloid-MPN', 'Myeloid-MDS', 'Bone-Cart',
	'Bone-Epith'
]


class MutationDataset(Dataset):
	def __init__(self, feature_file, label_file):
		self.samples = pd.read_csv(feature_file, header=None)
		self.labels = pd.read_csv(label_file, header=None, usecols=[1])

		assert len(self.samples) == len(self.labels)

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, index):
		if torch.is_tensor(index):
			index = index.tolist()

		sample = torch.tensor(self.samples.iloc[index].values)
		label = torch.tensor([classifications.index(label) for label in list(self.labels.iloc[index].values)])

		return sample.float(), label


class ConvMutationNet(nn.Module):
	def __init__(self):
		super().__init__()
		self.cv01 = nn.Conv1d(1, 2, kernel_size=5)
		self.cv02 = nn.Conv1d(2, 2, kernel_size=5)
		self.cv03 = nn.Conv1d(2, 4, kernel_size=5)
		self.cv04 = nn.Conv1d(4, 4, kernel_size=5)
		self.cv05 = nn.Conv1d(4, 8, kernel_size=5)
		self.cv06 = nn.Conv1d(8, 8, kernel_size=5)
		self.pool = nn.MaxPool1d(2, 2)
		self.fc01 = nn.Linear(8 * 711, 25)

	def forward(self, x):
		x = F.relu(self.cv01(x))
		x = F.relu(self.cv02(x))
		x = self.pool(x)
		x = F.relu(self.cv03(x))
		x = F.relu(self.cv04(x))
		x = self.pool(x)
		x = F.relu(self.cv05(x))
		x = F.relu(self.cv06(x))
		x = self.pool(x)
		x = x.view(-1, 8 * 711)
		x = self.fc01(x)
		return x


def setup():
	return ConvMutationNet()

def train(feature_file, label_file, batch_size=32, epochs=50):
	train_dataset = MutationDataset(feature_file + '.train', label_file + '.train')
	test_dataset = MutationDataset(feature_file + '.test', label_file + '.test')

	train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
	test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

	net = setup()
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(net.parameters(), lr=0.001)

	net.train()

	losses = []
	accuracies = []

	for epoch in range(epochs):
		total_loss = 0.0
		for i, (samples, labels) in enumerate(train_dataloader):
			samples, labels = Variable(torch.unsqueeze(samples, 1)), Variable(torch.squeeze(labels))
			optimizer.zero_grad()

			outputs = net(samples)
			outputs = torch.squeeze(outputs)

			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			total_loss += loss.item()
		losses.append(total_loss)
		print('[%2d] loss: %.3f' % (epoch + 1, total_loss))

		net.eval()
		accuracy = validate(net, test_dataloader)
		accuracies.append(accuracy)
		net.train()

	print('Losses:')
	print(losses)
	print('Accuracies:')
	print(accuracies)

	net.eval()
	validate(net, test_dataloader, compute_confusion=True)


def validate(net, test_dataloader, compute_confusion=False):
	total_correct = 0
	total_predicted = 0
	confusion_matrix = np.zeros((len(classifications), len(classifications)))
	for i, (samples, labels) in enumerate(test_dataloader):
		samples, labels = Variable(torch.unsqueeze(samples, 1)), Variable(torch.squeeze(labels))

		outputs = net(samples)
		outputs = torch.squeeze(outputs)

		_, predicted = torch.max(outputs, 1)
		total_predicted += predicted.size(0)
		total_correct += (predicted == labels).sum().item()

		if compute_confusion:
			for j in range(predicted.size(0)):
				confusion_matrix[labels[j]][predicted[j]] += 1

	if compute_confusion:
		print('confusion_matrix:')
		print(confusion_matrix)

	return total_correct / total_predicted

def main():
	parser = argparse.ArgumentParser(description='Classifier for training and inference')
	parser.add_argument('--input', help='The input file prefix for the preprocessed data')
	args = parser.parse_args()

	train(args.input + '.features', args.input + '.labels')


if __name__ == '__main__':
	main()
