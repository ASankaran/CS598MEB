import argparse

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


class MutationNet(nn.Module):
	def __init__(self):
		super().__init__()
		self.fc1 = nn.Linear(5750, 25)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		return x


def train(feature_file, label_file, batch_size=4, epochs=2):
	dataset = MutationDataset(feature_file, label_file)
	train_dataset, test_dataset = data.random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])
	train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
	test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

	net = MutationNet()
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

	for epoch in range(epochs):
		for i, (samples, labels) in enumerate(train_dataloader):
			samples, labels = Variable(samples), Variable(torch.squeeze(labels))
			optimizer.zero_grad()

			outputs = net(samples)

			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			print('[%d, %4d] loss: %.3f' % (epoch + 1, i + 1, loss.item()))

	total_loss = 0.0
	total_correct = 0
	total_predicted = 0
	for i, (samples, labels) in enumerate(test_dataloader):
		samples, labels = Variable(samples), Variable(torch.squeeze(labels))

		outputs = net(samples)

		loss = criterion(outputs, labels)
		total_loss += loss.item()

		_, predicted = torch.max(outputs, 1)
		total_predicted += predicted.size(0)
		total_correct += (predicted == labels).sum().item()

	print('validation loss: %.3f' % (loss.item()))
	print('validation accuracy: %.3f' % (total_correct / total_predicted))

def main():
	parser = argparse.ArgumentParser(description='Classifier for training and inference')
	parser.add_argument('--input', help='The input file prefix for the preprocessed data')
	args = parser.parse_args()

	train(args.input + '.features', args.input + '.labels')


if __name__ == '__main__':
	main()
