import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sn

from common import classifications

from dense import DenseMutationNet
from cnn import ConvMutationNet
from rnn import RecMutationNet
from res import ResMutationNet


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


def generate_loss_accuracy_plot(output_file, losses, accuracies):
	losses = np.array(losses)
	accuracies = np.array(accuracies)
	epochs = np.array(list(range(0, len(losses))))

	data = pd.DataFrame({
		'Loss': losses,
		'Accuracy': accuracies,
		'Epoch': epochs
	})

	fig = plt.figure()
	ax = fig.gca()

	sn.lineplot('Epoch', 'Loss', data=data, color='tab:red', ax=ax)
	ax = ax.twinx()
	sn.lineplot('Epoch', 'Accuracy', data=data, color='tab:blue', ax=ax)

	fig.tight_layout()
	fig.savefig(f'{output_file}/loss_accurracy.svg')


def generate_confusion_plot(output_file, confusion_matrix):
	confusion_matrix = pd.DataFrame(
		confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis],
		columns=np.array(classifications),
		index=np.array(classifications)
	)
	confusion_matrix.index.name = 'Actual'
	confusion_matrix.columns.name = 'Predicted'

	fig = plt.figure(figsize=(8, 8))
	ax = fig.gca()

	sn.heatmap(confusion_matrix, ax=ax, cmap='Blues', annot=True, annot_kws={'size': 8})

	fig.tight_layout()
	fig.savefig(f'{output_file}/confusion.svg')


def train(net, feature_file, label_file, batch_size=32, epochs=50):
	train_dataset = MutationDataset(feature_file + '.train', label_file + '.train')
	test_dataset = MutationDataset(feature_file + '.test', label_file + '.test')

	train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
	test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

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
		accuracy, _ = validate(net, test_dataloader)
		accuracies.append(accuracy)
		net.train()

	net.eval()
	accuracy, confusion_matrix = validate(net, test_dataloader)
	print(f'Accuracy: {accuracy}')

	return losses, accuracies, confusion_matrix


def validate(net, test_dataloader):
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

		for j in range(predicted.size(0)):
			confusion_matrix[labels[j]][predicted[j]] += 1

	return total_correct / total_predicted, confusion_matrix

def main():
	parser = argparse.ArgumentParser(description='Classifier for training and inference')
	parser.add_argument('--input', help='The input file prefix for the preprocessed data')
	parser.add_argument('--output', help='The output file destination')
	args = parser.parse_args()

	# losses, accuracies, confusion_matrix = train(DenseMutationNet(), args.input + '.features', args.input + '.labels')
	# generate_loss_accuracy_plot(f'{args.output}/dense', losses, accuracies)
	# generate_confusion_plot(f'{args.output}/dense', confusion_matrix)

	# losses, accuracies, confusion_matrix = train(ConvMutationNet(), args.input + '.features', args.input + '.labels')
	# generate_loss_accuracy_plot(f'{args.output}/cnn', losses, accuracies)
	# generate_confusion_plot(f'{args.output}/cnn', confusion_matrix)

	# losses, accuracies, confusion_matrix = train(RecMutationNet(), args.input + '.features', args.input + '.labels')
	# generate_loss_accuracy_plot(f'{args.output}/rnn', losses, accuracies)
	# generate_confusion_plot(f'{args.output}/rnn', confusion_matrix)

	losses, accuracies, confusion_matrix = train(ResMutationNet(), args.input + '.features', args.input + '.labels')
	generate_loss_accuracy_plot(f'{args.output}/res', losses, accuracies)
	generate_confusion_plot(f'{args.output}/res', confusion_matrix)


if __name__ == '__main__':
	main()
