import argparse

import pandas as pd


class Splitter(object):
	def __init__(self, input_file, output_file, minimum_samples=10, train_proportion=0.8):
		self.input_file = input_file
		self.output_file = output_file
		self.minimum_samples = minimum_samples
		self.train_proportion = train_proportion

	def run(self):
		features = pd.read_csv(self.input_file + '.features', header=None)
		labels = pd.read_csv(self.input_file + '.labels', header=None, names=['donor', 'label'])

		samples = {}

		for i, row in labels.iterrows():
			label = row['label']
			if label not in samples:
				samples[label] = []
			samples[label].append(i)

		train_samples, test_samples = self._split_samples(samples)
		self._write_output(features, labels, train_samples, test_samples)

	def _split_samples(self, samples):
		train_samples = []
		test_samples = []

		for label in samples:
			if len(samples[label]) < self.minimum_samples:
				print(f'Dropping {label} due to lack of samples')
				continue
			split_index = int(self.train_proportion * len(samples[label]))
			train_samples.extend(samples[label][: split_index])
			test_samples.extend(samples[label][split_index :])

		return train_samples, test_samples

	def _write_output(self, features, labels, train_samples, test_samples):
		train_features = features.iloc[train_samples]
		test_features = features.iloc[test_samples]
		train_labels = labels.iloc[train_samples]
		test_labels = labels.iloc[test_samples]

		with open(self.output_file + '.features.train', 'w') as train_feature_file:
			train_feature_file.write(train_features.to_csv(header=False, index=False))
		with open(self.output_file + '.features.test', 'w') as test_feature_file:
			test_feature_file.write(test_features.to_csv(header=False, index=False))
		with open(self.output_file + '.labels.train', 'w') as train_label_file:
			train_label_file.write(train_labels.to_csv(header=False, index=False))
		with open(self.output_file + '.labels.test', 'w') as test_label_file:
			test_label_file.write(test_labels.to_csv(header=False, index=False))


def main():
	parser = argparse.ArgumentParser(description='Split preprocessed data for training and inference')
	parser.add_argument('--input', help='The preprocessed input file prefix')
	parser.add_argument('--output', help='The output file destination')
	args = parser.parse_args()

	splitter = Splitter(args.input, args.output)
	splitter.run()


if __name__ == '__main__':
	main()
