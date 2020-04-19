import math
import argparse

import numpy as np
import pandas as pd
from PIL import Image


class Vizualizer(object):
	def __init__(self, input_file, output_file):
		self.input_file = input_file
		self.output_file = output_file

	def run(self):
		features = pd.read_csv(self.input_file + '.features', header=None)
		labels = pd.read_csv(self.input_file + '.labels', header=None, names=['donor', 'label'])

		samples = {}
		counts = {}

		for (i, feature_row), (j, label_row) in zip(features.iterrows(), labels.iterrows()):
			assert i == j
			label = label_row['label']
			if label not in samples:
				samples[label] = np.zeros((23, 250))
				counts[label] = 0
			samples[label] += np.array(feature_row).reshape((23, 250))
			counts[label] += 1

		averaged = {label: samples[label] / counts[label] for label in samples}

		overall_max = 0.0

		for label in averaged:
			sample_max = np.amax(averaged[label])
			if sample_max > overall_max:
				overall_max = sample_max

		greyscaled = {label: np.sqrt(np.sqrt(averaged[label] / overall_max)) * 255.0 for label in averaged}

		for label in greyscaled:
			image = Image.fromarray(greyscaled[label].astype(np.uint8))
			image.save(f'{self.output_file}/{label}.png', 'png')

		greyscaled_unique = {}

		for current_label in greyscaled:
			other_averaged = np.zeros((23, 250))
			for other_label in greyscaled:
				if current_label == other_label:
					continue
				other_averaged += greyscaled[other_label]
			other_averaged /= (len(greyscaled) - 1)
			greyscaled_unique[current_label] = np.maximum(greyscaled[current_label] - other_averaged, 0)

		greyscaled_unique = {label: np.sqrt(np.sqrt(greyscaled_unique[label] / 255.0)) * 255.0 for label in greyscaled_unique}

		for label in greyscaled_unique:
			image = Image.fromarray(greyscaled_unique[label].astype(np.uint8))
			image.save(f'{self.output_file}/{label}_unique.png', 'png')


def main():
	parser = argparse.ArgumentParser(description='Visualize preprocessed data for training and inference')
	parser.add_argument('--input', help='The preprocessed input file prefix')
	parser.add_argument('--output', help='The output file destination')
	args = parser.parse_args()

	splitter = Vizualizer(args.input, args.output)
	splitter.run()


if __name__ == '__main__':
	main()
