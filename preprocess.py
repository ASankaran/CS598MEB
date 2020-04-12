import argparse

import pandas as pd

class Preprocessor(object):
	def __init__(self, input_file, output_file, bucket_size=1000000, bucket_count=250, chromosomes=23, batch_size=10000):
		self.input_file = input_file
		self.output_file = output_file
		self.bucket_size = bucket_size
		self.bucket_count = bucket_count
		self.chromosomes = chromosomes
		self.batch_size = batch_size

		self.samples = {}

	def run(self):
		count = 0
		for chunk in pd.read_table(self.input_file, chunksize=self.batch_size):
			self._handle_chunk(chunk)
			count += self.batch_size
			print(f'Processed {count} rows')
		self._write_output()

	def _handle_chunk(self, chunk):
		for i, row in chunk.iterrows():
			self._maybe_init_sample(row['Donor_ID'], row['Project_Code'])
			self._update_sample(row['Donor_ID'], row['Project_Code'], row['Chromosome'], row['Start_position'])

	def _maybe_init_sample(self, identifier, label):
		if (identifier, label) in self.samples:
			return
		self.samples[(identifier, label)] = [0] * (self.chromosomes * self.bucket_count)

	def _update_sample(self, identifier, label, chromosome, location):
		if chromosome == 'X' or chromosome =='Y':
			chromosome = self.chromosomes
		chromosome = int(chromosome) - 1

		location = int(location)

		if chromosome >= self.chromosomes:
			print(f'Invalid chromosome: {chromosome}')
			return

		if int(location / self.bucket_size) >= self.bucket_count:
			print(f'Invalid location: {location}')
			return

		self.samples[(identifier, label)][(chromosome - 1) * self.bucket_count + int(location / self.bucket_size)] += 1

	def _write_output(self):
		with open(self.output_file + '.features', 'w') as feature_file:
			with open(self.output_file + '.labels', 'w') as label_file:
				for key, sample in self.samples.items():
					feature_line = ','.join([str(element) for element in sample])
					feature_file.write(feature_line)
					feature_file.write('\n')
					label_line = ','.join([str(element) for element in key])
					label_file.write(label_line)
					label_file.write('\n')


def main():
	parser = argparse.ArgumentParser(description='Preprocess data for training and inference')
	parser.add_argument('--input', help='The raw input file from PCAWG')
	parser.add_argument('--output', help='The output file destination')
	args = parser.parse_args()

	preprocessor = Preprocessor(args.input, args.output)
	preprocessor.run()


if __name__ == '__main__':
	main()
