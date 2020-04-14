import torch.nn as nn

from common import classifications


class DenseMutationNet(nn.Module):
	def __init__(self):
		super().__init__()
		self.fc01 = nn.Linear(5750, 512)
		self.fc02 = nn.Linear(512, 512)
		self.fc03 = nn.Linear(512, len(classifications))
		self.drop = nn.Dropout(0.5)

	def forward(self, x):
		x = self.fc01(x)
		x = self.drop(x)
		x = self.fc02(x)
		x = self.drop(x)
		x = self.fc03(x)
		return x