import torch.nn as nn
import torch.nn.functional as F

from common import classifications


class ConvMutationNet(nn.Module):
	def __init__(self):
		super().__init__()
		self.cv01 = nn.Conv1d(23, 32, kernel_size=5)
		self.cv02 = nn.Conv1d(32, 32, kernel_size=5)
		self.cv03 = nn.Conv1d(32, 64, kernel_size=5)
		self.cv04 = nn.Conv1d(64, 64, kernel_size=5)
		self.cv05 = nn.Conv1d(64, 128, kernel_size=5)
		self.cv06 = nn.Conv1d(128, 128, kernel_size=5)
		self.pool = nn.MaxPool1d(2, 2)
		self.fc01 = nn.Linear(128 * 24, len(classifications))

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
		x = x.view(-1, 128 * 24)
		x = self.fc01(x)
		return x