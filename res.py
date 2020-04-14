import functools

import torch.nn as nn
import torch.nn.functional as F

from common import classifications


def conv(in_channels, out_channels):
	return nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2, bias=False)


class ResidualBlock(nn.Module):
	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.conv1 = conv(in_channels, out_channels)
		self.bn1 = nn.BatchNorm1d(out_channels)
		self.conv2 = conv(out_channels, out_channels)
		self.bn2 = nn.BatchNorm1d(out_channels)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = nn.Sequential(
			conv(in_channels, out_channels),
			nn.BatchNorm1d(out_channels)
		) if in_channels != out_channels else nn.Identity()
        
	def forward(self, x):
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		out = self.conv2(out)
		out = self.bn2(out)
		out += self.downsample(x)
		out = self.relu(out)
		return out


class ResMutationNet(nn.Module):
	def __init__(self):
		super().__init__()
		self.layer1 = self.make_layer(1, 2, 3)
		self.layer2 = self.make_layer(2, 4, 3)
		self.layer3 = self.make_layer(4, 8, 3)
		self.avg_pool = nn.AvgPool1d(2, 2)
		self.fc = nn.Linear(int(5750 / 2 * 8), len(classifications))

	def make_layer(self, in_channels, out_channels, blocks):
		layers = []
		layers.append(ResidualBlock(in_channels, out_channels))
		for i in range(1, blocks):
			layers.append(ResidualBlock(out_channels, out_channels))
		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.avg_pool(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)
		return x
