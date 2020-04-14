import torch.nn as nn


class RecMutationNet(nn.Module):
	def __init__(self):
		super().__init__()
		self.rnn = nn.RNN(5750, 512, 3, batch_first=True)
		self.fc = nn.Linear(512, len(classifications))

	def forward(self, x):
		x, _ = self.rnn(x, torch.zeros(3, x.size(0), 512))
		x = x.contiguous().view(-1, 512)
		x = self.fc(x)
		return x