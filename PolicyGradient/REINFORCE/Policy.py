import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Policy(nn.Module):
    def __init__(self, lr, gamma):
        super(Policy, self).__init__()
        self.gamma = gamma
        self.data = []

        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 2)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=0)
        return x

    def put_data(self, item):
        self.data.append(item)

    def train_net(self):
        g = 0
        self.optimizer.zero_grad()
        for r, prob_a in self.data[::-1]:
            g = r + self.gamma * g
            loss = - g * torch.log(prob_a)

            loss.backward()
        self.optimizer.step()
        self.data = []