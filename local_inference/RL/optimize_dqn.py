import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque


# 定义带有卷积层的DuelingDQN网络
class DuelingDQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DuelingDQN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=2, stride=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=2, stride=1)
        self.fc1 = nn.Linear(64 * (state_size - 2), 128)
        self.fc2 = nn.Linear(128, 64)
        self.value_stream = nn.Linear(64, 1)
        self.advantage_stream = nn.Linear(64, action_size)

    def forward(self, x):
        x = x.unsqueeze(1)  # 增加通道维度
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # 展平
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))


# DQN智能体定义
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.99  # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_decay = 0.995  # 探索率衰减
        self.epsilon_min = 0.01  # 最小探索率
        self.learning_rate = 0.001  # 学习率
        self.model = DuelingDQN(state_size, action_size)
        self.target_model = DuelingDQN(state_size, action_size)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        act_values = self.model(state)
        return torch.argmax(act_values[0]).item()

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).unsqueeze(0)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            target = self.model(state).clone()
            if done:
                target[0][action] = reward
            else:
                t = self.target_model(next_state).detach()
                target[0][action] = reward + self.gamma * torch.max(t)
            output = self.model(state)
            loss = self.criterion(output, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)
