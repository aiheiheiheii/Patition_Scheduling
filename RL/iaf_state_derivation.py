import torch
import torch.nn as nn


class IAF(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(IAF, self).__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(state_dim, hidden_dim, batch_first=True)
        self.linear_mean = nn.Linear(hidden_dim, state_dim)
        self.linear_std = nn.Linear(hidden_dim, state_dim)

    def forward(self, state):
        # 将状态转换为PyTorch张量
        state = torch.FloatTensor(state).unsqueeze(0)

        # 通过LSTM层获取隐藏状态
        _, (hidden, _) = self.lstm(state)
        hidden = hidden.squeeze(0)

        # 计算衍生状态的均值和标准差
        mean = self.linear_mean(hidden)
        log_std = self.linear_std(hidden)
        std = torch.exp(log_std)

        # 从均值和标准差生成衍生状态
        derived_state = torch.normal(mean, std)

        return derived_state.detach().numpy()


class IAFStateDerivation:
    def __init__(self, state_dim, hidden_dim):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.iaf = IAF(state_dim, hidden_dim)

    def derive_state(self, state):
        # 使用IAF生成衍生状态
        derived_state = self.iaf(state)

        return derived_state