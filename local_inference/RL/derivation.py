import torch
import torch.nn as nn


# 定义GRU状态衍生函数类
class GRUDerivationFunction(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUDerivationFunction, self).__init__()
        self.hidden_size = hidden_size

        # 初始化重置门的参数
        self.reset_gate = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Sigmoid()
        )

        # 初始化更新门的参数
        self.update_gate = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Sigmoid()
        )

        # 初始化候选状态的参数
        self.candidate_state = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Tanh()
        )

    def forward(self, s_t, s_ti):
        # 将原始状态和新状态拼接起来作为输入
        input_combined = torch.cat((s_t, s_ti), dim=-1)

        # 计算重置门
        reset_gate = self.reset_gate(input_combined)

        # 计算更新门
        update_gate = self.update_gate(input_combined)

        # 计算候选状态
        candidate_input = torch.cat((reset_gate * s_t, s_ti), dim=-1)
        candidate_state = self.candidate_state(candidate_input)

        # 计算最终的衍生状态
        derived_state = update_gate * s_t + (1 - update_gate) * candidate_state

        # 对衍生状态中的CPU占用率和内存占用率进行裁剪,确保数值在合理范围内
        derived_state[:, :2] = torch.clamp(derived_state[:, :2], min=0.0, max=1.0)

        return derived_state


# 设置状态维度和隐藏层维度
state_dim = 3
hidden_dim = 3

# 创建GRU状态衍生函数实例
gru_derivation = GRUDerivationFunction(input_size=state_dim, hidden_size=hidden_dim)

# 设置初始状态
initial_state = torch.tensor([[0.15, 0.30, 168.0]])  # CPU占用率15%,内存占用率30%,带宽168kb/s

# 与环境交互后得到的新状态列表
s_t_list = [
    torch.tensor([[0.2, 0.35, 200.0]]),
    torch.tensor([[0.18, 0.32, 190.0]]),
    torch.tensor([[0.22, 0.38, 210.0]]),
    torch.tensor([[0.16, 0.28, 180.0]]),
    torch.tensor([[0.25, 0.4, 220.0]]),
    torch.tensor([[0.2, 0.36, 200.0]]),
    torch.tensor([[0.18, 0.33, 195.0]])
]

# 存储衍生状态的列表
derived_states = []

# 进行状态衍生
for new_state in s_t_list:
    # 调用GRU状态衍生函数进行衍生
    derived_state = gru_derivation(initial_state, new_state)

    # 将衍生状态添加到列表中
    derived_states.append(derived_state)

# 打印衍生状态
for i, state in enumerate(derived_states):
    print(f"第{i + 1}次衍生状态:")
    print(f"CPU占用率: {state[0, 0].item():.2%}")
    print(f"内存占用率: {state[0, 1].item():.2%}")
    print(f"带宽: {state[0, 2].item():.3f} kb/s")
    print("---")