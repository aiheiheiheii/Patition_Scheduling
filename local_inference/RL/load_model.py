import torch
from dqn import DQN  # 确保正确导入 DQN 类
from models.model_struct import model_cfg
from resource_utilization import get_windows_resource_info


# 定义加载模型的函数
def load_model(model_path, state_size, action_size):
    model = DQN(state_size, action_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


# 定义根据资源状态进行推理的函数
def get_segmentation_point(model, state):
    state = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        action_values = model(state)
    return torch.argmax(action_values[0]).item()


# 定义获取当前资源状态的函数
def get_current_state():
    resource_state = get_windows_resource_info()
    print(resource_state)
    state = [
        float(resource_state["cpu"][:-1]) / 100,
        float(resource_state["memory"]["Usage"][:-1]) / 100,
        float(resource_state["network"]["bytes_recv_per_sec"].split()[0])
    ]
    print(state)
    return state


if __name__ == "__main__":
    model_path = "dqn_model.pth"
    state_size = 3
    action_size = len(model_cfg["VGG5"])  # 根据模型的配置更新

    # 加载模型
    model = load_model(model_path, state_size, action_size)

    # 获取当前资源状态
    current_state = get_current_state()

    # 使用模型进行推理，获取切分点
    segmentation_point = get_segmentation_point(model, current_state)

    print(f"Based on the current resource state, the optimal segmentation point is at layer {segmentation_point}.")
