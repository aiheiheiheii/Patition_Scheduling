import numpy as np
import gym
from gym import spaces
from resource_utilization import get_all_server_info
from config import node_layer_indices, B, dataset_path, iterations
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class FederatedEnv(gym.Env):
    def __init__(self):
        super(FederatedEnv, self).__init__()

        # 定义动作空间和状态空间
        self.action_space = spaces.Discrete(len(node_layer_indices))
        self.observation_space = spaces.Box(low=0, high=1, shape=(len(node_layer_indices), 4), dtype=np.float32)

        # 加载数据集
        self.data_list, self.target_list = self.prepare_data()

        # 初始化状态
        self.state = self.get_state()

    def prepare_data(self):
        data_dir = dataset_path
        test_dataset = datasets.CIFAR10(
            data_dir,
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            ),
            download=True
        )
        test_loader = DataLoader(test_dataset, batch_size=B, shuffle=False, num_workers=0)
        data_cpu_list, target_cpu_list = [], []
        for data, target in test_loader:
            data_cpu_list.append(data)
            target_cpu_list.append(target)
        return data_cpu_list, target_cpu_list

    def get_state(self):
        server_infos = get_all_server_info()
        state = []
        for server in server_infos.values():
            cpu = float(server['cpu'].split()[-1]) / 100.0
            gpu = float(server['gpu']) / 100.0 if 'gpu' in server else 0.0
            memory = float(server['memory'].split()[2][1:-1]) / 100.0
            network = float(server['network'].split()[1])
            state.append([cpu, gpu, memory, network])
        return np.array(state)

    def step(self, action):
        # 进行推理计算并计算执行时间
        client_name = list(node_layer_indices.keys())[action]
        # 模拟推理过程
        inference_time = np.random.rand()  # 此处用随机数模拟执行时间
        reward = -inference_time  # 奖励为负的执行时间，即时间越短奖励越高
        self.state = self.get_state()  # 更新状态
        done = True  # 单步任务结束
        return self.state, reward, done, {}

    def reset(self):
        self.state = self.get_state()
        return self.state

    def render(self, mode='human', close=False):
        pass
