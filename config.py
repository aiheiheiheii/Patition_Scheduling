import sys
"""
 配置相关
"""

# Network configration
SERVER_ADDR = '192.168.215.128'
SERVER_PORT = 51000


# 本地服务器配置
local_server_list = [
    {
        "name": "client1",
        "ip": "127.0.0.1",
        "username": "root",
        "password": "123456",
        "application": {
            "VGG5": 9001
        }
    },
    {
        "name": "client2",
        "ip": "127.0.0.1",
        "username": "root",
        "password": "123456",
        "application": {
            "VGG5": 9002
        }
    },
    {
        "name": "client3",
        "ip": "127.0.0.1",
        "username": "root",
        "password": "123456",
        "application": {
            "VGG5": 9003
        }
    }
]
# 线上部署
server_list = [
    {
        "ip": "192.168.215.133",
        "username": "root",
        "password": "123456",
        "hostname": "client1",
        "application": {
            "VGG5": 9001,
            "VGG6": 9002
        }
    },
    {
        "ip": "192.168.215.128",
        "username": "root",
        "password": "123456",
        "hostname": "client2",
        "application": {
            "VGG5": 9001,
            "VGG6": 9002
        }
    },
    {
        "ip": "192.168.215.135",
        "username": "root",
        "password": "123456",
        "hostname": "client3",
        "application": {
            "VGG5": 9001,
            "VGG6": 9002
        }
    },
    {
        "ip": "192.168.215.137",
        "username": "root",
        "password": "123456",
        "hostname": "client4",
        "application": {
            "VGG5": 9001,
            "VGG6": 9002
        }
    }
    # 添加更多服务器
]
CLIENTS_LIST = [server["ip"] for server in server_list]
dataset_config = {
    'VGG5': "vgg5",
    'VGG6': ""
}
# Dataset configration
home = sys.path[0].split('SynerGist')[0] + 'SynerGist'
dataset_path = home + '/dataset/vgg5/'
# data length
N = 10000
# Batch size
B = 256
# 迭代次数 N为数据总数，B为批次大小
iterations = int(N / B)
# Number of devices
K = len(server_list)
# Number of groups
G = 3
model_name = 'VGG5'
model_size = 1.28
model_flops = 32.902
total_flops = 8488192
# Initial split layers
split_layer = [6, 6, 6]
model_len = 7


# RL training configration
LR = 0.01                  # Learning rate
max_episodes = 100         # max training episodes
max_time_steps = 100       # max time steps in one episode
exploration_times = 20	   # exploration times without std decay
n_latent_var = 64          # number of variables in hidden layer
action_std = 0.5           # constant std for action distribution (Multivariate Normal)
update_timestep = 10       # update policy every n time steps
K_epochs = 50              # update policy for K epochs
eps_clip = 0.2             # clip parameter for PPO
rl_gamma = 0.9             # discount factor
rl_b = 100				   # Batch size
rl_lr = 0.0003             # parameters for Adam optimizer
rl_betas = (0.9, 0.999)

node_layer_indices = {'client1': [0, 1], 'client2': [2, 3], 'client3': [4, 5, 6]}

# infer times for each device
iteration = {server['ip']: 5 for server in server_list}

# 状态衍生的时间窗口大小
window_size = 10
buffer_size = 10

# IAF model configuration
hidden_dim = 64
iaf_lr = 0.001

random = True
random_seed = 0
