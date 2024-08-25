# 导入PyTorch库，用于深度学习模型的构建和训练
import torch
# 导入socket模块，用于处理网络通信
import socket
# 导入多进程模块，用于并行处理任务
import multiprocessing
# 导入NumPy库，用于数学运算
import numpy as np

# 导入日志模块，用于记录日志信息
import logging
# 导入系统模块
import sys
# 添加上级目录到系统路径，以便能够导入同一项目中其他模块的类和函数
sys.path.append('../')
# 从RLEnv模块导入RL_Client类
from RLEnv import RL_Client
import config
import utils
from models.model_struct import model_cfg
# 设置日志格式，包括时间戳、记录器名称、日志级别和日志消息
logging.basicConfig(level=logging.INFO, format='%(asc_time)s - %(name)s - %(level_name)s - %(message)s')
# 获取logger对象，用于记录日志
logger = logging.getLogger(__name__)


# 如果配置中指定使用随机种子
if config.random:
	# 设置PyTorch随机种子，确保实验的可重复性
	torch.manual_seed(config.random_seed)
	# 设置NumPy随机种子
	np.random.seed(config.random_seed)
	# 记录随机种子
	logger.info('Random seed: {}'.format(config.random_seed))


# 初始化first标志，用于控制首次训练行为
first = True
# 获取本机IP地址
ip_address = utils.get_ip_by_hostname(socket.gethostname())
# 根据IP地址获取客户端索引
index = utils.get_index_by_ip(ip_address)
# 计算每个客户端的数据长度
data_len = config.N / config.K
# 获取模型分层索引
split_layer = config.split_layer[index]

logger.info('==> Preparing Data..')
# 获取CPU核心数
cpu_count = multiprocessing.cpu_count()
# 获取本地数据加载器和类别
train_loader, classes = utils.get_local_dataloader(index, cpu_count)

logger.info('==> Preparing RL_Client..')
# 创建RL_Client对象
rl_client = RL_Client(
	index,
	ip_address,
	config.SERVER_ADDR,
	config.SERVER_PORT,
	data_len,
	config.model_name,
	split_layer,
	model_cfg
)

# 第37行到第60行,修改为:
while True:
	# 接收分割点信息
    split_point = rl_client.recv_message(rl_client.sock, 'SPLIT_POINT')[1]
    rl_client.initialize(split_point)

    logger.info('===> Infer Start..')
    rl_client.infer(train_loader)

# 第63行到第85行,删除rl_client.reinitialize()函数的调用,并修改rl_client.initialize()函数的调用:
    logger.info('===> Infer Start..')
    if first:
        rl_client.infer(train_loader)
        rl_client.infer(train_loader)
        first = False
    else:
        rl_client.infer(train_loader)
