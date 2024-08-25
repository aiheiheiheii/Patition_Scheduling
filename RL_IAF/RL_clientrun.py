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

# 无限循环，用于持续接收服务器指令和执行任务
while True:
	# 接收重置标志
	reset_flag = rl_client.recv_message(rl_client.sock, 'RESET_FLAG')[1]
	# 如果需要重置
	if reset_flag:
		# 初始化RL_Client
		rl_client.initialize(len(model_cfg[config.model_name])-1)
	else:
		# 否则，接收分层信息
		logger.info('==> Next Timestep..')
		# 接收分层信息
		config.split_layer = rl_client.recv_message(rl_client.sock, 'SPLIT_LAYERS')[1]

		rl_client.reinitialize(config.split_layer[index])

	# 根据接收到的分层信息重新初始化RL_Client
	# 记录开始训练日志
	logger.info('==\u003e Training Start..')
	# 如果是第一次训练
	if first:
		# 进行推理
		rl_client.infer(train_loader)
		# 再次进行推理
		rl_client.infer(train_loader)
		# 设置first标志为False
		first = False
	else:
		# 否则，只进行一次推理
		rl_client.infer(train_loader)
