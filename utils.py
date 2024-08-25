import argparse  # 导入命令行参数解析器
import torch
import torch.nn.init as init  # 导入PyTorch神经网络参数初始化模块
import torchvision  # 导入PyTorch视觉库
import torchvision.transforms as transforms  # 导入PyTorch数据转换模块
from torch.utils.data import DataLoader, Subset  # 从PyTorch数据加载模块中导入数据加载器和子集工具

import pickle, struct, socket  # 导入pickle、struct和socket模块（用于消息传递）
from models.vgg5.vgg5 import *  # 从vgg模块中导入所有内容
from config import *  # 从config模块中导入所有内容
import collections  # 导入集合模块
import numpy as np  # 导入NumPy库

import logging  # 导入日志模块


def get_client_app_port(ip_address, model_name):
    """
    根据客户端的 ip地址 和 配置的模型名称 获取部署的模型的应用的端口
    :param ip_address: 客户端的ip地址
    :param model_name: 模型名称
    :return:
    """
    # 在服务器列表中查找对应的IP地址
    for server in server_list:
        if server['ip'] == ip_address:
            # 查找应用程序名称对应的端口
            return server['application'].get(model_name)
    # 如果没有找到IP地址或应用程序名称，返回None
    return None


def get_client_app_port_by_name(name, model_name):
    """
    根据客户端的 ip地址 和 配置的模型名称 获取部署的模型的应用的端口
    :param name: 客户端的名称
    :param model_name: 模型名称
    :return:
    """
    # 在服务器列表中查找对应的IP地址
    for server in local_server_list:
        if server['name'] == name:
            # 查找应用程序名称对应的端口
            return server['ip'], server['application'].get(model_name)
    # 如果没有找到IP地址或应用程序名称，返回None
    return None


def get_ip_by_hostname(hostname):
    """
    根据提供的主机名获取对应的 IP 地址。

    参数:
    - hostname: 要查找的主机名。

    返回:
    - 与主机名对应的 IP 地址。如果没有找到主机名，则返回 None。
    """
    for server in server_list:
        if server['hostname'] == hostname:
            return server['ip']
    return None


def get_index_by_ip(ip):
    """
    根据提供的 IP 地址获取在服务器列表中的索引（序号）。

    参数:
    - ip: 要查找的 IP 地址。

    返回:
    - 该 IP 地址在列表中的索引。如果没有找到 IP 地址，则返回 None。
    """
    for index, server in enumerate(server_list):
        if server['ip'] == ip:
            return index
    return None


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')  # 设置日志格式
logger = logging.getLogger(__name__)  # 获取logger对象

# 设置随机种子以确保结果的可重复性
np.random.seed(0)
torch.manual_seed(0)


def get_local_dataloader(CLIENT_IDEX, cpu_count):
    """
	定义获取本地数据加载器的函数
	:param CLIENT_IDEX:
	:param cpu_count:
	:return:
	"""
    # 获取数据集索引
    indices = list(range(N))
    # 根据客户端索引分割数据集索引
    part_tr = indices[int((N / K) * CLIENT_IDEX): int((N / K) * (CLIENT_IDEX + 1))]

    # 定义训练数据的转换操作
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 随机裁剪
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # 标准化
    ])

    # 加载CIFAR-10数据集
    trainset = torchvision.datasets.CIFAR10(
        root=dataset_path,
        train=True,
        download=False,
        transform=transform_train
    )
    # 获取当前客户端的数据子集
    subset = Subset(trainset, part_tr)
    # 创建数据加载器 数据子集、批大小、打乱、工作线程数
    trainloader = DataLoader(
        subset,
        batch_size=B,
        shuffle=True,
        num_workers=cpu_count
    )

    # CIFAR10 数据集的类别
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # 返回数据加载器和类别
    return trainloader, classes


def get_model(location, model_name, layer, device, cfg):
    """
	定义获取模型的函数
	:param location:
	:param model_name:
	:param layer:
	:param device:
	:param cfg:
	:return:
	"""
    # 复制配置
    cfg = cfg.copy()
    # 创建VGG5模型
    net = VGG5(location, model_name, layer, cfg)
    # 将模型移动到设备上
    net = net.to(device)
    # 打印模型结构
    logger.debug(str(net))
    return net


def send_msg(sock, msg):
    """
	发送消息
	:param sock:
	:param msg:
	:return:
	"""
    # 将消息序列化
    msg_pickle = pickle.dumps(msg)
    # 发送消息长度
    sock.sendall(struct.pack(">I", len(msg_pickle)))
    # 发送消息
    sock.sendall(msg_pickle)
    logger.debug(msg[0] + 'sent to' + str(sock.getpeername()[0]) + ':' + str(sock.getpeername()[1]))


def recv_msg(sock, expect_msg_type=None):
    """
	接收消息
	:param sock:
	:param expect_msg_type:
	:return:
	"""
    # 接收消息长度
    msg_len = struct.unpack(">I", sock.recv(4))[0]
    # 接收整个消息
    msg = sock.recv(msg_len, socket.MSG_WAITALL)
    # 反序列化消息
    msg = pickle.loads(msg)
    logger.debug(msg[0] + 'received from' + str(sock.getpeername()[0]) + ':' + str(sock.getpeername()[1]))

    # 如果期望的消息类型不为空，并且接收到的消息类型不匹配，则抛出异常
    if (expect_msg_type is not None) and (msg[0] != expect_msg_type):
        raise Exception("Expected " + expect_msg_type + " but received " + msg[0])
    return msg  # 返回消息


def split_weights_client(weights, cweights):
    """
	定义分割客户端权重的函数
	:param weights:
	:param cweights:
	:return:
	"""
    for key in cweights:
        assert cweights[key].size() == weights[key].size()  # 断言权重大小相同
        cweights[key] = weights[key]  # 从整体权重中分割出客户端权重
    return cweights


def split_weights_server(weights, cweights, sweights):
    """
	定义分割服务器权重的函数
	:param weights:
	:param cweights:
	:param sweights:
	:return:
	"""
    # 客户端权重的键
    ckeys = list(cweights)
    # 服务器权重的键
    skeys = list(sweights)
    # 整体权重的键
    keys = list(weights)

    for i in range(len(skeys)):
        assert sweights[skeys[i]].size() == weights[keys[i + len(ckeys)]].size()  # 断言权重大小相同
        sweights[skeys[i]] = weights[keys[i + len(ckeys)]]  # 从整体权重中分割出服务器权重

    return sweights


def concat_weights(weights, cweights, sweights):
    """
	定义合并权重的函数
	:param weights:
	:param cweights:
	:param sweights:
	:return:
	"""
    # 使用有序字典来保持权重的顺序
    concat_dict = collections.OrderedDict()

    # 客户端权重的键
    ckeys = list(cweights)
    # 服务器权重的键
    skeys = list(sweights)
    # 整体权重的键
    keys = list(weights)

    for i in range(len(ckeys)):
        concat_dict[keys[i]] = cweights[ckeys[i]]  # 合并客户端权重

    for i in range(len(skeys)):
        concat_dict[keys[i + len(ckeys)]] = sweights[skeys[i]]  # 合并服务器权重

    return concat_dict


def zero_init(net):
    """
	 定义零初始化的函数
	:param net:
	:return:
	"""
    # 遍历网络中的所有模块
    for m in net.modules():
        # 如果是卷积层
        if isinstance(m, nn.Conv2d):
            # 权重初始化为零
            init.zeros_(m.weight)
            # 如果偏置不是None
            if m.bias is not None:
                # 偏置初始化为零
                init.zeros_(m.bias)
        # 如果是批量归一化层
        elif isinstance(m, nn.BatchNorm2d):
            init.zeros_(m.weight)
            init.zeros_(m.bias)
            init.zeros_(m.running_mean)
            init.zeros_(m.running_var)
        # 如果是全连接层
        elif isinstance(m, nn.Linear):
            # 权重初始化为零
            init.zeros_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)  # 偏置初始化为零
    return net


def fed_avg(zero_model, w_local_list, totoal_data_size):
    """
	定义 FedAvg 聚合的函数
	:param zero_model:
	:param w_local_list:
	:param totoal_data_size:
	:return:
	"""
    # 获取权重键的列表
    keys = w_local_list[0][0].keys()

    for k in keys:
        # 对于每个本地权重
        for w in w_local_list:
            # 计算数据量的比重
            beta = float(w[1]) / float(totoal_data_size)
            # 如果是批次统计项
            if 'num_batches_tracked' in k:
                zero_model[k] = w[0][k]  # 直接赋值
            else:
                zero_model[k] += (w[0][k] * beta)  # 按比例累加权重

    return zero_model


def norm_list(alist):
    """
	定义归一化列表的函数
	:param alist:
	:return:
	"""
    return [l / sum(alist) for l in alist]


def str2bool(v):
    """
	 定义将字符串转换为布尔值的函数
	:param v:
	:return:
	"""
    # 如果已经是布尔值，直接返回
    if isinstance(v, bool):
        return v
    # 如果是字符串表示的真值
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    # 如果是字符串表示的假值
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    # 如果不是预期的字符串
    else:
        # 否则抛出异常
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_total_size(obj, seen=None):
    """
    递归计算对象及其引用对象的总内存大小
    :param obj:
    :param seen:
    :return:
    """
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_total_size(v, seen) for v in obj.values()])
        size += sum([get_total_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_total_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_total_size(i, seen) for i in obj])
    return size


def calculate_bandwidth_kbps(bytes_received, transmission_time):
    """
    根据接收的字节数和传输时间计算带宽（kb/s）
    :param bytes_received: 接收的字节数
    :param transmission_time: 传输时间（秒）
    :return: 带宽（kb/s）
    """
    bits_received = bytes_received * 8  # 将字节转换为位
    bandwidth_bps = bits_received / transmission_time  # 计算带宽（位/秒）
    bandwidth_kbps = bandwidth_bps / 1000  # 将带宽转换为千位/秒
    return bandwidth_kbps
