import torch
from torch import nn
from models.model_struct import model_cfg
from models.vgg5.vgg5 import VGG5
import socket
import time
import config
from network_utils import send_data, receive_data
from utils import get_client_app_port_by_name
import logging

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_model(layer_type, in_channels, out_channels, kernel_size, cumulative_layer_number):
    """
     获取当前节点需要计算的模型层

     参数:
     model (nn.Module): 模型
     type (str): 层类型('M':池化层, 'D':全连接层, 'C':卷积层)
     in_channels (int): 输入通道数
     out_channels (int): 输出通道数
     kernel_size (int): 卷积核大小
     start_layer (int): 起始层索引

     返回值:
     feature_seq (nn.Sequential): 卷积层和池化层
     dense_s (nn.Sequential): 全连接层
     next_layer (int): 下一层索引
     """
    # 存储特征层（卷积层或池化层）的序列
    feature_seq = []
    # 存储全连接层的序列
    dense_seq = []
    # 根据层的类型添加对应的层到序列中
    if layer_type == "M":
        # 如果是池化层，增加到特征层序列中
        feature_seq.append(model.features[cumulative_layer_number])
        cumulative_layer_number += 1
    elif layer_type == "D":
        # 如果是全连接层，增加到全连接层序列中
        dense_seq.append(model.denses[cumulative_layer_number - 11])
        cumulative_layer_number += 1
    elif layer_type == "C":
        # 如果是卷积层，增加连续三个卷积层到特征层序列中
        for _ in range(3):
            feature_seq.append(model.features[cumulative_layer_number])
            cumulative_layer_number += 1
    # 创建特征层和全连接层的 Sequential 容器
    return nn.Sequential(*feature_seq), nn.Sequential(*dense_seq), cumulative_layer_number


def calculate_output(node_layer_indices, data, cumulative_layer_number):
    """
    计算当前节点的输出

    参数:
    model (nn.Module): 模型
    data (Tensor): 输入数据
    start_layer (int): 起始层索引

    返回值:
    data (Tensor): 输出数据
    cumulative_layer_number: 累计层数
    """
    # 遍历当前主机节点上的层
    for index in node_layer_indices[client_name]:
        # 如果节点上的层不相邻，需要实现层之间的兼容性
        layer_type = model_cfg[model_name][index][0]  # 层的类型
        in_channels = model_cfg[model_name][index][1]  # 输入通道数
        out_channels = model_cfg[model_name][index][2]  # 输出通道数
        kernel_size = model_cfg[model_name][index][3]  # 卷积核大小

        # 获取模型的当前层
        features, dense, cumulative_layer_number = get_model(
            layer_type, in_channels, out_channels, kernel_size, cumulative_layer_number
        )

        # 选择特征层还是全连接层
        model_layer = features if len(features) > 0 else dense

        # 如果是全连接层，需要先将数据展平
        if layer_type == "D":
            data = data.view(data.size(0), -1)

        # 将数据通过模型层进行前向传播
        data = model_layer(data)
    return data, cumulative_layer_number


def node_inference(node_indices, data_list, cumulative_layer_number):
    """
    开始当前节点的推理

    参数:
    node_indices (list): 当前节点的层索引
    data_list (list): 数据列表
    cumulative_layer_number (int): 累计层数

    返回值:
    list: 推理结果列表
    int: 累计层数
    """
    logging.info("*********************开始推理************************")
    # 迭代处理每一批数据
    result_list = []
    start_layer = cumulative_layer_number
    for i in range(config.iterations):
        data = data_list[i]
        # 获取推理后的结果
        result, cumulative_layer_number = calculate_output(node_indices, data, start_layer)
        result_list.append(result)

    return result_list, cumulative_layer_number


def client(name, client_port=None):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 如果指定了客户端端口，则绑定到该端口
    if client_port:
        try:
            client_socket.bind(('', client_port))  # 绑定客户端的端口
            logging.info(f"绑定到本地端口 {client_port}")
        except socket.error as e:
            logging.error(f"绑定到端口 {client_port} 失败: {e}")
            return

    # 连接到服务器
    try:
        client_socket.connect(('localhost', 9000))
        logging.info(f"作为 {name} 连接到服务器")
    except socket.error as e:
        logging.error(f"连接到服务器失败: {e}")
        return

    send_data(client_socket, name)

    while True:
        data = receive_data(client_socket)
        if data:
            node_indices, data_list, cumulative_layer_number = data
            logging.info(f"{name} 收到数据: {node_indices}, {data_list}")
            start_time = time.time()
            processed_data_list, processed_cumulative_layer_number = node_inference(node_indices,
                                                                                    data_list,
                                                                                    cumulative_layer_number)
            end_time = time.time()
            process_time = end_time - start_time  # 修改计算时间的顺序
            response = [process_time, processed_data_list, processed_cumulative_layer_number]
            send_data(client_socket, response)
        else:
            break  # 如果没有数据，可能连接已关闭

    client_socket.close()


if __name__ == "__main__":
    client_name = 'client3'
    model_name = 'VGG5'
    host_ip, host_port = get_client_app_port_by_name(client_name, model_name)

    # 初始化模型并载入预训练权重
    model = VGG5("Client", model_name, len(model_cfg[model_name]) - 1, model_cfg)
    model.eval()
    model.load_state_dict(torch.load("../../models/vgg5/vgg5.pth"))

    # 调用客户端函数，并指定客户端端口号
    client(client_name, client_port=host_port)
