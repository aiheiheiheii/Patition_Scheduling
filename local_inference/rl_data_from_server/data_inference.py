import torch
from torch import nn
from models.model_struct import model_cfg
from models.vgg5.vgg5 import VGG5
from config import iterations
import logging
import torch.nn.functional as F


# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def calculate_accuracy(fx, y):
    """
    计算模型输出与真实标签之间的准确率

    参数:
    fx (Tensor): 模型输出
    y (Tensor): 真实标签

    返回值:
    acc (float): 准确率(0-100)
    """
    # 计算预测值，fx是模型输出，y是真实标签
    predictions = fx.max(1, keepdim=True)[1]
    # 将预测值和真实标签转换为相同形状
    correct = predictions.eq(y.view_as(predictions)).sum()
    # 计算准确率，correct是预测正确的样本数量
    acc = 100.00 * correct.float() / predictions.shape[0]
    return acc


def get_loss_acc(result_list, target):
    """
    计算结果的loss和acc
    :param result_list:    客户端的推理结果
    :param target:         真实标签
    :return:              loss和acc
    """
    loss_list, acc_list = [], []
    for i in range(iterations):
        loss = F.cross_entropy(result_list[i], target[i]).item()
        acc = calculate_accuracy(result_list[i], target[i])
        loss_list.append(loss)
        acc_list.append(acc)
    avg_loss = sum(loss_list) / len(loss_list)
    avg_acc = sum(acc_list) / len(acc_list)
    logging.info(f"Average Loss: {avg_loss:.4f}")
    logging.info(f"Average Accuracy: {avg_acc:.4f}%")
    return avg_loss, avg_acc


def get_model(model, layer_type, in_channels, out_channels, kernel_size, cumulative_layer_number):
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


def calculate_output(model, model_name, layer_indices, data, cumulative_layer_number):
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
    for index in layer_indices:
        # 如果节点上的层不相邻，需要实现层之间的兼容性
        layer_type = model_cfg[model_name][index][0]  # 层的类型
        in_channels = model_cfg[model_name][index][1]  # 输入通道数
        out_channels = model_cfg[model_name][index][2]  # 输出通道数
        kernel_size = model_cfg[model_name][index][3]  # 卷积核大小

        # 获取模型的当前层
        features, dense, cumulative_layer_number = get_model(
            model, layer_type, in_channels, out_channels, kernel_size, cumulative_layer_number
        )

        # 选择特征层还是全连接层
        model_layer = features if len(features) > 0 else dense

        # 如果是全连接层，需要先将数据展平
        if layer_type == "D":
            data = data.view(data.size(0), -1)

        # 将数据通过模型层进行前向传播
        data = model_layer(data)
    return data, cumulative_layer_number


def node_inference(model, model_name, node_indices, data_list, cumulative_layer_number):
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
    for i in range(iterations):
        data = data_list[i]
        # 获取推理后的结果
        result, cumulative_layer_number = calculate_output(model, model_name, node_indices, data, start_layer)
        result_list.append(result)

    return result_list, cumulative_layer_number


def data_inference(data_list, node_indices, cumulative_layer_number):
    """
    开始数据推理

    参数:
    data_list (list): 数据列表
    node_indices (list): 节点层索引列表
    cumulative_layer_number (int): 累计层数


    返回值:
    list: 推理结果列表
    int: 累计层数
    """

    model_name = 'VGG5'
    # 初始化模型并载入预训练权重
    model = VGG5("Server", model_name, len(model_cfg[model_name]) - 1, model_cfg)
    model.eval()
    model.load_state_dict(torch.load("../../models/vgg5/vgg5.pth"))

    # 开始推理
    result_list, cumulative_layer_number = node_inference(model, model_name, node_indices, data_list,
                                                          cumulative_layer_number)

    return result_list, cumulative_layer_number

