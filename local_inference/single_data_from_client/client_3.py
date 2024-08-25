import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import config
from models.model_struct import model_cfg
from models.vgg5.vgg5 import VGG5
from communication.communicator import NodeEnd
from utils import get_client_app_port_by_name


def get_next_client(current_client):
    """
    根据当前客户端获取下一个客户端
    :param current_client:
    :return:
    """
    keys = list(node_layer_indices.keys())
    try:
        current_client_index = keys.index(current_client)
        next_client_index = current_client_index + 1
        if next_client_index < len(keys):  # 确保不超出列表范围
            return keys[next_client_index]
        else:
            return None  # 当前键是最后一个键，没有下一个键
    except ValueError:
        return None  # 当前键不在字典中


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


def calculate_output(model, data, cumulative_layer_number):
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
    for index in node_layer_indices[current_client_name]:
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


def from_first(model, node):
    print("从第一层开始")
    data_dir = "../../dataset"
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
    test_loader = DataLoader(
        test_dataset, batch_size=256, shuffle=False, num_workers=4
    )

    # 创建一个新的连接来连接下一个客户端
    node.__init__(host_ip, host_port)
    start_layer = 0
    cumulative_layer_number = 0
    for data, target in test_loader:
        # 获取推理后的结果
        inferred_data, cumulative_layer_number = calculate_output(model, data, start_layer)
        data_cpu_list.append(inferred_data.cpu())
        target_list.append(target.cpu())
    # 获取下一个接收节点的地址，并建立通信
    next_client = get_next_client(current_client_name)
    # 如果不是最后一个客户端
    if next_client:
        next_ip, next_port = get_client_app_port_by_name(next_client, model_name)
        print("next_ip(%s), next_port= %s" % (next_ip, next_port))
        node.node_connect(next_ip, next_port)
        # 将处理后的数据发送给下一个客户端
        # 准备发送的消息内容，可能需要包含标签
        msg = [info, node_layer_indices, data_cpu_list, target_list, cumulative_layer_number]
        node.send_message(node.sock, msg)
        print(f"客户端{host_ip}:{host_port}将处理后的消息发送到客户端{next_ip}:{next_port}")

        # 关闭与下一个客户端的连接
        node.sock.close()
    # 如果是最后一个客户端，则计算结果，推理结束
    else:
        print("推理结束")


def node_inference(model, node, msg):
    print("持续推理")
    global info, node_layer_indices, data_cpu_list, target_list
    info, node_layer_indices, data_cpu_list, target_cpu_list, cumulative_layer_number = msg
    print(info)

    # 迭代次数 N为数据总数，B为批次大小
    iterations = int(config.N / config.B)
    # 迭代处理每一批数据
    result_list = []
    start_layer = cumulative_layer_number
    for i in range(iterations):
        data = data_cpu_list[i]
        # 获取推理后的结果
        result, cumulative_layer_number = calculate_output(model, data, start_layer)
        result_list.append(result)
    # 获取下一个客户端
    next_client = get_next_client(current_client_name)
    # 如果不是最后一个客户端
    if next_client:
        next_ip, next_port = get_client_app_port_by_name(next_client, model_name)
        print("next_ip(%s), next_port= %s" % (next_ip, next_port))

        # 创建一个新的连接来连接下一个客户端
        node.__init__(host_ip, host_port)
        node.node_connect(next_ip, next_port)

        # 将处理后的数据发送给下一个客户端
        msg = [info, node_layer_indices, result_list, target_cpu_list, cumulative_layer_number]
        node.send_message(node.sock, msg)
        print(f"客户端{host_ip}:{host_port}将处理后的消息发送到客户端{next_ip}:{next_port}")

        # 关闭与下一个客户端的连接
        node.sock.close()
    # 如果是最后一个客户端，则计算结果，推理结束
    else:
        for i in range(iterations):
            loss = F.cross_entropy(result_list[i], target_cpu_list[i])
            acc = calculate_accuracy(result_list[i], target_cpu_list[i])
            loss_list.append(loss)
            acc_list.append(acc)
        print("loss :{:.4}".format(sum(loss_list) / len(loss_list)))
        print("acc :{:.4}%".format(sum(acc_list) / len(acc_list)))
        print("")


def start():
    # 建立连接
    node = NodeEnd(host_ip, host_port)

    # 等待上一个客户端的连接
    connect_socket, _ = node.wait_for_connection()

    # 接收上一个客户端发送的数据
    msg = node.receive_message(connect_socket)
    print(f"客户端{host_ip}:{host_port}接收到消息: {msg}")
    # 关闭连接
    node.sock.close()

    # 初始化模型并载入预训练权重
    model = VGG5("Client", model_name, len(model_cfg[model_name]) - 1, model_cfg)
    model.eval()
    model.load_state_dict(torch.load("../../models/vgg5/vgg5.pth"))

    if len(msg) == 2:
        global info, node_layer_indices
        info, node_layer_indices = msg
        print(info)
        from_first(model, node)
    else:
        node_inference(model, node, msg)


if __name__ == '__main__':
    current_client_name = 'client3'
    model_name = 'VGG5'

    host_ip, host_port = get_client_app_port_by_name(current_client_name, model_name)
    print(host_ip, host_port)

    info = ''
    node_layer_indices = {}
    data_cpu_list, target_list = [], []
    # 损失列表
    loss_list = []
    # 准确率列表
    acc_list = []

    start()
