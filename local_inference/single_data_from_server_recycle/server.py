import socket
import time
import threading
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
from network_utils import send_data, receive_data
from config import B, dataset_path, iterations
from draw_graph import plot_times
from utils import get_total_size, calculate_bandwidth_kbps
import logging


# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_next_client(current_client):
    """
    获取下一个客户端
    :param current_client:  当前客户端
    :return:              下一个客户端
    """
    keys = list(node_layer_indices.keys())
    try:
        current_index = keys.index(current_client)
        next_index = current_index + 1
        if next_index < len(keys):
            return keys[next_index]
        return None
    except ValueError:
        return None


def prepare_data():
    """
    加载数据集并返回数据和标签
    :return:  数据集和标签
    """
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

    test_loader = DataLoader(
        test_dataset, batch_size=B, shuffle=False, num_workers=0
    )
    data_cpu_list, target_cpu_list = [], []
    for data, target in test_loader:
        data_cpu_list.append(data)
        target_cpu_list.append(target)
    return data_cpu_list, target_cpu_list


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


def handle_client(conn, client_name):
    """
    处理客户端的请求
    :param conn:          客户端的连接
    :param client_name:  客户端的名称
    :return:              None
    """
    for iteration in range(1, max_iterations + 1):
        logging.info(f"Start processing round {iteration}")

        start_time = time.time()
        message = receive_data(conn)
        end_time = time.time()
        transmit_time = end_time - start_time

        print(f"{client_name}: transmit time is {transmit_time} seconds.")
        # 记录客户端传输时间

        client_transmit_times[client_name].append(transmit_time)

        # 计算传输的数据量
        size = get_total_size(message)
        print(f"{client_name}: The total size of the list is {size} bytes.")
        bandwidth = calculate_bandwidth_kbps(size, transmit_time)

        client_transmit_bandwidths[client_name].append(bandwidth)

        if not message:
            continue

        inference_time, data_inference_list, processed_cumulative_layer_number = message
        logging.info(f"Processing time from {client_name}: {inference_time} seconds")

        # 记录客户端处理时间
        client_inference_times[client_name].append(inference_time)

        next_client_name = get_next_client(client_name)
        # 如果是最后一个客户端 则计算acc和loss。同时数据初始化
        if next_client_name is None:
            # 计算并记录这一轮的总执行时间
            round_time = sum(times[-1] for times in client_inference_times.values())
            round_times.append(round_time)

            # # 输出每个客户端的执行时间
            # for client, times in client_inference_times.items():
            #     logging.info(f"Round {iteration}, {client} execution time: {times[-1]} seconds")
            #
            # # 输出每个客户端的传输时间
            # for client, times in client_transmit_times.items():
            #     logging.info(f"Round {iteration}, {client} transmit time: {times[-1]} seconds")
            #
            # # 输出每个客户端的传输带宽
            # for client, bandwidth in client_transmit_bandwidths.items():
            #     logging.info(f"Round {iteration}, {client} bandwidth kbps: {bandwidth[-1]:.2f}")

            logging.info(f"Total time for round {iteration}: {round_time} seconds")
            # 计算结果
            get_loss_acc(data_inference_list, target_list)

            # 重新初始化
            logging.info("Round completed. Restarting with initial data.")
            data_inference_list = data_list
            processed_cumulative_layer_number = cumulative_layer_number
            next_client_name = list(node_layer_indices.keys())[0]

        next_client_conn = clients[next_client_name]
        # 序列化数据后发送
        data_to_send = [node_layer_indices, data_inference_list, processed_cumulative_layer_number]
        send_data(next_client_conn, data_to_send)


def accept_connections(server_socket):
    """
    接受并处理客户端的连接
    :param server_socket:  服务器套接字
    :return:              None
    """
    # 等待所有的客户端连接
    threads = []
    while len(clients) < len(node_layer_indices):
        conn, addr = server_socket.accept()
        client_name = receive_data(conn)
        clients[client_name] = conn
        print(f"{client_name} connected from {addr}")
        # 创建新线程来处理每个客户端
        client_thread = threading.Thread(target=handle_client, args=(conn, client_name))
        client_thread.start()
        threads.append(client_thread)

    # 用于记录每个客户端的处理时间, 传输时间, 传输带宽
    global client_inference_times, client_transmit_times, client_transmit_bandwidths
    client_inference_times = {name: [] for name in clients.keys()}
    client_transmit_times = {name: [] for name in clients.keys()}
    client_transmit_bandwidths = {name: [] for name in clients.keys()}
    first_client_name = list(node_layer_indices.keys())[0]
    # 消息封装
    data_to_send = [node_layer_indices, data_list, cumulative_layer_number]
    # 序列化数据后发送
    send_data(clients[first_client_name], data_to_send)

    # 等待所有客户端线程完成
    for thread in threads:
        thread.join()

    # 所有客户端线程完成后绘制图表
    plot_times(client_inference_times, client_transmit_times, client_transmit_bandwidths, round_times)


def main():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 创建一个 TCP 套接字，使用 IPv4 地址
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # 设置套接字选项，允许重新使用地址
    server_socket.bind(('localhost', 9000))  # 将套接字绑定到 localhost 上的 12345 端口
    server_socket.listen()  # 配置套接字进入监听状态，等待客户端连接
    print("Server is listening for connections...")
    accept_connections(server_socket)  # 调用 accept_connections 函数来接受并处理连接


if __name__ == "__main__":
    node_layer_indices = {'client1': [0, 1], 'client2': [2, 3], 'client3': [4, 5, 6]}
    model_name = 'VGG5'
    data_list, target_list = prepare_data()
    clients = {}
    # 记录客户的执行时间
    client_inference_times = {}
    # 记录客户的传输时间
    client_transmit_times = {}
    # 记录客户的传输带宽
    client_transmit_bandwidths = {}
    # 用于记录每轮的总执行时间
    round_times = []
    max_iterations = 1
    cumulative_layer_number = 0
    main()
