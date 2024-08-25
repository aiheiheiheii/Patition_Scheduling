import socket
import torch
import time
import matplotlib.pyplot as plt
import threading
from network_utils import send_data, receive_data
from config import dataset_path, B
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from utils import get_total_size, calculate_bandwidth_kbps
import logging
from optimize_dqn import DQNAgent
from data_inference import data_inference, get_loss_acc
from models.model_struct import model_cfg
from resource_utilization import get_linux_resource_info
logging.getLogger('matplotlib').propagate = False
# 获取 paramiko 日志记录器
paramiko_logger = logging.getLogger("paramiko")
# 设置日志级别为 WARNING
paramiko_logger.setLevel(logging.WARNING)

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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


def get_client_time_and_data(conn, client_name):
    # 记录客户端传输时间
    start_time = time.time()
    message = receive_data(conn)
    end_time = time.time()
    transmit_time = end_time - start_time
    client_transmit_times[client_name].append(transmit_time)
    print(f"{client_name}: transmit time is {transmit_time} seconds.")

    # 计算传输的数据量
    size = get_total_size(message)
    print(f"{client_name}: The total size of the list is {size} bytes.")
    # bandwidth = calculate_bandwidth_kbps(size, transmit_time)
    #
    # client_transmit_bandwidths[client_name].append(bandwidth)

    if not message:
        return None, None, None, None, None

    # 记录客户端的推理时间
    inference_time, data_inference_list, processed_cumulative_layer_number = message
    logging.info(f"Processing time from {client_name}: {inference_time} seconds")
    client_inference_times[client_name].append(inference_time)
    print(client_inference_times)
    return data_inference_list, processed_cumulative_layer_number, transmit_time, inference_time


def get_server_time(data_inference_list, target_list, processed_cumulative_layer_number):
    # 服务端推理时间
    start_time = time.time()
    result_list, _ = data_inference(data_inference_list, remain_layer_indices, processed_cumulative_layer_number)
    # 计算结果
    get_loss_acc(result_list, target_list)
    end_time = time.time()
    return end_time - start_time


def calculate_round_time(transmit_time, server_inference_time):
    """
    计算一轮的总执行时间
    :param transmit_time:           传输时间
    :param server_inference_time:   服务端推理时间
    :return:                        总执行时间
    """
    client_inference_time = sum(times[-1] for times in client_inference_times.values())
    round_time = client_inference_time + transmit_time + server_inference_time
    return round_time


def handle_client(conn, client_name):
    global client_layer_indices, remain_layer_indices, segment_point, layer_indices, data_list, target_list
    """
    处理客户端的请求
    :param conn:          客户端的连接
    :param client_name:  客户端的名称
    :return:              None
    """
    for iteration in range(1, max_iterations + 1):
        logging.info(f"Start processing round {iteration}")

        # 获取资源状态
        resource_state = get_linux_resource_info("192.168.215.133")
        print(f"Resource state: {resource_state}")
        state = [
            float(resource_state["cpu"][:-1]) / 100,
            float(resource_state["memory"]["Usage"][:-1]) / 100,
            float(resource_state["network"]["bytes_recv_per_sec"].split()[0])
        ]

        # 使用DQNAgent选择动作
        action = agent.act(state)
        segment_point = action
        segment_point = 1
        client_layer_indices = list(range(0, segment_point))
        remain_layer_indices = list(range(segment_point, len(model_cfg[model_name])))

        # 发送层索引到客户端
        data_to_send = [data_list, client_layer_indices, cumulative_layer_number]
        send_data(conn, data_to_send)

        # 接收客户端的请求
        data_inference_list, processed_cumulative_layer_number, transmit_time, inference_time = \
            get_client_time_and_data(conn, client_name)

        # 服务端推理时间
        server_inference_time = get_server_time(data_inference_list, target_list, processed_cumulative_layer_number)
        print(f"Server inference time: {server_inference_time} seconds")

        # 计算并记录这一轮的总执行时间
        round_time = calculate_round_time(transmit_time, server_inference_time)
        round_times.append(round_time)
        print(f"Round {iteration} total time: {round_time} seconds")

        # 计算奖励
        reward = -round_time

        # 获取新的资源状态
        next_resource_state = get_linux_resource_info("192.168.215.133")
        next_state = [
            float(next_resource_state["cpu"][:-1]) / 100,
            float(next_resource_state["memory"]["Usage"][:-1]) / 100,
            float(next_resource_state["network"]["bytes_sent_per_sec"].split()[0])
        ]

        # 存储经验
        agent.remember(state, action, reward, next_state, iteration == max_iterations)

        # 经验回放
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

        # 更新目标模型
        agent.update_target_model()

        # # 开始下一轮
        # next_client_name = client_name
        # segment_point, layer_indices = strategy.random_segmentation_point()
        # client_layer_indices = layer_indices[0]
        # remain_layer_indices = layer_indices[1]
        # next_client_conn = clients[next_client_name]
        # # 序列化数据后发送
        # data_to_send = [data_list, client_layer_indices, cumulative_layer_number]
        # send_data(next_client_conn, data_to_send)


def accept_connections(server_socket):
    """
    接受并处理客户端的连接
    :param server_socket:  服务器套接字
    :return:              None
    """
    # 等待所有的客户端连接
    threads = []
    while len(clients) < 1:
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

    # 等待所有客户端线程完成
    for thread in threads:
        thread.join()

    # 所有客户端线程完成后绘制图表
    # plot_times(client_inference_times, client_transmit_times, client_transmit_bandwidths, round_times)


def main():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 创建一个 TCP 套接字，使用 IPv4 地址
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # 设置套接字选项，允许重新使用地址
    server_socket.bind(('192.168.31.223', 9000))  # 将套接字绑定到 localhost 上的 12345 端口
    server_socket.listen()  # 配置套接字进入监听状态，等待客户端连接
    print("Server is listening for connections...")
    accept_connections(server_socket)  # 调用 accept_connections 函数来接受并处理连接

    # 保存训练后的模型
    torch.save(agent.model.state_dict(), "dqn_model2.pth")

    # 绘制训练曲线
    plt.plot(round_times)
    plt.xlabel('Round')
    plt.ylabel('Total Inference Time')
    plt.title('Training Curve')
    plt.savefig('training_curve.png')
    plt.show()


if __name__ == "__main__":
    model_name = 'VGG5'

    state_size = 3
    action_size = len(model_cfg[model_name])
    agent = DQNAgent(state_size, action_size)
    batch_size = 3

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
    max_iterations = 30
    cumulative_layer_number = 0
    main()
