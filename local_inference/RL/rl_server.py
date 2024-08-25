import socket
import torch
import matplotlib.pyplot as plt
import time
import threading
from network_utils import send_data, receive_data
from utils import get_total_size
import logging
from data_inference import data_inference, get_loss_acc
from models.model_struct import model_cfg
from dqn import DQNAgent
from resource_utilization import get_windows_resource_info

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_client_time_and_data(conn, client_name):
    # 记录客户端传输时间
    start_time = time.time()
    message = receive_data(conn)
    end_time = time.time()
    transmit_time = end_time - start_time
    client_transmit_times[client_name].append(transmit_time)
    logging.info(f"{client_name}: transmit time is {transmit_time} seconds.")

    # 计算传输的数据量
    size = get_total_size(message)
    logging.info(f"{client_name}: The total size of the list is {size} bytes.")

    if not message:
        return None, None, None, None, None

    # 记录客户端的推理时间
    target_list, inference_time, data_inference_list, processed_cumulative_layer_number = message
    logging.info(f"Processing time from {client_name}: {inference_time} seconds")
    client_inference_times[client_name].append(inference_time)
    return target_list, data_inference_list, processed_cumulative_layer_number, transmit_time, inference_time


def get_server_time(data_inference_list, target_list, processed_cumulative_layer_number):
    # 服务端推理时间
    start_time = time.time()
    result_list, _ = data_inference(data_inference_list, remain_layer_indices, processed_cumulative_layer_number)
    get_loss_acc(result_list, target_list)
    end_time = time.time()
    return end_time - start_time


def calculate_round_time(transmit_time, server_inference_time):
    client_inference_time = sum(times[-1] for times in client_inference_times.values())
    round_time = client_inference_time + transmit_time + server_inference_time
    return round_time


def handle_client(conn, client_name):
    global client_layer_indices, remain_layer_indices, segment_point, cumulative_layer_number
    for iteration in range(1, max_iterations + 1):
        logging.info(f"Start processing round {iteration}")

        # 获取资源状态
        resource_state = get_windows_resource_info()
        state = [
            float(resource_state["cpu"][:-1]) / 100,
            float(resource_state["memory"]["Usage"][:-1]) / 100,
            float(resource_state["network"]["bytes_recv_per_sec"].split()[0])
        ]

        # 使用DQNAgent选择动作
        action = agent.act(state)
        segment_point = action
        client_layer_indices = list(range(0, segment_point))
        remain_layer_indices = list(range(segment_point, len(model_cfg[model_name])))

        # 发送层索引到客户端
        data_to_send = [client_layer_indices, cumulative_layer_number]
        send_data(conn, data_to_send)

        # 接收客户端的数据
        target_list, data_inference_list, processed_cumulative_layer_number, transmit_time, inference_time = get_client_time_and_data(conn, client_name)

        if target_list is None:
            break

        # 服务端推理时间
        server_inference_time = get_server_time(data_inference_list, target_list, processed_cumulative_layer_number)
        logging.info(f"Server inference time: {server_inference_time} seconds")

        # 计算并记录这一轮的总执行时间
        round_time = calculate_round_time(transmit_time, server_inference_time)
        round_times.append(round_time)
        print(f"Round {iteration} total time: {round_time} seconds")

        # 计算奖励
        reward = -round_time

        # 获取新的资源状态
        next_resource_state = get_windows_resource_info()
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


def accept_connections(server_socket):
    threads = []
    while len(clients) < 1:
        conn, addr = server_socket.accept()
        client_name = receive_data(conn)
        clients[client_name] = conn
        logging.info(f"{client_name} connected from {addr}")
        client_thread = threading.Thread(target=handle_client, args=(conn, client_name))
        client_thread.start()
        threads.append(client_thread)

    global client_inference_times, client_transmit_times, client_transmit_bandwidths
    client_inference_times = {name: [] for name in clients.keys()}
    client_transmit_times = {name: [] for name in clients.keys()}
    client_transmit_bandwidths = {name: [] for name in clients.keys()}

    for thread in threads:
        thread.join()


def main():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('192.168.31.223', 9000))
    server_socket.listen()
    logging.info("Server is listening for connections...")
    accept_connections(server_socket)

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

    clients = {}
    client_inference_times = {}
    client_transmit_times = {}
    client_transmit_bandwidths = {}
    round_times = []
    max_iterations = 360
    cumulative_layer_number = 0
    main()
