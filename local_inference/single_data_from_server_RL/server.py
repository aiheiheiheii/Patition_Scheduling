import socket
import threading
import torch.nn.functional as F
from network_utils import send_data, receive_data
from config import B, dataset_path, iterations, max_iterations
import logging
from RLEnv import FederatedEnv
from PPO import PPO, Memory

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_next_client(current_client):
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
    predictions = fx.max(1, keepdim=True)[1]
    correct = predictions.eq(y.view_as(predictions)).sum()
    acc = 100.00 * correct.float() / predictions.shape[0]
    return acc


def get_loss_acc(result_list, target):
    loss_list, acc_list = []
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
    for iteration in range(1, max_iterations + 1):
        logging.info(f"Start processing round {iteration}")
        message = receive_data(conn)
        if not message:
            continue

        inference_time, data_inference_list, processed_cumulative_layer_number = message
        logging.info(f"Processing time from {client_name}: {inference_time} seconds")

        next_client = get_next_client(client_name)
        if next_client:
            next_client_address = (next_client['host'], next_client['port'])
            logging.info(f"Sending data to the next client: {next_client_address}")
            send_data(next_client_address, message)
        else:
            logging.info(f"Processing completed by client: {client_name}")
            # 计算平均损失和准确率
            avg_loss, avg_acc = get_loss_acc(data_inference_list, target)
            break

    conn.close()


def server_program():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('localhost', 5000))
    server_socket.listen(5)

    env = FederatedEnv()
    ppo = PPO(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n)
    memory = Memory([], [], [], [], [])

    logging.info("Server started and waiting for connections")

    while True:
        conn, address = server_socket.accept()
        logging.info(f"Connection from {address} established")
        threading.Thread(target=handle_client, args=(conn,)).start()

        state = env.reset()
        for t in range(10000):
            action = ppo.policy_old.act(state)
            state, reward, done, _ = env.step(action)
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            memory.states.append(state)
            memory.actions.append(action)
            if done:
                ppo.update(memory)
                memory = Memory([], [], [], [], [])
                state = env.reset()

if __name__ == "__main__":
    server_program()
