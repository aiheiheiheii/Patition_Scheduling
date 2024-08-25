import socket
import threading
from network_utils import send_data, receive_data

node_layer_indices = {
    'task1': {'client1': [0, 1], 'client2': [2, 3, 4, 5, 6]},
    'task2': {'client2': [0, 1, 2], 'client3': [3, 4, 5, 6]},
    'task3': {'client3': [0, 1, 2, 3], 'client1': [4, 5, 6]}
}

initial_data = [1, 2, 3, 4, 5, 6, 7]
clients = {}
task_results = {task: [] for task in node_layer_indices}
client_status = {client: True for client in set(client for task in node_layer_indices.values() for client in task)}
lock = threading.Lock()
condition = threading.Condition(lock)


def handle_client(conn, client_name, task_name, layer_indices, data):
    data_to_send = (layer_indices, data)
    send_data(conn, data_to_send)
    message = receive_data(conn)
    if not message:
        return None, None

    process_time, processed_data = message
    print(f"{client_name} completed part of {task_name} in {process_time} seconds")

    return processed_data, process_time


def process_task(task_name, client_name, layer_indices, data):
    processed_data, process_time = handle_client(clients[client_name], client_name, task_name, layer_indices, data)

    if processed_data is not None:
        with lock:
            task_results[task_name].append(processed_data)
            print(task_results)
            client_status[client_name] = True
            condition.notify_all()
    else:
        print(f"{client_name} received empty data. Closing connection.")
        clients[client_name].close()


def manage_tasks():
    # Send the initial data to the first client of each task
    for task_name, clients_indices in node_layer_indices.items():
        first_client = list(clients_indices.keys())[0]
        layer_indices = clients_indices[first_client]
        data = initial_data

        with lock:
            client_status[first_client] = False

        threading.Thread(target=process_task, args=(task_name, first_client, layer_indices, data)).start()

    while True:
        with lock:
            all_tasks_done = True
            for task_name, clients_indices in node_layer_indices.items():
                if len(task_results[task_name]) < len(clients_indices):
                    all_tasks_done = False

                    # Find the next client in the sequence for the task
                    completed_clients = len(task_results[task_name])
                    next_client = list(clients_indices.keys())[completed_clients]

                    if client_status[next_client]:
                        layer_indices = clients_indices[next_client]
                        data = task_results[task_name][-1] if task_results[task_name] else initial_data
                        client_status[next_client] = False
                        threading.Thread(target=process_task,
                                         args=(task_name, next_client, layer_indices, data)).start()

            if all_tasks_done:
                break

            condition.wait()


def accept_connections(server_socket):
    while len(clients) < 3:  # Assuming there are three clients to connect
        conn, addr = server_socket.accept()
        client_name = receive_data(conn)
        if client_name:
            clients[client_name] = conn
            print(f"{client_name} connected from {addr}")

    manage_tasks()


def main():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('localhost', 12345))
    server_socket.listen()
    print("Server is listening for connections...")
    accept_connections(server_socket)
    server_socket.close()


if __name__ == "__main__":
    main()
