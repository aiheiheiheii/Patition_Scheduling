# client1.py

import random
import socket
import time
from network_utils import send_data, receive_data


def process_data(data):
    time.sleep(random.uniform(0.1, 0.5))
    return data


def client(name, client_port=None):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    if client_port:
        try:
            client_socket.bind(('', client_port))
            print(f"Bound to local port {client_port}")
        except socket.error as e:
            print(f"Failed to bind to port {client_port}: {e}")
            return

    try:
        client_socket.connect(('localhost', 12345))
        print(f"Connected to server as {name}")
    except socket.error as e:
        print(f"Failed to connect to server: {e}")
        return

    send_data(client_socket, name)

    while True:
        data = receive_data(client_socket)
        if data:
            layer_indices, data = data
            print(f"{name} Received at : {layer_indices} ,{data}")

            # 执行时间
            start_time = time.time()
            processed_data = process_data(data)
            end_time = time.time()
            process_time = end_time - start_time
            # 拼接数据
            response = [process_time, processed_data]
            print(f"{name} Sending response: {response}")
            send_data(client_socket, response)
        else:
            print(f"{name} received empty data. Closing connection.")
            break

    client_socket.close()


if __name__ == "__main__":
    client_name = 'client2'
    client(client_name, client_port=9002)
