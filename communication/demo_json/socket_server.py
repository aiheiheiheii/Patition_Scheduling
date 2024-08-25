import socket
import threading
from message import Message

# 服务器配置
HOST = '127.0.0.1'
PORT = 9000

# 存储连接的客户端
clients = {}


# 处理客户端连接
def handle_client(client_socket):
    # 接收客户端发送的昵称
    data = client_socket.recv(1024).decode('utf-8')
    message = Message.from_json(data)
    endpoint = message.data
    clients[endpoint] = client_socket
    print(f"服务器 {endpoint} 上线。")
    broadcast(Message('join', endpoint).to_json())

    while True:
        try:
            data = client_socket.recv(1024).decode('utf-8')
            if data:
                message = Message.from_json(data)
                # 处理客户端之间的通信
                if message.type == 'private':
                    target_client = message.target
                    data = message.data
                    if target_client in clients:
                        target_socket = clients[target_client]
                        target_socket.send(Message('private', data, endpoint).to_json().encode('utf-8'))
                    else:
                        client_socket.send(Message('error', '目标客户端不在线').to_json().encode('utf-8'))
                elif message.type == 'public':
                    # 广播消息给所有在线客户端
                    broadcast(Message('public', message.data, endpoint).to_json())
            else:
                remove(client_socket, endpoint)
                break
        except socket.error:
            remove(client_socket, endpoint)
            break


# 广播消息给所有在线客户端
def broadcast(message):
    for client_socket in clients.values():
        client_socket.send(message.encode('utf-8'))


# 移除离线客户端
def remove(client_socket, endpoint):
    del clients[endpoint]
    client_socket.close()
    broadcast(Message('leave', endpoint).to_json())


# 接受新的客户端连接
def accept_connections():
    while True:
        client_socket, client_address = server.accept()
        print(f"新的连接来自: {client_address}")
        thread = threading.Thread(target=handle_client, args=(client_socket,))
        thread.start()


# 启动服务器
def start_server():
    print("*************SynerGist服务器已启动***************")
    accept_connections()


if __name__ == '__main__':
    # 创建服务器套接字
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen()

    # 启动SynerGist服务器
    start_server()
