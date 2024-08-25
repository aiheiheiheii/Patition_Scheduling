import socket
import threading
import json
from message import Message

# 服务器配置
HOST = '127.0.0.1'
PORT = 9000

# 客户端配置
CLIENT_HOST = '127.0.0.1'
CLIENT_PORT = 9003


# 创建客户端套接字
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.bind((CLIENT_HOST, CLIENT_PORT))
client_socket.connect((HOST, PORT))

# 获取客户端的IP地址和端口号
client_address = client_socket.getsockname()
client_ip = client_address[0]
client_port = client_address[1]

# 将客户端的IP地址和端口号作为昵称
endpoint = f"{client_ip}:{client_port}"

# 发送昵称给服务器
client_socket.send(Message('endpoint', endpoint).to_json().encode('utf-8'))


# 接收服务器消息
def receive_messages():
    while True:
        try:
            data = client_socket.recv(1024).decode('utf-8')
            message = Message.from_json(data)
            if message.type == 'public':
                print(f"{message.sender}: {message.data}")
            elif message.type == 'private':
                print(f"[私] {message.sender}: {message.data}")
            elif message.type == 'error':
                print(message.data)
        except socket.error:
            print("与服务器的连接已断开。")
            client_socket.close()
            break


# 发送消息给服务器
def send_messages():
    while True:
        message_type = input("请输入消息类型 (public/private): ")
        if message_type == 'private':
            target_username = input("请输入目标用户名: ")
            content = input("请输入消息内容: ")
            message = Message('private', content, endpoint, target_username)
        else:
            content = input("请输入消息内容: ")
            message = Message('public', content, endpoint)
        client_socket.send(message.to_json().encode('utf-8'))


# 启动消息接收线程
receive_thread = threading.Thread(target=receive_messages)
receive_thread.start()

# 启动消息发送线程
send_thread = threading.Thread(target=send_messages)
send_thread.start()
