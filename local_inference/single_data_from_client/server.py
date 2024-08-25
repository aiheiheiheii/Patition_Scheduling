from communication.communicator import NodeEnd
from utils import get_client_app_port_by_name


def start():
    # 建立连接
    node = NodeEnd(host_ip, host_port)

    # 准备发送的消息内容
    msg = [info, node_layer_indices]

    first_client = list(node_layer_indices.keys())[0]
    next_ip, next_port = get_client_app_port_by_name(first_client, model_name)
    print("next_ip(%s), next_port= %s" % (next_ip, next_port))

    # 创建一个新的连接来连接下一个客户端
    node.node_connect(next_ip, next_port)

    # 将处理后的数据发送给下一个客户
    node.send_message(node.sock, msg)
    print(f"客户端{host_ip}:{host_port}将处理后的消息发送到客户端{next_ip}:{next_port}")

    # 关闭与下一个客户端的连接
    node.sock.close()


if __name__ == '__main__':

    host_ip = '127.0.0.1'
    host_port = 9000

    model_name = 'VGG5'

    node_layer_indices = {'client2': [0, 1], 'client1': [2, 3], 'client3': [4, 5, 6]}

    info = "MSG_FROM_NODE_ADDRESS(%s), port= %s" % (host_ip, host_port)

    start()
