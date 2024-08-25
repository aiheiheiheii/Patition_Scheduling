from socket.communicator import NodeEnd
from utils import get_client_app_port_by_name


def start():
    # 建立连接
    node = NodeEnd(host_ip, host_port)

    # 准备发送的消息内容
    msg = [info, node_layer_indices, layer_node_indices]

    first_client = list(node_layer_indices.keys())[0]
    next_ip, next_port = get_client_app_port_by_name(first_client, model_name)
    print("next_ip  ", next_ip)
    print("next_port  ", next_port)

    # 创建一个新的连接来连接下一个客户端
    node.node_connect(next_ip, next_port)

    # 将处理后的数据发送给下一个客户端
    node.send_message(node.sock, msg)
    print(f"客户端{host_ip}:{host_port}将处理后的消息发送到客户端{next_ip}:{next_port}")

    # 关闭与下一个客户端的连接
    node.sock.close()


def process_data(msg):
    # 在这里对数据进行处理,这里只是简单地在消息前面加上"processed_by_client1_"
    processed_msg = ["processed_by_client1_" + str(m) for m in msg]
    return processed_msg


if __name__ == '__main__':

    host_ip = '127.0.0.1'
    host_port = 9000

    model_name = 'VGG5'

    node_layer_indices = {'client2': [0, 1], 'client1': [2, 3], 'client3': [4, 5, 6]}
    layer_node_indices = {
        0: '127.0.0.1',
        1: '127.0.0.1',
        2: '127.0.0.1',
        3: '127.0.0.1',
        4: '127.0.0.1',
        5: '127.0.0.1',
        6: '127.0.0.1'
    }

    info = "MSG_FROM_NODE_ADDRESS(%s), port= %s" % (host_ip, host_port)

    start()
