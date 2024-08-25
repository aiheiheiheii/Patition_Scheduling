from socket.communicator import NodeEnd
from utils import get_client_app_port_by_name


def get_next_key(current_key):
    keys = list(node_layer_indices.keys())
    try:
        current_key_index = keys.index(current_key)
        next_key_index = current_key_index + 1
        if next_key_index < len(keys):  # 确保不超出列表范围
            return keys[next_key_index]
        else:
            return None  # 当前键是最后一个键，没有下一个键
    except ValueError:
        return None  # 当前键不在字典中


def process_data(data):
    # 在这里对数据进行处理,这里只"
    data = data + " 处理 "
    return data


def from_first(node):
    print("从第一层开始")
    next_client = get_next_key(name)

    # 如果不是最后一个客户端
    if next_client:
        next_ip, next_port = get_client_app_port_by_name(next_client, model_name)
        print("next_ip(%s), next_port= %s" % (next_ip, next_port))

        # 创建一个新的连接来连接下一个客户端
        node.__init__(host_ip, host_port)
        node.node_connect(next_ip, next_port)

        # 将处理后的数据发送给下一个客户端
        data = "FIRST_LAYER_PROCESS_BY_NODE_ADDRESS(%s), host= %s" % (host_ip, host_port)
        msg = [info, node_layer_indices, layer_node_indices, data]
        node.send_message(node.sock, msg)
        print(f"客户端{host_ip}:{host_port}将处理后的消息发送到客户端{next_ip}:{next_port}")

        # 关闭与下一个客户端的连接
        node.sock.close()
    # 如果是最后一个客户端，则计算结果，推理结束
    else:
        print("推理结束")


def node_inference(node, msg):
    global info, node_layer_indices, layer_node_indices, current_layers
    info, node_layer_indices, layer_node_indices, data = msg

    print(data)

    print("持续推理")

    next_client = get_next_key(name)

    # 如果不是最后一个客户端
    if next_client:
        next_ip, next_port = get_client_app_port_by_name(next_client, model_name)
        print("next_ip(%s), next_port= %s" % (next_ip, next_port))

        # 创建一个新的连接来连接下一个客户端
        node.__init__(host_ip, host_port)
        node.node_connect(next_ip, next_port)

        # 本地需要处理的数据
        data = process_data(data)
        print("processed data: ", data)

        # 将处理后的数据发送给下一个客户端
        msg = [info, node_layer_indices, layer_node_indices, data]
        node.send_message(node.sock, msg)
        print(f"客户端{host_ip}:{host_port}将处理后的消息发送到客户端{next_ip}:{next_port}")

        # 关闭与下一个客户端的连接
        node.sock.close()
    # 如果是最后一个客户端，则计算结果，推理结束
    else:
        print("推理结束")


def start():
    # 建立连接
    node = NodeEnd(host_ip, host_port)

    # 等待上一个客户端的连接
    connect_socket, _ = node.wait_for_connection()

    # 接收上一个客户端发送的数据
    msg = node.receive_message(connect_socket)
    print(f"客户端{host_ip}:{host_port}接收到消息: {msg}")
    # 关闭连接
    node.sock.close()

    if len(msg) == 3:
        global info, node_layer_indices, layer_node_indices, current_layers
        info, node_layer_indices, layer_node_indices = msg
        current_layers = node_layer_indices[name]
        print(info)
        from_first(node)
    else:
        node_inference(node, msg)


if __name__ == '__main__':
    name = 'client2'
    model_name = 'VGG5'

    host_ip, host_port = get_client_app_port_by_name(name, model_name)
    print(host_ip, host_port)

    current_layers = []
    info = ''
    node_layer_indices, layer_node_indices = {}, {}

    start()
