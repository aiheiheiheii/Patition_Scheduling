from communication.communicator import NodeEnd
from utils import get_client_app_port_by_name
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from config import dataset_config, B, dataset_path


def prepare_data():
    """
    加载数据
    :return:
    """
    data_dir = dataset_path + dataset_config.get(model_name)
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


def start():
    # 建立连接
    node = NodeEnd(host_ip, host_port)

    # 获取第一个客户端的地址
    first_client = list(node_layer_indices.keys())[0]
    next_ip, next_port = get_client_app_port_by_name(first_client, model_name)
    print("next_ip(%s), next_port= %s" % (next_ip, next_port))

    # 创建一个新的连接来连接下一个客户端
    node.node_connect(next_ip, next_port)

    # 准备发送的消息内容
    cumulative_layer_number = 0
    data_list, target_list = prepare_data()
    msg = [info, node_layer_indices, data_list, target_list, cumulative_layer_number]

    # 将处理后的数据发送给下一个客户端
    node.send_message(node.sock, msg)
    print(f"客户端{host_ip}:{host_port}将处理后的消息发送到客户端{next_ip}:{next_port}")

    # 关闭与下一个客户端的连接
    node.sock.close()


if __name__ == '__main__':
    host_ip = '127.0.0.1'
    host_port = 9000

    model_name = 'VGG5'

    node_layer_indices = {'client1': [0, 1], 'client2': [2, 3], 'client3': [4, 5, 6]}

    info = "MSG_FROM_NODE_ADDRESS(%s), port= %s" % (host_ip, host_port)

    start()
