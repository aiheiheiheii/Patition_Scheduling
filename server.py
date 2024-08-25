"""
    服务器执行文件，主要负责：
        1、根据客户端的资源使用情况。将模型文件进行分层
        ip及对应节点位序
"""
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from communication.communicator import NodeEnd
from models.model_struct import model_cfg
from strategy.segment_strategy import NetworkSegmentationStrategy
from strategy.resource_utilization import get_all_server_info
from utils import get_client_app_port
from config import dataset_config, B


def prepare_data():
    """
    加载数据
    :return:
    """
    data_dir = dataset_config.get(model_name)
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
        test_dataset, batch_size=B, shuffle=False, num_workers=4
    )
    data_cpu_list, target_cpu_list = [], []
    for data, target in test_loader:
        data_cpu_list.append(data)
        target_cpu_list.append(target)
    return data_cpu_list, target_cpu_list


def start():
    # 建立连接
    node = NodeEnd(docker_host_ip, host_port)

    # 获取第一个客户端的地址
    first_client_ip = list(node_layer_indices.keys())[0]
    first_client_port = get_client_app_port(first_client_ip, model_name)

    # 连接分层策略给的第一个客户端
    node.node_connect(first_client_ip, first_client_port)

    # 准备发送的消息内容
    cumulative_layer_number = 0
    data_list, target_list = prepare_data()
    msg = [info, node_layer_indices, data_list, target_list, cumulative_layer_number]

    # 发送信息
    node.send_message(node.sock, msg)
    print(f"服务端{host_ip}:{host_port}将数据发送到客户端{first_client_ip}:{first_client_port}")
    # 关闭连接
    node.sock.close()


if __name__ == '__main__':
    # docker 容器中
    docker_host_ip = '0.0.0.0'

    # 服务器的ip 端口
    host_ip = '192.168.215.132'
    host_port = 9001

    # 模型名称
    model_name = "VGG5"

    # 获取所有节点的资源情况
    nodes_resource_infos = get_all_server_info()

    # 分割策略类
    segmentation_strategy = NetworkSegmentationStrategy(model_name, model_cfg)

    # 利用所有节点的资源情况的资源感知分割点方法进行分割
    # segmentation_points:  [2, 4]
    # node_layer_indices:  {'192.168.215.130': [0, 1], '192.168.215.131': [2, 3], '192.168.215.129': [4, 5, 6]}
    segmentation_points, node_layer_indices = (segmentation_strategy
                                               .resource_aware_segmentation_points(nodes_resource_infos))
    print('*' * 40)
    print("resource_aware_segmentation_points  segmentation_points: ", segmentation_points)
    print("resource_aware_segmentation_points  node_layer_indices: ", node_layer_indices)

    info = "MSG_FROM_NODE_ADDRESS(%s), host= %s" % (host_ip, host_port)

    # 开始
    start()
