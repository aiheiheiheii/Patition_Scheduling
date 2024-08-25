import torch
from torch import nn
from models.model_struct import model_cfg
from models.vgg5.vgg5 import VGG5
import socket
import time
import config
from network_utils import send_data, receive_data
from utils import get_client_app_port_by_name
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_model(layer_type, in_channels, out_channels, kernel_size, cumulative_layer_number):
    feature_seq = []
    dense_seq = []
    if layer_type == "M":
        feature_seq.append(model.features[cumulative_layer_number])
        cumulative_layer_number += 1
    elif layer_type == "D":
        dense_seq.append(model.denses[cumulative_layer_number - 11])
        cumulative_layer_number += 1
    elif layer_type == "C":
        for _ in range(3):
            feature_seq.append(model.features[cumulative_layer_number])
            cumulative_layer_number += 1
    return nn.Sequential(*feature_seq), nn.Sequential(*dense_seq), cumulative_layer_number

def calculate_output(node_layer_indices, data, cumulative_layer_number):
    for index in node_layer_indices[client_name]:
        layer_type = model_cfg[model_name][index][0]
        in_channels = model_cfg[model_name][index][1]
        out_channels = model_cfg[model_name][index][2]
        kernel_size = model_cfg[model_name][index][3]
        features, dense, cumulative_layer_number = get_model(layer_type, in_channels, out_channels, kernel_size, cumulative_layer_number)
        model_layer = features if len(features) > 0 else dense
        if layer_type == "D":
            data = data.view(data.size(0), -1)
        data = model_layer(data)
    return data, cumulative_layer_number

def node_inference(node_indices, data_list, cumulative_layer_number):
    logging.info("*********************开始推理************************")
    result_list = []
    start_layer = cumulative_layer_number
    for i in range(config.iterations):
        data = data_list[i]
        result, cumulative_layer_number = calculate_output(node_indices, data, start_layer)
        result_list.append(result)
    return result_list, cumulative_layer_number

def client(name, client_port=None):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if client_port:
        try:
            client_socket.bind(('', client_port))
            logging.info(f"绑定到本地端口 {client_port}")
        except socket.error as e:
            logging.error(f"绑定到端口 {client_port} 失败: {e}")
            return

    try:
        client_socket.connect(('localhost', 9000))
        logging.info(f"作为 {name} 连接到服务器")
    except socket.error as e:
        logging.error(f"连接到服务器失败: {e}")
        return

    send_data(client_socket, name)

    while True:
        data = receive_data(client_socket)
        if data:
            node_indices, data_list, cumulative_layer_number = data
            logging.info(f"{name} 收到数据: {node_indices}, {data_list}")
            start_time = time.time()
            processed_data_list, processed_cumulative_layer_number = node_inference(node_indices, data_list, cumulative_layer_number)
            end_time = time.time()
            process_time = end_time - start_time
            response = [process_time, processed_data_list, processed_cumulative_layer_number]
            send_data(client_socket, response)
        else:
            break

    client_socket.close()

if __name__ == "__main__":
    client_name = 'client3'
    model_name = 'VGG5'
    host_ip, host_port = get_client_app_port_by_name(client_name, model_name)
    model = VGG5("Client", model_name, len(model_cfg[model_name]) - 1, model_cfg)
    model.eval()
    model.load_state_dict(torch.load("../../models/vgg5/vgg5.pth"))
    client(client_name, client_port=host_port)
