import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import time
import numpy as np
# 导入线程模块，用于多线程处理
import threading
import json
# 导入操作符模块，用于数学运算
import operator

import sys
sys.path.append('../')
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import config
import utils
from correspondence import *
from models.model_struct import model_cfg


# 如果配置中指定使用随机种子
if config.random:
    # 设置PyTorch随机种子，确保实验的可重复性
    torch.manual_seed(config.random_seed)
    # 设置NumPy随机种子
    np.random.seed(config.random_seed)
    # 记录随机种子
    logger.info('Random seed: {}'.format(config.random_seed))


class Env(Correspondence):
    def __init__(self, index, ip_address, server_port, clients_list, model_name, model_cfg, batch_size):
        super(Env, self).__init__(index, ip_address)

        # 环境索引
        self.infer_state = None
        self.threads = None
        self.net_threads = None
        self.criterion = None
        self.optimizers = None
        self.nets = None
        self.split_layers = None
        self.cluster_centers = None
        self.group_model = None
        self.baseline = None
        self.offloading_state = None
        self.network_state = None
        self.index = index
        # 客户端列表
        self.clients_list = clients_list
        # 模型名称
        self.model_name = model_name
        # 批处理大小
        self.batch_size = batch_size
        # 模型配置(模型的层级结构)
        self.model_cfg = model_cfg
        # 状态维度
        self.state_dim = 2 * config.G
        # 动作维度
        self.action_dim = config.G
        # 初始化分组标签列表
        self.group_labels = []
        # 获取模型FLOPS列表
        self.model_flops_list = self.get_model_flops_list(model_cfg, model_name)
        # 断言模型层数与FLOPS列表长度相同
        assert len(self.model_flops_list) == config.model_len

        # Server configration
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.port = server_port
        # self.model_name = model_name
        # 绑定服务器IP和端口
        self.sock.bind((self.ip, self.port))
        # 初始化客户端套接字字典
        self.client_socks = {}

        # 等待所有客户端连接
        while len(self.client_socks) < config.K:
            # 监听连接
            self.sock.listen(5)
            # 接受客户端连接
            (client_sock, (ip, port)) = self.sock.accept()
            # 将客户端套接字添加到字典
            self.client_socks[str(ip)] = client_sock

        self.uni_net = utils.get_model('Unit', self.model_name, 0, self.device, self.model_cfg)

    def reset(self, done, first):
        """
        重置环境状态，通常在每个episode开始时调用。将环境恢复到一个已知的初始状态，以便开始新的episode。
        :param done:
        :param first:
        :return:
        """
        # 初始化split_layers为空列表
        split_layers = []
        # 修改: 根据配置的客户端数量，随机生成分割点
        for _ in range(config.K - 1):
            split_layers.append(np.random.randint(1, config.model_len - 1))
        split_layers.sort()  # 对分割点进行排序
        split_layers.append(config.model_len - 1)  # 将最后一层添加到分割点列表中

        config.split_layer = split_layers
        thread_number = config.K
        client_ips = config.CLIENTS_LIST
        # 初始化客户端，包括模型和优化器的设置。
        self.initialize(split_layers)
        # 创建重置标志消息
        msg = ['RESET_FLAG', True]
        # 向所有客户端发送重置标志消息
        self.scatter(msg)

        # 测试网络速度并记录网络状态
        self.test_network(thread_number, client_ips)
        # 初始化网络状态字典
        self.network_state = {}
        for s in self.client_socks:  # 遍历每个客户端
            msg = self.recv_message(self.client_socks[s], 'MSG_TEST_NETWORK_SPEED')  # 接收网络速度消息
            self.network_state[msg[1]] = msg[2]  # 存储网络速度

        # 经典联邦学习训练
        if first:
            # 执行两次推理（训练）以模拟客户端的本地训练。
            self.infer(thread_number, client_ips)
            self.infer(thread_number, client_ips)
        else:
            self.infer(thread_number, client_ips)

        # 获取当前的卸载状态
        self.offloading_state = self.get_offloading_state(
            split_layers,
            self.clients_list,
            self.model_cfg,
            self.model_name
        )
        # 设置为基线，用于后续奖励计算的参考。
        self.baseline = self.infer_state  # Set baseline for normalization
        # 如果没有分组标签
        if len(self.group_labels) == 0:
            self.group_model, self.cluster_centers, self.group_labels = self.group(self.baseline, self.network_state)

        logger.info('Baseline: ' + json.dumps(self.baseline))

        # 合并和归一化环境状态
        state = self.concat_norm(self.clients_list, self.network_state, self.infer_state, self.offloading_state)  # 构建状态
        # 断言状态维度与状态长度相同
        assert self.state_dim == len(state)
        # 返回状态
        return np.array(state)


    def step(self, action, done):
        """
        执行环境的一个步骤，包括执行动作、更新状态、计算奖励等。
        :param action:
        :param done:
        :return:
        """
        # 修改: 直接将动作转换为分层信息，不需要扩展动作
        config.split_layer = self.action_to_layer(action)
        # 获取分层信息
        split_layers = config.split_layer
        # 记录当前分层操作
        logger.info('Current OPs: ' + str(split_layers))
        # 线程数量
        thread_number = config.K
        # 客户端IP列表
        client_ips = config.CLIENTS_LIST
        # 初始化
        self.initialize(split_layers)

        msg = ['RESET_FLAG', False]  # 创建非重置标志消息
        self.scatter(msg)  # 向所有客户端发送非重置标志消息

        msg = ['SPLIT_LAYERS', config.split_layer]  # 创建分层信息消息
        self.scatter(msg)  # 向所有客户端发送分层信息消息

        # 测试网络速度
        self.test_network(thread_number, client_ips)
        # 初始化网络状态字典
        self.network_state = {}
        # 遍历每个客户端
        for s in self.client_socks:
            msg = self.recv_message(self.client_socks[s], 'MSG_TEST_NETWORK_SPEED')  # 接收网络速度消息
            self.network_state[msg[1]] = msg[2]  # 存储网络速度

        # 进行推理
        self.infer(thread_number, client_ips)
        # 获取卸载状态
        self.offloading_state = self.get_offloading_state(
            split_layers,
            self.clients_list,
            self.model_cfg,
            self.model_name
        )
        # 计算奖励、最大时间和是否完成
        reward, maxtime, done = self.calculate_reward(self.infer_state)
        # 记录每次迭代的训练时间
        logger.info('Training time per iteration: ' + json.dumps(self.infer_state))
        # 构建状态
        state = self.concat_norm(self.clients_list, self.network_state, self.infer_state, self.offloading_state)
        # 断言状态维度与状态长度相同
        assert self.state_dim == len(state)
        # 返回状态、奖励、最大时间和是否完成
        return np.array(state), reward, maxtime, done

    def initialize(self, split_layers):
        """
        根据客户端是否需要卸载任务，初始化客户端模型和优化器。
        :param split_layers:
        :return:
        """
        # 存储分层信息
        self.split_layers = split_layers
        # 初始化客户端模型字典
        self.nets = {}
        # 初始化优化器字典
        self.optimizers = {}
        # 遍历每个客户端
        for i in range(len(split_layers)):
            # 获取客户端IP
            client_ip = config.CLIENTS_LIST[i]
            # 如果客户端需要卸载
            if split_layers[i] < config.model_len - 1:
                # 获取服务器端模型
                self.nets[client_ip] = utils.get_model(
                    'Server',
                    self.model_name,
                    split_layers[i],
                    self.device,
                    self.model_cfg
                )
                # 初始化优化器
                self.optimizers[client_ip] = optim.SGD(
                    self.nets[client_ip].parameters(),
                    lr=config.LR,
                    momentum=0.9
                )
        # 定义损失函数
        self.criterion = nn.CrossEntropyLoss()

    def test_network(self, thread_number, client_ips):
        """
         测试网络速度，用于确定数据传输时间。
        :param thread_number:
        :param client_ips:
        :return:
        """
        self.net_threads = {}  # 初始化网络测试线程字典
        for i in range(len(client_ips)):  # 遍历每个客户端
            self.net_threads[client_ips[i]] = threading.Thread(
                target=self._thread_network_testing,
                args=(client_ips[i],))  # 创建网络测试线程
            self.net_threads[client_ips[i]].start()  # 启动线程

        for i in range(len(client_ips)):  # 等待所有线程结束
            self.net_threads[client_ips[i]].join()

    def _thread_network_testing(self, client_ip):
        """
        线程函数，用于测试与特定客户端的网络连接速度。
        :param client_ip:
        :return:
        """
        msg = self.recv_message(self.client_socks[client_ip], 'MSG_TEST_NETWORK_SPEED')  # 接收网络测试消息
        msg = ['MSG_TEST_NETWORK_SPEED', self.uni_net.cpu().state_dict()]  # 创建网络测试消息
        self.send_message(self.client_socks[client_ip], msg)  # 发送网络测试消息

    def infer(self, thread_number, client_ips):
        """
        执行模型推理（训练）。根据客户端是否需要卸载任务，可能会在本地或服务器上执行。
        :param thread_number:
        :param client_ips:
        :return:
        """
        self.threads = {}  # 初始化训练线程字典
        for i in range(len(client_ips)):  # 遍历每个客户端
            if self.split_layers[i] == config.model_len - 1:  # 如果客户端不需要卸载
                # 创建不卸载推理线程
                self.threads[client_ips[i]] = threading.Thread(target=self._thread_infer_no_offloading,
                                                               args=(client_ips[i],))
                logger.debug(str(client_ips[i]) + ' no offloading infer start')  # 记录信息
                self.threads[client_ips[i]].start()  # 启动线程
            else:  # 如果客户端需要卸载
                logger.debug(str(client_ips[i]))
                # 创建卸载推理线程
                self.threads[client_ips[i]] = threading.Thread(target=self._thread_infer_offloading,
                                                               args=(client_ips[i],))
                logger.debug(str(client_ips[i]) + ' offloading infer start')  # 记录信息
                self.threads[client_ips[i]].start()  # 启动线程

        for i in range(len(client_ips)):  # 等待所有线程结束
            self.threads[client_ips[i]].join()

        self.infer_state = {}  # 初始化推理状态字典
        for s in self.client_socks:  # 遍历每个客户端
            msg = self.recv_message(self.client_socks[s], 'MSG_INFER_SPEED')  # 接收推理速度消息
            self.infer_state[msg[1]] = msg[2]  # 存储推理速度

    def _thread_infer_no_offloading(self, client_ip):
        """
        不卸载推理线程函数，用于在客户端本地执行全部模型训练。
        :param client_ip:
        :return:
        """
        pass

    def _thread_infer_offloading(self, client_ip):
        """
        卸载推理线程函数，用于执行卸载任务，即部分模型层在客户端执行，其余层在服务器执行。
        :param client_ip:
        :return:
        """
        for i in range(config.iteration[client_ip]):  # 遍历每次迭代
            msg = self.recv_message(self.client_socks[client_ip], 'MSG_LOCAL_ACTIVATIONS_CLIENT_TO_SERVER')  # 接收客户端激活值和标签
            smashed_layers = msg[1]  # 获取激活值
            labels = msg[2]  # 获取标签

            self.optimizers[client_ip].zero_grad()  # 梯度清零
            inputs, targets = smashed_layers.to(self.device), labels.to(self.device)  # 移动数据到设备
            outputs = self.nets[client_ip](inputs)  # 前向传播
            loss = self.criterion(outputs, targets)  # 计算损失
            loss.backward()  # 反向传播

            self.optimizers[client_ip].step()  # 更新参数

            # 发送梯度给客户端
            msg = ['MSG_SERVER_GRADIENTS_SERVER_TO_CLIENT_' + str(client_ip), inputs.grad]  # 创建梯度消息
            self.send_message(self.client_socks[client_ip], msg)  # 发送梯度消息

    def scatter(self, msg):
        """
        向所有客户端发送消息。
        :param msg:
        :return:
        """
        for i in self.client_socks:  # 遍历每个客户端套接字
            self.send_message(self.client_socks[i], msg)  # 发送消息

    def get_offloading_state(self, split_layer, clients_list, model_cfg, model_name):
        """
         计算每个客户端的卸载状态，即确定有多少计算卸载到了服务器。
        :param split_layer:
        :param clients_list:
        :param model_cfg:
        :param model_name:
        :return:
        """
        offloading_state = {}  # 初始化卸载状态字典
        offload = 0  # 初始化卸载量

        assert len(split_layer) == len(clients_list)  # 检查分层长度和客户端数量一致
        for i in range(len(clients_list)):  # 遍历每个客户端
            for l in range(len(model_cfg[model_name])):  # 遍历每一层
                if l <= split_layer[i]:  # 如果层索引小于等于分层索引
                    offload += model_cfg[model_name][l][5]  # 累加卸载量

            offloading_state[clients_list[i]] = offload / config.total_flops  # 计算卸载比例
            offload = 0  # 重置卸载量

        return offloading_state  # 返回卸载状态字典

    def get_model_flops_list(self, model_cfg, model_name):
        """
        获取模型的浮点运算次数（FLOPS）列表，用于确定模型的计算复杂度。
        :param model_cfg:
        :param model_name:
        :return:
        """
        model_state_flops = []  # 初始化模型状态FLOPS列表
        cumulated_flops = 0  # 初始化累计FLOPS

        for l in model_cfg[model_name]:  # 遍历模型配置
            cumulated_flops += l[5]  # 累加FLOPS
            model_state_flops.append(cumulated_flops)  # 添加到列表

        model_flops_list = np.array(model_state_flops)  # 转换为NumPy数组
        model_flops_list = model_flops_list / cumulated_flops  # 归一化

        return model_flops_list  # 返回模型FLOPS列表

    def concat_norm(self, clients_list, network_state, infer_state, offloading_state):
        """
        将网络状态、推理状态和卸载状态合并为环境状态，并进行归一化处理。
        :param clients_list:
        :param network_state:
        :param infer_state:
        :param offloading_state:
        :return:
        """
        network_state_order = []  # 初始化网络状态列表
        infer_state_order = []  # 初始化推理状态列表
        offloading_state_order = []  # 初始化卸载状态列表
        for c in clients_list:  # 遍历每个客户端
            network_state_order.append(network_state[c])  # 添加网络状态
            infer_state_order.append(infer_state[c])  # 添加推理状态
            offloading_state_order.append(offloading_state[c])  # 添加卸载状态

        group_max_index = [0 for i in range(config.G)]  # 初始化每组最大索引列表
        group_max_value = [0 for i in range(config.G)]  # 初始化每组最大值列表
        for i in range(len(clients_list)):  # 遍历每个客户端
            label = self.group_labels[i]  # 获取客户端分组标签
            if infer_state_order[i] >= group_max_value[label]:  # 如果推理状态大于等于当前最大值
                group_max_value[label] = infer_state_order[i]  # 更新最大值
                group_max_index[label] = i  # 更新最大索引

        infer_state_order = np.array(infer_state_order)[np.array(group_max_index)]  # 获取每组最大推理状态
        offloading_state_order = np.array(offloading_state_order)[np.array(group_max_index)]  # 获取每组最大卸载状态
        network_state_order = np.array(network_state_order)[np.array(group_max_index)]  # 获取每组最大网络状态
        state = np.append(infer_state_order, offloading_state_order)  # 合并推理状态和卸载状态

        return state  # 返回环境状态

    def calculate_reward(self, infer_state):
        """
        计算奖励、最大时间和完成标志，奖励基于推理时间与基线时间的比较。
        :param infer_state:
        :return:
        """
        rewards = {}  # 初始化奖励字典
        reward = 0  # 初始化奖励
        done = False  # 初始化完成标志为False

        max_base_time = max(self.baseline.items(), key=operator.itemgetter(1))[1]  # 获取基准最大时间
        max_infer_time = max(infer_state.items(), key=operator.itemgetter(1))[1]  # 获取推理最大时间
        max_infer_time_index = max(infer_state.items(), key=operator.itemgetter(1))[0]  # 获取推理最大时间索引

        if max_infer_time >= 1 * max_base_time:  # 如果推理最大时间大于等于基准最大时间
            done = True  # 设置完成标志为True
        # reward += - 1 # 减小奖励
        else:
            done = False  # 设置完成标志为False

        for k in infer_state:  # 遍历每个客户端
            if infer_state[k] < self.baseline[k]:  # 如果推理时间小于基准时间
                r = (self.baseline[k] - infer_state[k]) / self.baseline[k]  # 计算奖励比例
                reward += r  # 增加奖励
            else:  # 如果推理时间大于基准时间
                r = (infer_state[k] - self.baseline[k]) / infer_state[k]  # 计算惩罚比例
                reward -= r  # 减小奖励

        return reward, max_infer_time, done  # 返回奖励、最大时间和完成标志

    # 该函数将动作转换为分层信息
    def action_to_layer(self, action):
        """
        将动作转换为分层信息
        :param action:
        :return:
        """
        split_layer = []  # 初始化分层列表
        prev_idx = 0
        for i in range(len(action)):
            idx = int(prev_idx + action[i] * (config.model_len - 1 - prev_idx))
            if idx >= config.model_len - 1:
                idx = config.model_len - 2
            split_layer.append(idx)
            prev_idx = idx + 1

        split_layer.append(config.model_len - 1)  # 添加最后一层
        return split_layer  # 返回分层列表


class RL_Client(Correspondence):
    def __init__(self, index, ip_address, server_addr, server_port, data_len, model_name, split_layer, model_cfg):
        super(RL_Client, self).__init__(index, ip_address)
        self.criterion = None
        self.optimizer = None
        self.net = None
        self.split_layer = None
        # 客户端IP地址
        self.ip_address = ip_address
        self.data_len = data_len
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = model_name
        self.model_cfg = model_cfg
        # 获取整个模型
        self.uni_net = utils.get_model('Unit', self.model_name, 0, self.device, self.model_cfg)

        logger.info('==> Connecting to Server..')
        # 连接服务器
        self.sock.connect((server_addr, server_port))

    def initialize(self, split_layer):
        """
        初始化函数
        :param split_layer:
        :return:
        """
        self.split_layer = split_layer
        # 获取客户端模型
        self.net = utils.get_model('Client', self.model_name, self.split_layer, self.device, self.model_cfg)
        # 初始化优化器
        self.optimizer = optim.SGD(self.net.parameters(),
                                   lr=config.LR,
                                   momentum=0.9)
        # 定义损失函数
        self.criterion = nn.CrossEntropyLoss()

        # First test network speed
        network_time_start = time.time()
        # 创建网络测试消息
        msg = ['MSG_TEST_NETWORK_SPEED', self.uni_net.cpu().state_dict()]
        self.send_message(self.sock, msg)
        # 接收网络测试消息
        msg = self.recv_message(self.sock, 'MSG_TEST_NETWORK_SPEED')[1]
        network_time_end = time.time()
        # 计算网络速度
        network_speed = (2 * config.model_size * 8) / (network_time_end - network_time_start)  # format is Mbit/s
        # 创建网络速度消息
        msg = ['MSG_TEST_NETWORK_SPEED', self.ip, network_speed]
        self.send_message(self.sock, msg)

    def infer(self, train_loader):
        """
        推理函数,进行训练
        :param train_loader:
        :return:
        """
        # 移动模型到设备
        self.net.to(self.device)
        # 设置为训练模式
        self.net.train()

        s_time_infer = time.time()
        # No offloading
        if self.split_layer == len(model_cfg[self.model_name]) - 1:
            # 遍历训练数据
            for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(train_loader)):
                # 移动数据到设备
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                # 前向传播
                outputs = self.net(inputs)
                # 计算损失
                loss = self.criterion(outputs, targets)
                # 反向传播
                loss.backward()
                # 更新参数
                self.optimizer.step()
                # 如果达到指定迭代次数
                if batch_idx >= config.iteration[self.ip_address] - 1:
                    break
        else:  # Offloading training
            for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(train_loader)):
                # 移动数据到设备
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                # 前向传播
                outputs = self.net(inputs)

                # 创建激活值和标签消息
                msg = ['MSG_LOCAL_ACTIVATIONS_CLIENT_TO_SERVER', outputs.cpu(), targets.cpu()]
                self.send_message(self.sock, msg)

                # Wait receiving server gradients
                gradients = self.recv_message(self.sock)[1].to(self.device)

                outputs.backward(gradients)  # 反向传播
                self.optimizer.step()  # 更新参数

                # 如果达到指定迭代次数
                if batch_idx >= config.iteration[self.ip_address] - 1:
                    break

        e_time_infer = time.time()
        logger.info('Training time: ' + str(e_time_infer - s_time_infer))

        infer_speed = (e_time_infer - s_time_infer) / config.iteration[self.ip_address]
        msg = ['MSG_INFER_SPEED', self.ip, infer_speed]
        self.send_message(self.sock, msg)

    def reinitialize(self, split_layers):
        """
        重新初始化函数
        :param split_layers:
        :return:
        """
        self.initialize(split_layers)

