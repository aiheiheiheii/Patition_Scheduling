import torch.nn as nn


# 定义 VGG5 类，继承自 torch.nn.Module
class VGG5(nn.Module):
    """
    构造函数，初始化 VGG 网络
    可以根据传入的配置cfg和分割层split_layer来决定在客户端或服务器上构建网络的哪一部分
    """
    def __init__(self, location, vgg_name, split_layer, cfg):
        super(VGG5, self).__init__()  # 调用父类的构造函数
        # 断言 split_layer 小于 cfg[vgg_name] 的长度，确保分割层是有效的
        assert split_layer < len(cfg[vgg_name])
        self.split_layer = split_layer  # 设置分割层
        self.location = location  # 设置网络位置（'Server' 或 'Client'）
        self.features, self.denses = self._make_layers(cfg[vgg_name])  # 创建网络层
        self._initialize_weights()  # 初始化网络权重

    def forward(self, x):
        """
        前向传播函数，定义数据通过网络的正向计算过程
        :param x:
        :return:
        """
        # 如果有特征层（卷积层和池化层）
        if len(self.features) > 0:
            out = self.features(x)
        else:
            out = x
        # 如果有密集层（全连接层）
        if len(self.denses) > 0:
            out = out.view(out.size(0), -1)  # 展平特征图
            out = self.denses(out)
        return out  # 返回网络输出

    def _make_layers(self, cfg):
        """
        根据配置信息创建网络层
        :param cfg:
        :return:
        """
        features = []  # 特征层列表
        denses = []  # 密集层列表
        # 根据网络位置选择配置
        if self.location == 'Server':
            cfg = cfg[:self.split_layer + 1]  # 服务器部分的配置（分割层之后）
        if self.location == 'Client':
            cfg = cfg[:self.split_layer + 1]  # 客户端部分的配置（分割层之前）
        if self.location == 'Unit':
            pass  # 如果是完整模型，不做处理

        # 遍历配置中的每层
        for x in cfg:
            in_channels, out_channels = x[1], x[2]  # 获取输入和输出通道数
            kernel_size = x[3]  # 获取卷积核大小
            # 根据层类型添加不同的层
            if x[0] == 'M':
                features += [nn.MaxPool2d(kernel_size=kernel_size, stride=2)]  # 添加池化层
            if x[0] == 'D':
                denses += [nn.Linear(in_channels, out_channels)]  # 添加全连接层
            if x[0] == 'C':
                features += [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1),  # 添加卷积层
                             nn.BatchNorm2d(out_channels),  # 添加批量归一化层
                             nn.ReLU(inplace=True)]  # 添加ReLU激活函数

        return nn.Sequential(*features), nn.Sequential(*denses)  # 返回特征层和密集层的序列

    def _initialize_weights(self):
        """
        初始化网络权重，是训练开始前的标准步骤。
        这个类可能是为了在联邦学习环境中使用，
        其中模型的一部分在客户端上执行，而另一部分在服务器上执行。
        :return:
        """
        # 遍历模块中的所有子模块
        for m in self.modules():
            # 如果是卷积层
            if isinstance(m, nn.Conv2d):
                # 使用 RELU 正则化初始化权重
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # 如果有偏置项，初始化为常数 0
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # 如果是批量归一化层
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # 如果是全连接层
            elif isinstance(m, nn.Linear):
                # 初始化权重为正态分布
                nn.init.normal_(m.weight, 0, 0.01)
                # 初始化偏置为常数 0
                nn.init.constant_(m.bias, 0)
