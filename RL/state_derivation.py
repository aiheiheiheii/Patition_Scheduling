"""
StateDerivation类的构造函数接受两个参数:状态维度state_dim和时间窗口大小window_size。
它初始化了一个状态缓存区state_buffer,用于存储之前一段时间内的状态。

derive_state方法接受当前状态作为输入,并执行以下步骤
    将当前状态添加到状态缓存区state_buffer中。
    如果缓存区长度超过了时间窗口大小,则删除最早的状态。
    计算状态缓存区中所有状态的平均值state_mean。
    计算状态缓存区中相邻状态之间的差值state_diff。
    计算状态变化量的平均值state_diff_mean。
    将当前状态、平均状态和平均状态变化量拼接起来,形成衍生后的状态derived_state。
"""
import numpy as np

class StateDerivation:
    def __init__(self, state_dim, window_size):
        self.state_dim = state_dim
        self.window_size = window_size
        self.state_buffer = []

    def derive_state(self, state):
        # 将当前状态添加到状态缓存区
        self.state_buffer.append(state)

        # 如果缓存区长度超过窗口大小,则删除最早的状态
        if len(self.state_buffer) > self.window_size:
            self.state_buffer.pop(0)

        # 计算状态缓存区中所有状态的平均值
        state_mean = np.mean(self.state_buffer, axis=0)

        # 计算状态缓存区中相邻状态之间的差值
        state_diff = np.diff(self.state_buffer, axis=0)

        # 计算状态变化量的平均值
        if len(state_diff) > 0:
            state_diff_mean = np.mean(state_diff, axis=0)
        else:
            state_diff_mean = np.zeros(self.state_dim)

        # 将当前状态、平均状态和平均状态变化量拼接起来
        derived_state = np.concatenate((state, state_mean, state_diff_mean), axis=0)

        return derived_state