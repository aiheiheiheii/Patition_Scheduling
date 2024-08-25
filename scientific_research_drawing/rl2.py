import numpy as np


# 定义 sigmoid 函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 定义门控融合函数
def gated_fusion(s_t, s_t_i, W_g, b_g):
    # 将 s_t 和 s_t_i 拼接起来
    concatenated = np.concatenate((s_t, s_t_i), axis=-1)
    # 计算门控值 g
    g = sigmoid(np.dot(concatenated, W_g) + b_g)
    # 计算衍生状态
    s_derived = g * s_t + (1 - g) * s_t_i
    return s_derived


# 初始化状态 s_t
s_t = np.array([0.15, 0.30, 168])  # cpu占用率 15%, 内存占用率 30%, 带宽 168kb/s

# 模拟环境交互后的新状态 s_{t_i}
s_t_i_list = [
    np.array([0.20, 0.25, 150]),  # 例子状态1
    np.array([0.18, 0.28, 160]),  # 例子状态2
    np.array([0.22, 0.27, 155]),  # 例子状态3
    np.array([0.17, 0.29, 165]),  # 例子状态4
    np.array([0.19, 0.26, 158]),  # 例子状态5
    np.array([0.21, 0.24, 162]),  # 例子状态6
    np.array([0.16, 0.31, 170])   # 例子状态7
]

# 初始化门控单元的可学习参数 W_g 和 b_g
W_g = np.random.randn(6)  # 6 表示 s_t 和 s_{t_i} 拼接后的长度
b_g = np.random.randn()

# 设定衍生次数 k
k = 7

# 初始化衍生状态
s_derived_list = []

# 递归地应用状态衍生函数 k 次
s_current = s_t
for i in range(k):
    s_current = gated_fusion(s_current, s_t_i_list[i], W_g, b_g)
    s_derived_list.append(s_current)
    # 打印每次衍生后的状态
    print(f"第{i+1}次衍生状态: CPU占用率={s_current[0]:.2%}, 内存占用率={s_current[1]:.2%}, 带宽={s_current[2]:.2f}kb/s")

