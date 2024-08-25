import numpy as np


def sigmoid(x):
    """sigmoid激活函数"""
    return 1 / (1 + np.exp(-x))


def state_derivation(s_t, s_t_list, W_g, b_g):
    """
    状态衍生函数

    参数:
    s_t: 原始状态,形状为(3,)的numpy数组
    s_t_list: 与环境交互后得到的新状态列表,每个元素形状为(3,)的numpy数组
    W_g: 门控单元的权重参数,形状为(1, 6)的numpy数组
    b_g: 门控单元的偏置参数,标量

    返回:
    s_t_derived: 衍生状态列表,每个元素形状为(3,)的numpy数组
    """
    s_t_derived = []
    for s_t_i in s_t_list:
        # 拼接原始状态和新状态
        concat_state = np.concatenate((s_t, s_t_i), axis=0)

        # 计算门控系数
        g = sigmoid(np.dot(W_g, concat_state) + b_g)

        # 融合原始状态和新状态
        s_t_i_derived = g * s_t + (1 - g) * s_t_i

        s_t_derived.append(s_t_i_derived)

    return s_t_derived


# 原始状态
s_t = np.array([0.15, 0.3, 168])

# 与环境交互后得到的新状态列表
s_t_list = [
    np.array([0.2, 0.35, 200]),
    np.array([0.18, 0.32, 190]),
    np.array([0.22, 0.38, 210]),
    np.array([0.16, 0.28, 180]),
    np.array([0.25, 0.4, 220]),
    np.array([0.2, 0.36, 200]),
    np.array([0.18, 0.33, 195])
]

# 门控单元的参数
W_g = np.random.randn(1, 6)
b_g = np.random.randn()

# 衍生状态
s_t_derived = state_derivation(s_t, s_t_list, W_g, b_g)

# 打印衍生状态
for i, s in enumerate(s_t_derived):
    print(f"第{i + 1}次衍生状态: CPU占用率={s[0]:.2f}, 内存占用率={s[1]:.2f}, 带宽={s[2]:.2f}kb/s")