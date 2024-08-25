import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


# 生成初始状态的函数
def generate_states(num_episodes, base_state):
    states = np.random.normal(loc=base_state, scale=0.1, size=num_episodes)  # 生成以 base_state 为中心的平稳数据
    upper_spikes = np.random.choice([0, 1], size=num_episodes, p=[0.7, 0.3])  # 生成上波动，概率为 0.3
    lower_spikes = np.random.choice([0, 1], size=num_episodes, p=[0.7, 0.3])  # 生成下波动，概率为 0.3
    states += upper_spikes * np.random.normal(loc=1.5, scale=0.5, size=num_episodes)  # 为上波动部分添加噪声
    states -= lower_spikes * np.random.normal(loc=1.5, scale=0.5, size=num_episodes)  # 为下波动部分添加噪声
    return states  # 返回生成的状态数据


# 生成衍生状态的函数
def generate_derivative_states(num_episodes, states):
    derivative_states_x = []  # 初始化衍生状态的 x 坐标列表
    derivative_states_y = []  # 初始化衍生状态的 y 坐标列表
    for i in range(num_episodes):  # 遍历每个训练周期
        num_derivative_points = np.random.randint(1, 11)  # 随机生成衍生点的数量
        for _ in range(num_derivative_points):  # 生成对应数量的衍生点
            derivative_states_x.append(i)  # 添加衍生点的 x 坐标
            derivative_states_y.append(states[i] + np.random.normal(loc=0, scale=0.5))  # 添加衍生点的 y 坐标，带有噪声
    return derivative_states_x, derivative_states_y  # 返回衍生状态的坐标列表


# 绘制对比图的函数
def plot_comparison(num_episodes_list, base_state, target_state, num_episodes_derivative):
    plt.figure(figsize=(12, 12))

    for num_episodes in num_episodes_list:
        states = generate_states(num_episodes, base_state)  # 生成初始状态

        y_min = min(states) - 1  # 确定 y 轴的最小值
        y_max = max(states) + 1  # 确定 y 轴的最大值

        # 绘制每个训练轮次的图
        plt.subplot(len(num_episodes_list) + 1, 1, num_episodes_list.index(num_episodes) + 1)
        plt.scatter(range(num_episodes), states, c='blue', alpha=0.6, edgecolors='w', linewidth=0.5,
                    label='Original State')
        plt.title(f'Original States for {num_episodes} Episodes')
        plt.xlabel('Episode')
        plt.ylabel('State')
        plt.ylim(y_min, y_max)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.gca().yaxis.set_ticklabels([])
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        if num_episodes_list.index(num_episodes) == 0:
            plt.legend()  # 只在第一张图上显示图例

    # 为衍生状态绘制单独的图
    states_derivative = generate_states(num_episodes_derivative, base_state)  # 生成初始状态
    derivative_states_x, derivative_states_y = generate_derivative_states(num_episodes_derivative,
                                                                          states_derivative)  # 生成衍生状态

    y_min_derivative = min(min(states_derivative), min(derivative_states_y)) - 1  # 确定 y 轴的最小值
    y_max_derivative = max(max(states_derivative), max(derivative_states_y)) + 1  # 确定 y 轴的最大值

    plt.subplot(len(num_episodes_list) + 1, 1, len(num_episodes_list) + 1)
    plt.scatter(range(num_episodes_derivative), states_derivative, c='blue', alpha=0.6, edgecolors='w', linewidth=0.5,
                label='Original State')
    plt.scatter(derivative_states_x, derivative_states_y, c='red', alpha=0.3, edgecolors='w', linewidth=0.5,
                label='Derivative State')
    plt.title(f'Original and Derivative States for {num_episodes_derivative} Episodes')
    plt.xlabel('Episode')
    plt.ylabel('State')
    plt.ylim(y_min_derivative, y_max_derivative)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.gca().yaxis.set_ticklabels([])
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()  # 显示图例

    plt.tight_layout()
    plt.show()


# 主函数
def main():
    np.random.seed(42)  # 设置随机种子
    base_state = 5  # 基准状态值
    target_state = 5  # 目标状态值
    num_episodes_list = [20, 50, 100]  # 多个训练轮次数列表
    num_episodes_derivative = 20  # 用于状态衍生的训练轮次数

    plot_comparison(num_episodes_list, base_state, target_state, num_episodes_derivative)  # 绘制对比图


# 执行主函数
if __name__ == "__main__":
    main()
