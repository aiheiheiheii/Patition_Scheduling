import matplotlib.pyplot as plt  # 导入 matplotlib 库用于绘图
import numpy as np  # 导入 numpy 库用于数值计算
from matplotlib.ticker import MaxNLocator  # 从 matplotlib.ticker 导入 MaxNLocator 用于 x 轴整数刻度


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


# 找出最优状态的函数
def find_optimal_states(num_episodes, states, derivative_states_x, derivative_states_y, target_state):
    optimal_states_x = []  # 初始化最优状态的 x 坐标列表
    optimal_states_y = []  # 初始化最优状态的 y 坐标列表
    for i in range(num_episodes):  # 遍历每个训练周期
        original_point = states[i]  # 获取当前周期的原始点
        derivative_points = [y for x, y in zip(derivative_states_x, derivative_states_y) if x == i]  # 获取当前周期的所有衍生点
        all_points = [original_point] + derivative_points  # 合并原始点和衍生点
        optimal_point = min(all_points, key=lambda y: abs(y - target_state))  # 找出距离目标状态最近的点
        optimal_states_x.append(i)  # 添加最优点的 x 坐标
        optimal_states_y.append(optimal_point)  # 添加最优点的 y 坐标
    return optimal_states_x, optimal_states_y  # 返回最优状态的坐标列表


# 绘制原始状态图的函数
def plot_states(num_episodes, states, y_min, y_max):
    plt.figure(figsize=(10, 6))  # 设置绘图窗口大小
    # 绘制原始状态点
    plt.scatter(range(num_episodes), states, c='blue', alpha=0.6, edgecolors='w', linewidth=0.5, label='Original State')
    plt.title('Scatter Plot of Original Environment States over Training Episodes')  # 设置标题
    plt.xlabel('Episode')  # 设置 x 轴标签
    plt.ylabel('State')  # 设置 y 轴标签
    plt.ylim(y_min, y_max)  # 设置 y 轴范围
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # 设置 x 轴刻度为整数
    plt.gca().yaxis.set_ticklabels([])  # 隐藏 y 轴刻度标签
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # 显示网格
    plt.legend()  # 显示图例
    plt.show()  # 显示图形


# 绘制所有状态图的函数
def plot_all_states(num_episodes, states, derivative_states_x, derivative_states_y, optimal_states_x, optimal_states_y, y_min, y_max):
    plt.figure(figsize=(10, 6))  # 设置绘图窗口大小
    # 绘制原始状态点
    plt.scatter(range(num_episodes), states, c='blue', alpha=0.6, edgecolors='w', linewidth=0.5, label='Original State')
    # 绘制衍生状态点
    plt.scatter(derivative_states_x, derivative_states_y, c='red', alpha=0.3, edgecolors='w', linewidth=0.5, label='Derivative State')
    # 绘制最优状态点
    plt.scatter(optimal_states_x, optimal_states_y, c='green', marker='*', s=100, edgecolors='w', linewidth=0.5, label='Optimal State')
    # 设置标题
    plt.title('Scatter Plot of Environment States, Derivative States, and Optimal States over Training Episodes')
    plt.xlabel('Episode')  # 设置 x 轴标签
    plt.ylabel('State')  # 设置 y 轴标签
    plt.ylim(y_min, y_max)  # 设置 y 轴范围
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # 设置 x 轴刻度为整数
    plt.gca().yaxis.set_ticklabels([])  # 隐藏 y 轴刻度标签
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # 显示网格
    plt.legend()  # 显示图例
    plt.show()  # 显示图形


# 主函数
def main():
    np.random.seed(42)  # 设置随机种子
    num_episodes = 20  # 模拟训练次数
    base_state = 5  # 基准状态值
    target_state = 5  # 目标状态值

    states = generate_states(num_episodes, base_state)  # 生成初始状态
    derivative_states_x, derivative_states_y = generate_derivative_states(num_episodes, states)  # 生成衍生状态
    optimal_states_x, optimal_states_y = find_optimal_states(num_episodes, states, derivative_states_x, derivative_states_y, target_state)  # 找出最优状态

    y_min = min(min(states), min(derivative_states_y), min(optimal_states_y)) - 1  # 确定 y 轴的最小值
    y_max = max(max(states), max(derivative_states_y), max(optimal_states_y)) + 1  # 确定 y 轴的最大值

    plot_states(num_episodes, states, y_min, y_max)  # 绘制原始状态图
    plot_all_states(num_episodes, states, derivative_states_x, derivative_states_y, optimal_states_x, optimal_states_y, y_min, y_max)  # 绘制所有状态图


# 执行主函数
if __name__ == "__main__":
    main()
