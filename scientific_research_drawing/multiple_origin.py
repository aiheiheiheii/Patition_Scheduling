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


# 绘制对比图的函数
def plot_comparison(num_episodes_list, base_state):
    plt.figure(figsize=(12, 8))

    for num_episodes in num_episodes_list:
        states = generate_states(num_episodes, base_state)  # 生成初始状态

        y_min = min(states) - 1  # 确定 y 轴的最小值
        y_max = max(states) + 1  # 确定 y 轴的最大值

        # 绘制每个训练轮次的图
        plt.subplot(len(num_episodes_list), 1, num_episodes_list.index(num_episodes) + 1)
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

    plt.tight_layout()
    plt.show()


# 主函数
def main():
    np.random.seed(42)  # 设置随机种子
    base_state = 5  # 基准状态值
    num_episodes_list = [20, 50, 100]  # 多个训练轮次数列表

    plot_comparison(num_episodes_list, base_state)  # 绘制对比图


# 执行主函数
if __name__ == "__main__":
    main()
