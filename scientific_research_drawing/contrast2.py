import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

# 生成初始状态的函数
def generate_states(num_episodes, base_state, prev_states=None):
    if prev_states is None:
        prev_states = []
    additional_episodes = num_episodes - len(prev_states)
    if additional_episodes > 0:
        additional_states = np.random.normal(loc=base_state, scale=0.1, size=additional_episodes)
        upper_spikes = np.random.choice([0, 1], size=additional_episodes, p=[0.7, 0.3])
        lower_spikes = np.random.choice([0, 1], size=additional_episodes, p=[0.7, 0.3])
        additional_states += upper_spikes * np.random.normal(loc=1.5, scale=0.5, size=additional_episodes)
        additional_states -= lower_spikes * np.random.normal(loc=1.5, scale=0.5, size=additional_episodes)
        states = np.concatenate((prev_states, additional_states))
    else:
        states = prev_states
    return states

# 生成衍生状态的函数
def generate_derivative_states(num_episodes, states, min_derivative_points_first_episode=5):
    derivative_states_x = []
    derivative_states_y = []

    for i in range(num_episodes):
        if i == 0:
            num_derivative_points = np.random.randint(min_derivative_points_first_episode, 11)
        else:
            num_derivative_points = np.random.randint(0, 11)

        for _ in range(num_derivative_points):
            derivative_states_x.append(i)
            derivative_states_y.append(states[i] + np.random.normal(loc=0, scale=0.8))
    return derivative_states_x, derivative_states_y

# 绘制对比图的函数
def plot_comparison(num_episodes_list, base_state, target_state, num_episodes_derivative):
    plt.figure(figsize=(12, 12))

    # 初始化状态数据列表
    all_states = []

    for idx, num_episodes in enumerate(num_episodes_list):
        if idx == 0:
            states = generate_states(num_episodes, base_state)
        else:
            states = generate_states(num_episodes, base_state, all_states[-1])
        all_states.append(states)

        y_min = min(states) - 1
        y_max = max(states) + 1

        plt.subplot(len(num_episodes_list) + 1, 1, idx + 1)
        plt.scatter(range(num_episodes), states, c='blue', alpha=0.6, edgecolors='w', linewidth=0.5,
                    label='Original State')
        plt.title(f'Original States for {num_episodes} Episodes')
        plt.xlabel('Episode')
        plt.ylabel('State')
        plt.xlim(-1, num_episodes)
        plt.ylim(y_min, y_max)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.gca().set_xticks(np.arange(0, num_episodes, step=max(1, num_episodes // 10)))
        plt.gca().set_xticks(list(plt.gca().get_xticks()) + [num_episodes - 1])
        plt.gca().yaxis.set_ticklabels([])
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        if idx == 0:
            plt.legend()

    states_derivative = all_states[0]
    derivative_states_x, derivative_states_y = generate_derivative_states(num_episodes_derivative, states_derivative)

    y_min_derivative = min(min(states_derivative), min(derivative_states_y)) - 1
    y_max_derivative = max(max(states_derivative), max(derivative_states_y)) + 1

    plt.subplot(len(num_episodes_list) + 1, 1, len(num_episodes_list) + 1)
    plt.scatter(range(num_episodes_derivative), states_derivative, c='blue', alpha=0.6, edgecolors='w', linewidth=0.5,
                label='Original State')
    plt.scatter(derivative_states_x, derivative_states_y, c='red', alpha=0.3, edgecolors='w', linewidth=0.5,
                label='Derivative State')
    plt.title(f'Original and Derivative States for {num_episodes_derivative} Episodes')
    plt.xlabel('Episode')
    plt.ylabel('State')
    plt.xlim(-1, num_episodes_derivative)
    plt.ylim(y_min_derivative, y_max_derivative)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.gca().set_xticks(np.arange(0, num_episodes_derivative, step=max(1, num_episodes_derivative // 10)))
    plt.gca().set_xticks(list(plt.gca().get_xticks()) + [num_episodes_derivative - 1])
    plt.gca().yaxis.set_ticklabels([])
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()

    plt.tight_layout()
    plt.show()

# 主函数
def main():
    base_state = 5
    target_state = 5
    num_episodes_list = [20, 50, 100]
    num_episodes_derivative = 20

    plot_comparison(num_episodes_list, base_state, target_state, num_episodes_derivative)

if __name__ == "__main__":
    main()
