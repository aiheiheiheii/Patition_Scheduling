from matplotlib import pyplot as plt
import logging
logging.getLogger('matplotlib').propagate = False

def plot_times(inference_times, transmit_times, client_bandwidths, round_times):
    plot_inference_times(inference_times)
    plot_transmit_times(transmit_times)
    plot_transmit_bandwidths(transmit_times)
    plot_total_round_times(round_times)


def plot_inference_times(inference_times):
    max_iterations = max(len(times) for times in inference_times.values())
    plt.figure()
    for client, times in inference_times.items():
        plt.plot(range(1, len(times) + 1), times, marker='o', label=f'{client} Inference')

    plt.title('Inference Time for All Clients')
    plt.xlabel('Round')
    plt.ylabel('Time (seconds)')
    plt.xticks(range(1, max_iterations + 1))
    plt.grid(True)
    plt.legend()
    plt.savefig('inference_times.png')
    plt.show()


def plot_transmit_times(transmit_times):
    max_iterations = max(len(times) for times in transmit_times.values())
    plt.figure()
    for client, times in transmit_times.items():
        plt.plot(range(1, len(times) + 1), times, marker='x', label=f'{client} Transmit')

    plt.title('Transmit Time for All Clients')
    plt.xlabel('Round')
    plt.ylabel('Time (seconds)')
    plt.xticks(range(1, max_iterations + 1))
    plt.grid(True)
    plt.legend()
    plt.savefig('transmit_times.png')
    plt.show()


def plot_transmit_bandwidths(client_bandwidths):
    max_iterations = max(len(bandwidths) for bandwidths in client_bandwidths.values())
    plt.figure()
    for client, bandwidths in client_bandwidths.items():
        plt.plot(range(1, len(bandwidths) + 1), bandwidths, marker='o', label=f'{client} Bandwidth')

    plt.title('Bandwidth for All Clients')
    plt.xlabel('Round')
    plt.ylabel('Bandwidth (kb/s)')
    plt.xticks(range(1, max_iterations + 1))
    plt.grid(True)
    plt.legend()
    plt.savefig('bandwidths.png')
    plt.show()


def plot_transmit_times_and_bandwidths(transmit_times, client_bandwidths):
    max_iterations = max(max(len(times) for times in transmit_times.values()),
                         max(len(bandwidths) for bandwidths in client_bandwidths.values()))

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Transmit Time (seconds)', color=color)
    for client, times in transmit_times.items():
        ax1.plot(range(1, len(times) + 1), times, marker='x', label=f'{client} Transmit Time', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # 共享 x 轴
    color = 'tab:blue'
    ax2.set_ylabel('Bandwidth (kb/s)', color=color)
    for client, bandwidths in client_bandwidths.items():
        ax2.plot(range(1, len(bandwidths) + 1), bandwidths, marker='o', linestyle='--', label=f'{client} Bandwidth',
                 color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # 调整布局以防止标签重叠
    plt.title('Transmit Time and Bandwidth for All Clients')
    fig.legend(loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    plt.xticks(range(1, max_iterations + 1))
    plt.grid(True)
    plt.savefig('transmit_times_and_bandwidths.png')
    plt.show()


def plot_total_round_times(round_times):
    plt.figure()
    plt.plot(range(1, len(round_times) + 1), round_times, marker='o', color='r')
    plt.title('Total Execution Time per Round')
    plt.xlabel('Round')
    plt.ylabel('Time (seconds)')
    plt.xticks(range(1, len(round_times) + 1))
    plt.grid(True)
    plt.savefig('total_round_times.png')
    plt.show()
