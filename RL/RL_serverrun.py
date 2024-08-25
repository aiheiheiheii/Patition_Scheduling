"""
导入必要的模块和库。
根据配置设置随机种子。
创建RL环境(Env对象)和PPO智能体(PPO对象)。
开始RL训练循环:
  1、遍历每个episode。
  2、在环境中重置状态。
  3、遍历每个时间步:
        从PPO智能体选择动作。
        在环境中执行动作,获取新状态、奖励等信息。
        保存奖励和完成标志到Memory对象中。
        如果达到更新时间步,使用PPO算法更新智能体,并记录结果。
        如果当前episode完成或达到控制更新epoch数量,跳出时间步循环。
该代码的主要作用是实现PPO算法,通过与环境的交互来学习最优策略,以最大化累积奖励。它记录了每个更新epoch的结果,并定期保存智能体模型权重。
"""
import pickle
import torch
import tqdm
import numpy as np
import torch.optim as optim
import sys
import torch.nn as nn
# 添加上级目录到系统路径，以便能够导入项目中的其他模块
sys.path.append('../')
# 从RLEnv模块导入Env类，这是强化学习环境的类
from RLEnv import Env
# 导入配置模块，包含实验的配置参数
import config
# 导入PPO模块，包含PPO智能体的实现
import PPO
from models.model_struct import model_cfg
# 导入日志模块，用于记录日志信息
import logging
# 设置日志格式，包括时间戳、日志记录器名称、日志级别和日志消息
logging.basicConfig(level=logging.INFO, format='%(asc_time)s - %(name)s - %(level_name)s - %(message)s')
# 创建日志记录器对象
logger = logging.getLogger(__name__)


# 如果配置中指定使用随机种子
if config.random:
    # 设置PyTorch随机种子，确保实验的可重复性
    torch.manual_seed(config.random_seed)
    # 设置NumPy随机种子
    np.random.seed(config.random_seed)
    # 记录随机种子
    logger.info('Random seed: {}'.format(config.random_seed))

# 创建Env对象，初始化环境
env = Env(0,
          config.SERVER_ADDR,
          config.SERVER_PORT,
          config.CLIENTS_LIST,
          config.model_name,
          model_cfg,
          config.rl_b
          )

# 设置运行设备为CUDA或CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 创建PPO智能体
# 获取状态维度和动作维度
state_dim = env.state_dim
action_dim = env.action_dim
# 创建Memory对象，用于存储训练过程中的信息
memory = PPO.Memory()
# 创建PPO对象，初始化PPO智能体
ppo = PPO.PPO(state_dim,
              action_dim,
              config.action_std,
              config.rl_lr,
              config.rl_betas,
              config.rl_gamma,
              config.K_epochs,
              config.eps_clip)

# RL训练
logger.info('==> RL Training Start.')  # 记录信息
# 初始化时间步长
time_step = 0
# 初始化更新epoch
update_epoch = 1
# 初始化结果字典
res = {'rewards': [], 'maxtime': [], 'actions': [], 'std': []}

# # 创建IAF模型的优化器和损失函数
# iaf_optimizer = optim.Adam(env.state_derivation.iaf.parameters(), lr=config.iaf_lr)
# iaf_loss_fn = nn.MSELoss()

# 遍历每个episode
for i_episode in tqdm.tqdm(range(1, config.max_episodes + 1)):
    # 初始化完成标志为False
    done = False
    # 如果是第一个episode
    if i_episode == 1:
        # 设置first标志为True
        first = True
        # 重置环境，获取初始状态
        state = env.reset(done, first)
    else:
        # 否则
        first = False
        # 重置环境，获取初始状态
        state = env.reset(done, first)
        print("State shape: ", state.shape)

        # 遍历每个时间步
        for t in range(config.max_time_steps):
            # 更新时间步长
            time_step += 1
            # 从PPO智能体选择动作、动作均值和标准差
            action, action_mean, std = ppo.select_action(state, memory)
            # 在环境中执行动作，获取新状态、奖励、最大时间和完成标志
            state, reward, maxtime, done = env.step(action, done)

            # # 更新IAF模型的参数
            # env.update_state_derivation(iaf_optimizer, iaf_loss_fn)

            # 记录当前奖励和最大时间
            logger.info('Current reward: ' + str(reward))
            logger.info('Current maxtime: ' + str(maxtime))

            # 保存奖励和完成标志
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            # 如果达到更新时间步
            if time_step % config.update_timestep == 0:
                # 使用PPO算法更新智能体
                ppo.update(memory)
                # 记录更新epoch
                logger.info('Agent has been updated: ' + str(update_epoch))
                # 如果超过探索次数
                if update_epoch > config.exploration_times:
                    # 衰减探索
                    ppo.explore_decay(update_epoch - config.exploration_times)

                # 清空Memory对象
                memory.clear_memory()
                # 重置时间步长
                time_step = 0
                # 更新epoch
                update_epoch += 1

                # 记录每个更新epoch的结果
                with open('RL_res.pkl', 'wb') as f:
                    # 打开文件
                    pickle.dump(res, f)
                    # 序列化结果字典

                # 保存智能体每个更新epoch的模型权重
                torch.save(ppo.policy.state_dict(), './PPO.pth')

            # 添加奖励、最大时间、动作和动作均值、标准差到结果列表
            res['rewards'].append(reward)
            res['maxtime'].append(maxtime)
            res['actions'].append((action, action_mean))
            res['std'].append(std)

            # 如果当前episode完成
            if done:
                # 跳出时间步循环
                break

            # 如果达到控制更新epoch数量，停止训练
            if update_epoch > 50:
                break
