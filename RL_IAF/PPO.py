import torch
import torch.nn as nn
# 从PyTorch分布中导入多元正态分布
from torch.distributions import MultivariateNormal

import logging

# 设置日志格式
logging.basicConfig(level=logging.INFO, format='%(asc_time)s - %(name)s - %(level_name)s - %(message)s')
logger = logging.getLogger(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 定义Memory类，用于存储经验
class Memory:
    def __init__(self):
        self.actions = []  # 初始化动作列表
        self.states = []  # 初始化状态列表
        self.log_probs = []  # 初始化对数概率列表
        self.rewards = []  # 初始化奖励列表
        self.is_terminals = []  # 初始化是否结束列表

    def clear_memory(self):
        del self.actions[:]  # 清空动作列表
        del self.states[:]  # 清空状态列表
        del self.log_probs[:]  # 清空对数概率列表
        del self.rewards[:]  # 清空奖励列表
        del self.is_terminals[:]  # 清空是否结束列表


class ActorCritic(nn.Module):
    """
    ActorCritic类的构造函数,继承自nn.Module
    """
    def __init__(self, state_dim, action_dim, action_std):
        super(ActorCritic, self).__init__()  # 调用父类构造函数
        # action mean range -1 to 1  输出动作的均值
        # 由于衍生后的状态维度是原始状态维度的3倍(当前状态、平均状态和平均状态变化量),将Actor网络和Critic网络的输入层大小乘以3,以匹配衍生后的状态维度。
        self.actor = nn.Sequential(  # 定义Actor网络
            nn.Linear(state_dim * 3, 64),  # 全连接层
            nn.Tanh(),  # Tanh激活函数
            nn.Linear(64, 32),  # 全连接层
            nn.Tanh(),  # Tanh激活函数
            nn.Linear(32, action_dim),  # 全连接层
            nn.Sigmoid()  # Sigmoid激活函数
        )

        # critic  输出状态值
        self.critic = nn.Sequential(  # 定义Critic网络
            nn.Linear(state_dim * 3, 64),  # 全连接层
            nn.Tanh(),  # Tanh激活函数
            nn.Linear(64, 32),  # 全连接层
            nn.Tanh(),  # Tanh激活函数
            nn.Linear(32, 1)  # 全连接层
        )

        self.state_dim = state_dim  # 状态维度
        self.action_dim = action_dim  # 动作维度
        self.init_action_std = action_std  # 初始动作标准差
        self.action_std = action_std  # 动作标准差
        self.action_var = torch.full((action_dim,), action_std * action_std).to(device)  # 动作方差

    def forward(self):
        """
        forward函数,抛出未实现的错误,因为ActorCritic不直接使用forward
        :return:
        """
        raise NotImplementedError

    def act(self, state, memory):
        """
        根据当前状态生成动作
        使用Actor网络计算动作均值,创建多元正态分布,采样动作,计算动作的对数概率
        :param state:
        :param memory:
        :return:
        """
        action_mean = self.actor(state)  # 计算动作均值
        cov_mat = torch.diag(self.action_var).to(device)  # 计算协方差矩阵
        logger.info(' Current action mean: ' + str(action_mean))  # 记录当前动作均值

        dist = MultivariateNormal(action_mean, cov_mat)  # 创建多元正态分布
        action = dist.sample()  # 采样动作
        logger.info('Current action sample: ' + str(action))  # 记录当前动作
        action_log_prob = dist.log_prob(action)  # 计算动作对数概率

        memory.states.append(state)  # 将状态存入memory
        memory.actions.append(action)  # 将动作存入memory
        memory.log_probs.append(action_log_prob)  # 将对数概率存入memory

        return action.detach(), action_mean.detach()  # 返回动作和动作均值

    def evaluate(self, state, action):
        """
        评估给定状态和动作
        使用Actor网络计算动作均值,创建多元正态分布
        :param state:
        :param action:
        :return:
        """
        action_mean = self.actor(state)  # 计算动作均值

        action_var = self.action_var.expand_as(action_mean)  # 扩展动作方差
        cov_mat = torch.diag_embed(action_var).to(device)  # 计算协方差矩阵

        dist = MultivariateNormal(action_mean, cov_mat)  # 创建多元正态分布

        action_log_probs = dist.log_prob(action)  # 计算动作对数概率
        dist_entropy = dist.entropy()  # 计算分布熵
        state_value = self.critic(state)  # 计算状态价值

        return action_log_probs, torch.squeeze(state_value), dist_entropy  # 返回动作对数概率、状态价值和分布熵

    def std_decay(self, epoch):
        """
        根据当前轮数衰减动作标准差和方差
        :param epoch:
        :return:
        """
        self.action_std = self.init_action_std * (0.9 ** epoch)  # 计算当前动作标准差
        self.action_var = torch.full((self.action_dim,), self.action_std * self.action_std).to(device)  # 更新动作方差


class PPO:
    """
    PPO类的构造函数,存储各种超参数
    """
    def __init__(self, state_dim, action_dim, action_std, lr, betas, gamma, k_epochs, eps_clip, env):
        self.env = env
        self.lr = lr  # 学习率
        self.betas = betas  # Adam优化器的beta参数
        self.gamma = gamma  # 折扣因子
        self.eps_clip = eps_clip  # PPO裁剪参数
        self.K_epochs = k_epochs  # PPO更新次数

        self.policy = ActorCritic(state_dim, action_dim, action_std).to(device)  # 创建Actor-Critic网络
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)  # 创建优化器

        self.policy_old = ActorCritic(state_dim, action_dim, action_std).to(device)  # 创建旧Actor-Critic网络
        self.policy_old.load_state_dict(self.policy.state_dict())  # 复制参数

        self.MseLoss = nn.MSELoss()  # 创建均方误差损失函数

    def explore_decay(self, epoch):
        """
        衰减当前和旧策略的动作标准差
        :param epoch:
        :return:
        """
        self.policy.std_decay(epoch)  # 更新当前Actor-Critic网络动作标准差
        self.policy_old.std_decay(epoch)  # 更新旧Actor-Critic网络动作标准差

    def select_action(self, state, memory):
        """
        使用旧策略选择动作
        将状态转换为张量,使用旧策略的act函数生成动作
        :param state:
        :param memory:
        :return:
        """
        derived_state = self.env.get_derived_state(state)
        derived_state = torch.FloatTensor(derived_state.reshape(1, -1)).to(device)  # 转换状态为tensor
        actions = self.policy_old.act(derived_state, memory)  # 使用旧Actor-Critic网络选择动作
        stds = self.policy_old.action_var  # 获取旧动作方差
        return actions[0].cpu().data.numpy().flatten(), actions[
            1].cpu().data.numpy().flatten(), stds.cpu().data.numpy().flatten()  # 返回动作、动作均值和动作方差

    def exploit(self, state):
        """
        使用当前策略进行开发
        将状态转换为张量,使用当前策略的Actor网络计算动作均值
        :param state:
        :return:
        """
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)  # 转换状态为tensor
        action_mean = self.policy.actor(state)  # 使用当前Actor-Critic网络计算动作均值
        return action_mean[0].cpu().data.numpy().flatten()  # 返回动作均值

    def update(self, memory):
        # 使用蒙特卡洛方法估计奖励:  从后往前遍历奖励和终止标志, 计算折扣奖励,并插入到rewards列表的开头
        rewards = []  # 初始化奖励列表
        discounted_reward = 0  # 初始化折扣奖励
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):  # 反向遍历奖励和是否结束列表
            if is_terminal:  # 如果是终止状态
                discounted_reward = 0  # 重置折扣奖励
            discounted_reward = reward + (self.gamma * discounted_reward)  # 计算折扣奖励
            rewards.insert(0, discounted_reward)  # 将折扣奖励插入列表头部

        # 对奖励进行归一化处理:
        rewards = torch.tensor(rewards).to(device)  # 转换奖励为tensor
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)  # 归一化奖励

        # 将状态、动作和对数概率的列表转换为张量,并进行维度压缩和分离
        old_states = torch.squeeze(torch.stack(memory.states).to(device), 1).detach()  # 转换状态列表为tensor
        old_actions = torch.squeeze(torch.stack(memory.actions).to(device), 1).detach()  # 转换动作列表为tensor
        old_log_probs = torch.squeeze(torch.stack(memory.log_probs), 1).to(device).detach()  # 转换对数概率列表为tensor

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # 使用当前策略评估旧的状态和动作,得到对数概率、状态值和分布熵
            log_probs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)  # 计算新动作对数概率、状态价值和分布熵

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(log_probs - old_log_probs.detach())  # 计算新旧策略比率

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()  # 计算优势函数
            surr1 = ratios * advantages  # 计算surrogate loss第一项
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages  # 计算surrogate loss第二项
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy  # 计算总损失

            # take gradient step
            self.optimizer.zero_grad()  # 梯度清零
            loss.mean().backward()  # 反向传播
            self.optimizer.step()  # 更新参数

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())  # 将新策略复制到旧策略
