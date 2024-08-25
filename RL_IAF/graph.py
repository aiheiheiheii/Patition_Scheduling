import matplotlib.pyplot as plt
import numpy as np

# 创建数据
x = np.linspace(0, 500, 500)  # 生成0到500的500个点
y1 = np.clip(1 - 0.002 * (x - 100), a_min=0, a_max=1)  # 模拟G1
y2 = np.clip(0.8 - 0.002 * (x - 150), a_min=0, a_max=0.8)  # 模拟G2
y3 = np.clip(0.4 - 0.001 * (x - 200), a_min=0, a_max=0.4)  # 模拟G3

# 绘制折线图
plt.figure(figsize=(10, 5))
plt.plot(x, y1, label='G1', color='blue')
plt.plot(x, y2, label='G2', color='green')
plt.plot(x, y3, label='G3', color='red')

# 添加图例
plt.legend()

# 添加网格线
plt.grid(True)

# 设置图表标题和轴标签
plt.title('Action vs. Truncated Round')
plt.xlabel('Truncated Round')
plt.ylabel('Action')

# 显示图表
plt.show()