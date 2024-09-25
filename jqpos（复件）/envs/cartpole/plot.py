import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
data = pd.read_csv('actiona.csv')

# 提取第一列、第二列、第三列数据
col1 = data.iloc[:, 0].values
col2 = data.iloc[:, 1].values
# col3 = data.iloc[:, 2].values

# 绘制折线图
plt.plot(col1, label='action', c='blue')
plt.plot(col2, label='qvel',c='red')
# plt.plot(col3, label='qpos', c='green')

# 设置图例、标题、坐标轴标签
plt.legend()
# plt.title('Data Visualization')
plt.xlabel('step')
plt.ylabel('COM position (m)')

# 显示图形
plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
#
# x = np.linspace(-6*np.pi, 6*np.pi, 1000)
# y = np.mod( x  , 2*np.pi) - np.pi
#
# plt.plot(x, -y)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('y = x % (2*pi)')
# plt.grid(True)
# plt.show()