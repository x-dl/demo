import csv
import matplotlib.pyplot as plt

# 读取CSV文件
with open('actiona.csv', 'r') as f:
    reader = csv.reader(f)
    header = next(reader)  # 跳过表头
    data = []
    for row in reader:
        data.append(row)

# 提取第一列数据
x_list = [float(row[0]) for row in data]
y_list = [float(row[1]) for row in data]
# 画折线图
fig, ax = plt.subplots()
ax.plot(x_list,label='qpos')
ax.plot(y_list,color="red",label='action')
# 添加标题和标签
# ax.set_title('Example Data')
ax.set_xlabel('step')
ax.set_ylabel('Value')

# 显示图形
plt.show()




