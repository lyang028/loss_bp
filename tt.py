import matplotlib.pyplot as plt
import matplotlib
import numpy as np


# 计算分位数
def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


# 坐标轴样式
def set_axis_style(ax, labels):
    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('Sample name')


# 创建演示数据集
np.random.seed(2020)
data = [sorted(np.random.normal(0, std, 100)) for std in range(1, 5)]

# 定义画布
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))

# 设置标题
ax.set_title('Customized violin plot')

# 绘制小提琴图
parts = ax.violinplot(
    data, showmeans=False, showmedians=False,
    showextrema=False)
cmap = plt.get_cmap('rainbow')

print(cmap(5))
# # 设置填充颜色和边框颜色
for pc in parts['bodies']:
    pc.set_facecolor(cmap(1))
    pc.set_edgecolor('black')
    pc.set_alpha(0.5)
#
# # 计算中位数，上四分位和下四分位，绘制小提琴内部盒状图
# quartile1, medians, quartile3 = np.percentile(data, [25, 50, 75], axis=1)
# whiskers = np.array([
#     adjacent_values(sorted_array, q1, q3)
#     for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
# whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]
#
# # 添加点和竖线
# inds = np.arange(1, len(medians) + 1)
# ax.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
# ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
# ax.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)
#
# # 设置坐标轴样式
labels = ['A', 'B', 'C', 'D']
set_axis_style(ax, labels)
#
# plt.subplots_adjust(bottom=0.15, wspace=0.05)
plt.show()
