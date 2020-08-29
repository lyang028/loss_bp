import numpy as np


# def row_suture():
#
#     a = np.array(([1,1,1],[2,2,2],[3,3,3],[4,4,4]))
#     b = np.array(([1,1,1],[2,2,2],[3,3,3],[4,4,4]))
#     o = np.concatenate([a,b],axis=1)
#     print(o)
# # row_suture()
#
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
#
# fig = plt.figure()
# ax = Axes3D(fig)  # 生成一个3d对象
# X = np.arange(-4, 4, 0.25)
# Y = np.arange(-4, 4, 0.25)
# X, Y = np.meshgrid(X, Y)  # 对X,Y数组进行扩充
# R = np.sqrt(X ** 2 + Y ** 2)
# Z = np.sin(R)
# ax.set_xlabel('X label', color='r')  # 设置x坐标
# ax.set_ylabel('Y label', color='r')
# ax.set_zlabel('Z label')
# ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.hot)  # 生成一个曲面
# #ax.contourf(X,Y,Z,offset=2,alpha=0.75,cmap=plt.cm.hot)#为等高线填充颜色
# ax.contour(X, Y, Z, offset=0, colors='black')  # 生成等高线 offset参数是等高线所处的位置
# # ax.set_zlim(-2, 2)  # 设置z的范围
# # plt.show()
# plt.savefig('test_contuor.png')
# plt.close()
#
#
# ratio = Z.max() - Z.min()
# rag = np.arange(0,10,1)*ratio/10+Z.min()
# print(rag)
# C = plt.contour(X, Y, Z,rag,cmap = 'rainbow')  # 如果想要在等高线上标出相应的值，需要重新生成一个对象，不能是3d对象
# plt.clabel(C, inline=True, fontsize=10)  # 在等高线上标出对应的z值
# # ax.set_zlim(-2, 2)  # 设置z的范围
# # plt.show()
#
# plt.savefig('test_c.png')
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
import numpy as np
from dataReader import read_csv

# Fixing random state for reproducibility
np.random.seed(19680801)


def randrange(n, vmin, vmax):
    '''
    Helper function to make an array of random numbers having shape (n, )
    with each number distributed Uniform(vmin, vmax).
    '''
    return (vmax - vmin)*np.random.rand(n) + vmin


array = np.array(read_csv('MDS_test/output.csv'), dtype=float)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

n = 100

# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
colors = list(range(len(array)))
ax.scatter(array[:,0], array[:,1], array[:,2], c = colors)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()