from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2
def test():
    fig = plt.figure()  # 定义新的三维坐标轴
    ax3 = plt.axes(projection='3d')

    # 定义三维数据
    xx = np.arange(0, 5, 0.5)
    yy = np.arange(0, 5, 0.5)
    zz = np.arange(0, 100, 1)
    X, Y = np.meshgrid(xx, yy)
    key = []
    x = 0
    for i in range(len(xx)):
        for j in range(len(xx)):
            key.append('x' + str(xx[i]) + 'y' + str(yy[j]))
            x += 1

    print(x)
    dic = dict(zip(key, zz))
    print(dic)
    print(len(dic))
    print(len(xx))

    Z = np.zeros(X.shape)
    print(Z)
    for i in range(len(xx)):
        for j in range(len(xx)):
            Z[i, j] = dic['x' + str(X[i, j]) + 'y' + str(Y[i, j])]

    # #作图
    ax3.plot_surface(X, Y, Z, cmap='rainbow')
    # ax3.contour(X,Y,Z, zdim='z',offset=-2，cmap='rainbow)   #等高线图，要设置offset，为Z的最小值
    plt.show()

import cnn_model as cm

def row_suture():
    # 数独是个 9x9 的二维数组
    # 包含 9 个 3x3 的九宫格
    # sudoku = np.array([
    #     [2, 8, 7, 1, 6, 5, 9, 4, 3],
    #     [9, 5, 4, 7, 3, 2, 1, 6, 8],
    #     [6, 1, 3, 8, 4, 9, 7, 5, 2],
    #     [8, 7, 9, 6, 5, 1, 2, 3, 4],
    #     [4, 2, 1, 3, 9, 8, 6, 7, 5],
    #     [3, 6, 5, 4, 2, 7, 8, 9, 1],
    #     [1, 9, 8, 5, 7, 3, 4, 2, 6],
    #     [5, 4, 2, 9, 1, 6, 3, 8, 7],
    #     [7, 3, 6, 2, 8, 4, 5, 1, 9]
    # ])
    sudoku = np.zeros([9,12])
    a = ([1,1,1],[2,2,2],[3,3,3],[4,4,4])
    # 要将其变成 3x3x3x3 的四维数组
    # 但不能直接 reshape，因为这样会把一行变成一个九宫格
    shape = (4, 3, 3, 3)
    sequares = sudoku.reshape(shape)
    print(sequares)
    xx = sequares[:,:,0,0]
    sequares[:, :, 0, 0] = a
    print(sequares)
    strides = sudoku.itemsize * np.array([36, 3, 12, 1])
    squares = np.lib.stride_tricks.as_strided(sudoku, shape=shape, strides=strides)
    xx = squares[:,:,0,0]
    # 大行之间隔 27 个元素，大列之间隔 3 个元素
    # 小行之间隔 9 个元素，小列之间隔 1 个元素

    print(sequares.reshape([9,12]))

def col_images(set):
    output = set[0]
    for i in range(2):
        output = np.vstack((output, set[i + 1]))
    return output
def extract_images():
    x_0 = cm.x_train[ cm.y_index_train == 0]
    x_1 = cm.x_train[ cm.y_index_train == 1]
    x_2 = cm.x_train[cm.y_index_train == 2]
    x_3 = cm.x_train[cm.y_index_train == 3]
    x_4 = cm.x_train[cm.y_index_train == 4]
    x_5 = cm.x_train[cm.y_index_train == 5]
    x_6 = cm.x_train[cm.y_index_train == 6]
    x_7 = cm.x_train[cm.y_index_train == 7]
    x_8 = cm.x_train[cm.y_index_train == 8]
    x_9 = cm.x_train[cm.y_index_train == 9]

    output = col_images(x_0)
    output = np.concatenate([output,col_images(x_1)],axis=1)
    output = np.concatenate([output, col_images(x_2)], axis=1)
    output = np.concatenate([output, col_images(x_3)], axis=1)
    output = np.concatenate([output, col_images(x_4)], axis=1)
    output = np.concatenate([output, col_images(x_5)], axis=1)
    output = np.concatenate([output, col_images(x_6)], axis=1)
    output = np.concatenate([output, col_images(x_7)], axis=1)
    output = np.concatenate([output, col_images(x_8)], axis=1)
    output = np.concatenate([output, col_images(x_9)], axis=1)

    # cv2.imshow('test',output)
    cv2.imwrite('MNIST.png',output*255)
    # cv2.waitKey()

# row_suture()
extract_images()