import cv2
from keras.datasets import mnist, fashion_mnist
import numpy as np
# (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train = x_train.reshape(len(x_train), -1).astype('float32') / 255.
# x_test = x_test.reshape(len(x_test), -1).astype('float32') / 255.
def extract_onetwo():
    x_train_1 = x_train[y_train == 1]
    x_train_2 = x_train[y_train == 2]

    x_output = np.array(x_train_1[0])
    for i in [1, 2, 3, 4, 6, 7, 8, 11, 10]:
        x_output = np.c_[x_output, x_train_1[i]]

    x_output_2 = np.array(x_train_2[0])
    for i in [3, 8, 12, 13, 14, 16, 17, 18, 19]:
        x_output_2 = np.c_[x_output_2, x_train_2[i + 1]]

    x_output = np.r_[x_output, x_output_2]
    cv2.imshow('test', x_output)
    cv2.imwrite('output.png', x_output)
    cv2.waitKey(0)
def extract_all():
    xset = []
    for i in range(10):
        xset.append(x_train[y_train == i][0])


    x_output = np.array(xset[0])

    for i in range(9):
        x_output = np.c_[x_output,xset[i + 1]]

    cv2.imshow('test', x_output)
    cv2.imwrite('output.png', x_output)
    cv2.waitKey(0)


# extract_onetwo()
extract_all()