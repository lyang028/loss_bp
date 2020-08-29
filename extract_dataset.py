import cv2
from keras.datasets import mnist, fashion_mnist

# (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train = x_train.reshape(len(x_train), -1).astype('float32') / 255.
# x_test = x_test.reshape(len(x_test), -1).astype('float32') / 255.

cv2.imshow('test',x_train[5])
cv2.waitKey(0)