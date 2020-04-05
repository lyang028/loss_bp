
from scipy.optimize import curve_fit
import dataReader as dr
import numpy as np
import matplotlib.pyplot as plt

input1 = np.array(dr.read_csv('record/multi_label6/C_dis_random.csv'),dtype='float')[:,0]
input2 = np.array(dr.read_csv('record/multi_label5/C_dis_random.csv'),dtype='float')[:,0]
input3 = np.array(dr.read_csv('record/multi_label4/C_dis_random.csv'),dtype='float')[:,0]
input4 = np.array(dr.read_csv('record/multi_label3/C_dis_random.csv'),dtype='float')[:,0]
input5 = np.array(dr.read_csv('record/multi_label2/C_dis_random.csv'),dtype='float')[:,0]
input6 = np.array(dr.read_csv('record/model_cnn_zero/C_dis_random.csv'),dtype='float')[:,0]


print(input)

def func(x, a, b):
    return a*np.exp(b-x)
def normalize(input):
    x = np.array(range(len(input)))
    x = x / len(input)
    return  x
def normalize_y(input):
    # y_ori = (1 - input) / (1 - input[0])
    # return  y_ori

    return  1 - input

x1 = normalize(input1)
x2 = normalize(input2)
x3 = normalize(input3)
x4 = normalize(input4)
x5 = normalize(input5)
x6 = normalize(input6)

y1 = normalize_y(input1)
y2 = normalize_y(input2)
y3 = normalize_y(input3)
y4 = normalize_y(input4)
y5 = normalize_y(input5)
y6 = normalize_y(input6)

# popt, pcov = curve_fit(func,x,y_ori)
# a = popt[0]
# b = popt[1]
# y = func(x,a,b)
plt.plot(x1,y1,label = '1')
plt.plot(x2,y2,label = '2')
plt.plot(x3,y3,label = '3')
plt.plot(x4,y4,label = '4')
plt.plot(x5,y5,label = '5')
plt.plot(x6,y6,label = '6')

# plt.plot(x,y)
plt.legend()
plt.show()