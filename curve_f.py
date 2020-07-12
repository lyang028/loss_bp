
from scipy.optimize import curve_fit
import dataReader as dr
import numpy as np
import matplotlib.pyplot as plt
import os


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

def draw_summary():
    path_list = [
        'record/model_cnn_zero/',
        'record/multi_label2/',
        'record/multi_label3/',
        'record/multi_label4/',
        'record/multi_label5/',
        'record/multi_label6/',
        'record/multi_label7/',
        'record/multi_label8/',
        'record/model_cnn/',
    ]
    file_list = ['C_dis_random.csv', 'E_dis_random.csv', 'W_dis_random.csv']
    input_c_dis = []
    input_E_dis = []
    input_W_dis = []
    for path in path_list:
        input_c_dis.append(np.array(dr.read_csv(path + file_list[0]), dtype='float')[:, 0])
    for path in path_list:
        input_E_dis.append(np.array(dr.read_csv(path + file_list[1]), dtype='float')[:, 0])
    for path in path_list:
        input_W_dis.append(np.array(dr.read_csv(path + file_list[2]), dtype='float')[:, 0])
    index = 0
    for data in input_c_dis:
        x = normalize(data)
        y = normalize_y(data)
        plt.plot(x, y, label=str(index))
        index = index + 1
    # popt, pcov = curve_fit(func,x,y_ori)
    # a = popt[0]
    # b = popt[1]
    # y = func(x,a,b)

    # plt.plot(x,y)
    plt.legend()
    plt.show()
    index = 0
    for data in input_W_dis:
        x = normalize(data)
        y = normalize_y(data)
        plt.plot(x, y, label=str(index))
        index = index + 1

    plt.legend()
    plt.show()
    plt.close()
    index = 0
    for data in input_E_dis:
        x = normalize(data)
        y = normalize_y(data)
        plt.plot(x, y, label=str(index))
        index = index + 1

    plt.legend()
    plt.show()
    plt.close()


def draw(path_array, labels = [],output_path = 'no',xaxis = 'blank', yaxis = 'blank'):
    plt.rcParams['figure.figsize'] = (8.0, 4.0)
    if len(labels) == 0:
        for path in path_array:
            array = np.array(dr.read_csv(path), dtype=float)[:, 0]
            plt.plot(range(len(array)), array, label=path)
        plt.legend()
        plt.show()
    else:
        index = 0
        for path in path_array:
            array = np.array(dr.read_csv(path), dtype=float)[:, 0]
            plt.plot(range(len(array)), array, label=labels[index])
            index+=1
        plt.xlabel(xaxis)
        plt.ylabel(yaxis)
        plt.legend()
        if output_path == 'no':
            plt.show()
        else:
            plt.savefig(output_path)
def draw_standard(xcoord,ycoord,output_path = 'no',xaxis = 'blank', yaxis = 'blank',ratio = (8,4)):
    plt.rcParams['figure.figsize'] = ratio
    x = np.array(dr.read_csv(xcoord), dtype=float)[:, 0]
    y = np.array(dr.read_csv(ycoord), dtype=float)[:, 0]
    plt.plot(x, y)
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    if output_path == 'no':
        plt.show()
    else:
        plt.savefig(output_path)


def extract_entropydecrease(patharray,axis = 'no',xaxis = 'blank', yaxis = 'blank',ratio = (4,4),sort = 'no'):
    plt.rcParams['figure.figsize'] = ratio
    ed = []
    for path in patharray:
        array = np.array(dr.read_csv(path), dtype=float)
        ed.append(array[0])
    if axis == 'no':
        plt.plot(range(len(ed)), ed)
    else:
        x = np.array(dr.read_csv(axis), dtype=float)[:,0]
        print(x)
        x.sort()
        print(x)
        if sort !='no':
            x.sort()
            ed.sort()
        plt.plot(x, ed)
        plt.xlabel(xaxis)
        plt.ylabel(yaxis)
    ax = plt.gca()  # 获取当前图像的坐标轴信息
    ax.xaxis.get_major_formatter().set_powerlimits((0, 1))  # 将坐标轴的base number设置为一位。
    ax.yaxis.get_major_formatter().set_powerlimits((0, 1))  # 将坐标轴的base number设置为一位。
    plt.show()

def draw_entropy_bar(patharray,data_address = 'none',xaxis = 'blank', yaxis = 'blank',ratio = (4,4),ylimitaion = (400,500)):
    plt.rcParams['figure.figsize'] = ratio

    if data_address == 'none':
        ed = []
        for path in patharray:
            array = np.array(dr.read_csv(path), dtype=float)
            ed.append(array[0,0])
        print(ed)
        plt.bar(range(len(ed)), ed)
    else:
        x = np.array(dr.read_csv(data_address), dtype=float)[:, 0]
        plt.bar(range(len(x)),x)
    plt.ylim(ylimitaion)
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    ax = plt.gca()  # 获取当前图像的坐标轴信息
    ax.xaxis.get_major_formatter().set_powerlimits((0, 1))  # 将坐标轴的base number设置为一位。
    ax.yaxis.get_major_formatter().set_powerlimits((0, 1))  # 将坐标轴的base number设置为一位。
    plt.show()

#figure 1
# path_array = ['record/cnn_mlabel/1/KL_div.csv',
#               'record/cnn_mlabel/2/KL_div.csv',
#               'record/cnn_mlabel/3/KL_div.csv',
#               'record/cnn_mlabel/4/KL_div.csv',
#               'record/cnn_mlabel/5/KL_div.csv',
#               'record/cnn_mlabel/6/KL_div.csv',
#               'record/cnn_mlabel/7/KL_div.csv',
#               'record/cnn_mlabel/8/KL_div.csv',
#               'record/cnn_mlabel/9/KL_div.csv',
#               'record/cnn_standard/KL_div.csv']
# draw(path_array,['$A_0$','$A_1$','$A_2$','$A_3$','$A_4$','$A_5$','$A_6$','$A_7$','$A_8$','$A_9$'],xaxis='Training steps',yaxis='$\Delta S$')
#figure 2
# extract_entropydecrease(path_array,'data_evaluation_minst/len_mnist.csv',xaxis='Training steps',yaxis='$\Delta S$')
# extract_entropydecrease(path_array,'data_evaluation_minst/multilabel_RI.csv',xaxis='Relative Information',yaxis='$\Delta S$')
#figure 3
# path_array = ['record/cnn_mix_all/0/KL_div.csv',
#               'record/cnn_mix_all/1/KL_div.csv',
#               'record/cnn_mix_all/2/KL_div.csv',
#               'record/cnn_mix_all/3/KL_div.csv',
#               'record/cnn_mix_all/4/KL_div.csv',
#               'record/cnn_mix_all/5/KL_div.csv',
#               'record/cnn_mix_all/6/KL_div.csv',
#               'record/cnn_mix_all/7/KL_div.csv',
#               'record/cnn_mix_all/8/KL_div.csv',
#               'record/cnn_mix_all/9/KL_div.csv']
# extract_entropydecrease(path_array,'data_evaluation_minst/mixed_label_RI.csv',xaxis='Relative Information',yaxis='$\Delta S$',sort='Yes')
# draw_entropy_bar(path_array,xaxis='Experiment Index',yaxis='$\Delta S$',ylimitaion= [0.0000001,0.000009],ratio=[4.5,4])
# draw_entropy_bar([],data_address='data_evaluation_minst/mixed_label_RI.csv',xaxis='Experiment Index',yaxis='$Relative Information$',ylimitaion=(400,460),ratio=[4.5,4])
# draw_entropy_bar([],data_address='data_evaluation_minst/averagenumber_label_RI.csv',xaxis='label Index',yaxis='$Relative Information$',ratio=[4.5,4],ylimitaion=(0.004,0.0085))

#figure3
draw_standard('data_evaluation_minst/real_error_detection.csv','data_evaluation_minst/error_label_detection.csv',
              xaxis='Real error rate',yaxis='Information cleanliness')

# path_array = ['record/error_label10/KL_div.csv',

#               'record/error_label20/KL_div.csv',
#               'record/error_label30/KL_div.csv',
#               'record/error_label40/KL_div.csv',
#               'record/error_label50/KL_div.csv',
#               'record/error_label60/KL_div.csv',
#               'record/error_label70/KL_div.csv',
#               'record/error_label80/KL_div.csv',
#               'record/error_label90/KL_div.csv']
# path_array = ['record/error_label70/KL_div.csv',
#               'record/error_label80/KL_div.csv',
#               'record/error_label90/KL_div.csv']
# path_array = ['record/cnn_standard/KL_div.csv'
#     ,'record/cnn_standard1/KL_div.csv'
#     , 'record/cnn_random_order/KL_div.csv'
# , 'record/cnn_error_label/KL_div.csv'
#     ,'record/cnn_random_init/KL_div.csv']

# path_array = ['data_evaluation_minst/multilabel_RI.csv']
# draw(path_array)

# path_array = ['record/cnn_mix_all/0/KL_div.csv','record/cnn_mix_all/1/KL_div.csv','record/cnn_mix_all/2/KL_div.csv','record/cnn_mix_all/3/KL_div.csv',
#               'record/cnn_mix_all/4/KL_div.csv', 'record/cnn_mix_all/5/KL_div.csv','record/cnn_mix_all/6/KL_div.csv','record/cnn_mix_all/7/KL_div.csv',
#               'record/cnn_mix_all/8/KL_div.csv','record/cnn_mix_all/9/KL_div.csv']

# path_array = ['record/cnn_mlabel/1/KL_div.csv','record/cnn_mlabel/2/KL_div.csv','record/cnn_mlabel/3/KL_div.csv',
#               'record/cnn_mlabel/4/KL_div.csv','record/cnn_mlabel/5/KL_div.csv','record/cnn_mlabel/6/KL_div.csv','record/cnn_mlabel/7/KL_div.csv',
#               'record/cnn_mlabel/8/KL_div.csv','record/cnn_mlabel/9/KL_div.csv','record/cnn_standard/KL_div.csv',]

# path_array = ['F:/entropy_analysis/CNN_mlabels/cnn_mlabel1','F:/entropy_analysis/CNN_mlabels/cnn_mlabel2','F:/entropy_analysis/CNN_mlabels/cnn_mlabel3',
#             'F:/entropy_analysis/CNN_mlabels/cnn_mlabel4','F:/entropy_analysis/CNN_mlabels/cnn_mlabel5','F:/entropy_analysis/CNN_mlabels/cnn_mlabel6',
#             'F:/entropy_analysis/CNN_mlabels/cnn_mlabel7','F:/entropy_analysis/CNN_mlabels/cnn_mlabel8','E:/loss_bp/cnn_standard']
