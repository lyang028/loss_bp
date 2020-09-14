
from scipy.optimize import curve_fit
import dataReader as dr
import numpy as np
import matplotlib.pyplot as plt
import os
from itertools import combinations, permutations
from mpl_toolkits.mplot3d import Axes3D


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


def draw(path_array, labels = [],output_path = 'no',xaxis = 'blank', yaxis = 'blank',range_select = [-1,-1],ratio = (8.0, 4.0),dpi = 100,xoffset = [1,0],tight_layout = True,font_size = [20,10,10]):
    plt.rcParams['figure.figsize'] = ratio
    plt.rcParams['figure.dpi'] = dpi  # 分辨率
    plt.xlabel(xaxis, fontsize=font_size[0])
    plt.ylabel(yaxis, fontsize=font_size[0])
    plt.xticks(fontsize=font_size[1])
    plt.yticks(fontsize=font_size[1])
    if len(labels) == 0:
        # plt.axes(xscale='log')
        for path in path_array:
            array = np.array(dr.read_csv(path), dtype=float)[:, 0]
            if range_select[0] == -1:
                x = np.array(range(len(array)))*xoffset[0]+xoffset[1]
                plt.plot(x, array)
            else:
                array = array[range_select[0]:range_select[1]]
                x = np.array(range(len(array))) * xoffset[0] + xoffset[1]
                plt.plot(x, array)
        if tight_layout:
            plt.tight_layout()
        plt.show()
    else:
        index = 0
        for path in path_array:
            array = np.array(dr.read_csv(path), dtype=float)[:, 0]
            if range_select[0] == -1:
                plt.plot(range(len(array)), array, label=labels[index])
            else:
                array = array[range_select[0]:range_select[1]]
                plt.plot(range(len(array)), array, label=labels[index])
            index+=1
        plt.legend(fontsize = font_size[2])
        if tight_layout:
            plt.tight_layout()
        if output_path == 'no':
            plt.show()
        else:
            plt.savefig(output_path)
def draw_standard(xcoord,ycoord,output_path = 'no',xaxis = 'blank', yaxis = 'blank',ratio = (8,4),tight_layout = False,font_size = [10,10]):
    plt.rcParams['figure.figsize'] = ratio
    x = np.array(dr.read_csv(xcoord), dtype=float)[:, 0]
    y = np.array(dr.read_csv(ycoord), dtype=float)[:, 0]

    plt.xlabel(xaxis, fontsize=font_size[0])
    plt.ylabel(yaxis, fontsize=font_size[0])
    plt.xticks(fontsize=font_size[1])
    plt.yticks(fontsize=font_size[1])
    plt.plot(x, y)
    if tight_layout:
        plt.tight_layout()
    if output_path == 'no':
        plt.show()
    else:
        plt.savefig(output_path)


def extract_entropydecrease(patharray,axis = 'no',xaxis = 'blank', yaxis = 'blank',ratio = (4,4),sort = 'no',tight_layout = True,font_size = [20,10]):
    plt.rcParams['figure.figsize'] = ratio
    plt.xlabel(xaxis, fontsize=font_size[0])
    plt.ylabel(yaxis, fontsize=font_size[0])
    plt.xticks(fontsize=font_size[1])
    plt.yticks(fontsize=font_size[1])
    if tight_layout:
        plt.tight_layout()
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
    ax = plt.gca()  # 获取当前图像的坐标轴信息
    ax.xaxis.get_major_formatter().set_powerlimits((0, 1))  # 将坐标轴的base number设置为一位。
    ax.yaxis.get_major_formatter().set_powerlimits((0, 1))  # 将坐标轴的base number设置为一位。
    plt.show()

def draw_entropy_bar(patharray,data_address = 'none',xaxis = 'blank', yaxis = 'blank',ratio = (4,4),ylimitaion = (400,500),tight_layout = True, font_size = [20,10]):
    plt.rcParams['figure.figsize'] = ratio
    plt.xlabel(xaxis, fontsize=font_size[0])
    plt.ylabel(yaxis, fontsize=font_size[0])
    plt.xticks(fontsize=font_size[1])
    plt.yticks(fontsize=font_size[1])
    if tight_layout:
        plt.tight_layout()
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



def optimzie_key(f,accuracy):
    f = round(f,accuracy+1)
    str_f = str(f)
    if f ==0 :
        str_f = '0.0'
    return  str_f


def draw_contour(RI_path,weight_path,output_path,sample_rate = 0.01,font_size = [20,10,10]):
    RI = np.array(dr.read_csv(RI_path),dtype=float)
    test_center_w = np.array(dr.read_csv(weight_path),dtype=float)
    zz = np.array(RI, dtype='float')
    dim = range(3)
    dim_set = combinations(dim, 2)
    for dim_select in dim_set:
        key = []
        for w in test_center_w:
            w_d1 = optimzie_key(w[dim_select[0]], 2)
            w_d2 = optimzie_key(w[dim_select[1]], 2)
            mark1 = 'x' + str(w_d1)
            mark2 = 'y' + str(w_d2)
            key.append(mark1 + mark2)
        dic = dict(zip(key, zz))

        xx = np.arange(0, 1 + 0.01, sample_rate)
        yy = np.arange(0, 1 + sample_rate, sample_rate)
        X, Y = np.meshgrid(xx, yy)

        Z = np.zeros(X.shape)
        for i in range(len(xx)):
            for j in range(len(xx)):
                w_d1 = optimzie_key(X[i, j], 2)
                w_d2 = optimzie_key(Y[i, j], 2)
                mark1 = 'x' + str(w_d1)
                mark2 = 'y' + str(w_d2)
                if mark1 + mark2 in dic:
                    Z[i, j] = dic[mark1 + mark2]
                else:
                    Z[i, j] = 0
        # plt.rcParams['figure.figsize'] = [6,4]
        ax3 = plt.axes(projection='3d')
        handle = ax3.plot_surface(X, Y, Z, cmap='nipy_spectral')
        zoom = Z.max() - Z.min()
        # rag = np.arange(0, 10, 0.2) * ratio / 10 + Z.min() + 0.0001
        # ax3.contour(X, Y, Z,rag, offset=0, cmap='hot')
        # ax3.set_zlim(0, 1)  # 设置z的范围
        ax3.set_zlabel('RI', fontsize=font_size[0])
        # ax3.set_zticks([])
        plt.xlabel('$s_ ' + str(dim_select[0]) + '$ weight', fontsize=font_size[0])
        plt.ylabel('$s_ ' + str(dim_select[1]) + '$ weight', fontsize=font_size[0])
        # fig = plt.figure()
        plt.colorbar(handle)
        plt.tight_layout()
        plt.show()
        plt.close()

        plt.rcParams['figure.figsize'] = [4.2, 4]
        plt.xlabel('$s_ ' + str(dim_select[0]) + '$ weight', fontsize=font_size[0])
        plt.ylabel('$s_ ' + str(dim_select[1]) + '$ weight', fontsize=font_size[0])
        zoom = Z.max() - Z.min()
        rag = np.arange(0, 10, 0.2) * zoom / 10 + Z.min()+0.0001

        C = plt.contour(X, Y, Z, rag, cmap='rainbow')
        plt.tight_layout()
        plt.show()
        # plt.clabel(C, inline=True, fontsize=10)  # 在等高线上标出对应的z值
        # ax3.set_zlim(-1, 1)  # 设置z的范围
        # plt.savefig(output_path + str(dim_select[0]) + '_' + str(dim_select[1]) + 'contour.png')
        plt.close()

def draw_scatter(path_set,data_address = 'none',xaxis = 'blank', yaxis = 'blank',ratio = (4,4),ylimitaion = (400,500),axis = [],range_select = [],open_color = True,font_size = 20):
    plt.rcParams['figure.figsize'] = ratio
    for path in path_set:
        array = np.array(dr.read_csv(path), dtype=float)
        if len(range_select) != 0:
            array = array[range_select[0]:range_select[1]]
        x = array[:, 0]
        y = array[:, 1]
        if open_color:
            colors = list(range(len(array)))
            plt.scatter(x,y, c=colors, alpha=0.5)
        else:
            plt.scatter(x, y, alpha=0.5)

    # plt.show()

    if len(axis) != 0:
        plt.axis(axis)
    if xaxis != 'blank':
        plt.xlabel(xaxis)
    if yaxis!='blank':
        plt.ylabel(yaxis)

    plt.show()

def draw_scatter3D(path_set,data_address = 'none',xaxis = 'blank', yaxis = 'blank',ratio = (4,4),ylimitaion = (400,500),axis = [],range_select = [],open_color = True):
    plt.rcParams['figure.figsize'] = ratio
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    for path in path_set:
        array = np.array(dr.read_csv(path), dtype=float)
        if len(range_select) != 0:
            array = array[range_select[0]:range_select[1]]


        if open_color:
            colors = list(range(len(array)))
            ax.scatter(array[:, 0], array[:, 1], array[:, 2], c=colors)
        else:
            ax.scatter(array[:, 0], array[:, 1], array[:, 2])
        # plt.show()
    if len(axis) != 0:
        plt.axis(axis)
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)

    # area = (30 * np.random.rand(N)) ** 2  # 0 to 15 point radii



    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

    # plt.ylim(ylimitaion)
    # plt.xlabel(xaxis)
    # plt.ylabel(yaxis)
    # ax = plt.gca()  # 获取当前图像的坐标轴信息
    # ax.xaxis.get_major_formatter().set_powerlimits((0, 1))  # 将坐标轴的base number设置为一位。
    # ax.yaxis.get_major_formatter().set_powerlimits((0, 1))  # 将坐标轴的base number设置为一位。
#figure 1
path_array = ['record/cnn_mlabel/1/KL_div.csv',
              'record/cnn_mlabel/2/KL_div.csv',
              'record/cnn_mlabel/3/KL_div.csv',
              'record/cnn_mlabel/4/KL_div.csv',
              'record/cnn_mlabel/5/KL_div.csv',
              'record/cnn_mlabel/6/KL_div.csv',
              'record/cnn_mlabel/7/KL_div.csv',
              'record/cnn_mlabel/8/KL_div.csv',
              'record/cnn_mlabel/9/KL_div.csv',
              'record/cnn_standard/KL_div.csv']
# draw(path_array,['$A_0$','$A_1$','$A_2$','$A_3$','$A_4$','$A_5$','$A_6$','$A_7$','$A_8$','$A_9$'],xaxis='Training Steps',yaxis='$\Delta S$',ratio=[12,4],tight_layout=True,font_size=[20,10,10])
#figure 2
# extract_entropydecrease(path_array,'data_evaluation_minst/len_mnist.csv',xaxis='Len(D)',yaxis='$\Delta S$')
# extract_entropydecrease(path_array,'data_evaluation_minst/multilabel_RI.csv',xaxis='RI',yaxis='$\Delta S$')
#figure 3
path_array = ['record/cnn_mix_all/0/KL_div.csv',
              'record/cnn_mix_all/1/KL_div.csv',
              'record/cnn_mix_all/2/KL_div.csv',
              'record/cnn_mix_all/3/KL_div.csv',
              'record/cnn_mix_all/4/KL_div.csv',
              'record/cnn_mix_all/5/KL_div.csv',
              'record/cnn_mix_all/6/KL_div.csv',
              'record/cnn_mix_all/7/KL_div.csv',
              'record/cnn_mix_all/8/KL_div.csv',
              'record/cnn_mix_all/9/KL_div.csv']
# extract_entropydecrease(path_array,'data_evaluation_minst/mixed_label_RI.csv',xaxis='RI',yaxis='$\Delta S$',sort='Yes')
# draw_entropy_bar(path_array,xaxis='Index',yaxis='$\Delta S$',ylimitaion= [0.0000001,0.000009],ratio=[4.5,4])
# draw_entropy_bar([],data_address='data_evaluation_minst/mixed_label_RI.csv',xaxis='Index',yaxis='RI',ylimitaion=(400,460),ratio=[4.5,4])
# draw_entropy_bar([],data_address='data_evaluation_minst/averagenumber_label_RI.csv',xaxis='Label',yaxis='RI',ratio=[4.5,4],ylimitaion=(0.004,0.0085))

#figure3
# draw_standard('data_evaluation_minst/real_error_detection.csv','data_evaluation_minst/error_label_detection.csv',
#               xaxis='Mislabel Rate',yaxis='IC',ratio=[12,4],tight_layout=True,font_size=[20,10])

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

# path_array = ['Final_experiment/performance/dis.csv']
# path_array = ['Final_experiment/performance/dis_limitation_e.csv']
# draw(path_array)
# draw(path_array,range_select=[350,3000])
#
# draw(path_array,range_select=[0,300])


#figure************************************** error rate

# path_array = ['record/error_label90/KL_div.csv','record/error_label80/KL_div.csv','record/error_label70/KL_div.csv',
#               'record/error_label60/KL_div.csv']
# draw(path_array)
#
# path_array = ['record/error_label50/KL_div.csv',
#               'record/error_label40/KL_div.csv','record/error_label30/KL_div.csv','record/error_label20/KL_div.csv',
#               'record/error_label10/KL_div.csv']
# draw(path_array)
#
# path_array = ['record/error_label10/KL_div.csv','record/error_label20/KL_div.csv','record/error_label30/KL_div.csv',
#               'record/error_label40/KL_div.csv','record/error_label50/KL_div.csv','record/error_label60/KL_div.csv',
#               'record/error_label70/KL_div.csv','record/error_label80/KL_div.csv','record/error_label90/KL_div.csv']
# labels = ['$A_{10}$','$A_{20}$','$A_{30}$','$A_{40}$','$A_{50}$','$A_{60}$','$A_{70}$','$A_{80}$','$A_{90}$']
# draw(path_array,labels=labels,xaxis='Training steps',yaxis='$\Delta S$',tight_layout=True,font_size=[20,10,10])

# path_array = ['record/error_label10/KL_div_sigmoid.csv','record/error_label20/KL_div_sigmoid.csv','record/error_label30/KL_div_sigmoid.csv',
#               'record/error_label40/KL_div_sigmoid.csv','record/error_label50/KL_div_sigmoid.csv','record/error_label60/KL_div_sigmoid.csv',
#               'record/error_label70/KL_div_sigmoid.csv','record/error_label80/KL_div_sigmoid.csv','record/error_label90/KL_div_sigmoid.csv']
# labels = ['$A_{10}$','$A_{20}$','$A_{30}$','$A_{40}$','$A_{50}$','$A_{60}$','$A_{70}$','$A_{80}$','$A_{90}$']
# draw(path_array,labels=labels,xaxis='Training steps',yaxis='$\Delta S$')

# path_array = ['record/error_label10/E_dis.csv','record/error_label20/E_dis.csv','record/error_label30/E_dis.csv',
#               'record/error_label40/E_dis.csv','record/error_label50/E_dis.csv','record/error_label60/E_dis.csv',
#               'record/error_label70/E_dis.csv','record/error_label80/E_dis.csv','record/error_label90/E_dis.csv']
# labels = ['$A_{10}$','$A_{20}$','$A_{30}$','$A_{40}$','$A_{50}$','$A_{60}$','$A_{70}$','$A_{80}$','$A_{90}$']
# draw(path_array,labels=labels,xaxis='Training steps',yaxis='$\Delta S Eucil$',ratio=[4,4])
# path_array = ['record/error_label10/acc.csv','record/error_label20/acc.csv','record/error_label30/acc.csv',
#               'record/error_label40/acc.csv','record/error_label50/acc.csv','record/error_label60/acc.csv',
#               'record/error_label70/acc.csv','record/error_label80/acc.csv','record/error_label90/acc.csv']
# labels = ['$A_{10}$','$A_{20}$','$A_{30}$','$A_{40}$','$A_{50}$','$A_{60}$','$A_{70}$','$A_{80}$','$A_{90}$']
# draw(path_array,labels=labels,xaxis='Training steps',yaxis='$Accuracy$',ratio=[8,4],dpi = 100)
#
#
# path_array = ['record/error_label10/loss.csv','record/error_label20/loss.csv','record/error_label30/loss.csv',
#               'record/error_label40/loss.csv','record/error_label50/loss.csv','record/error_label60/loss.csv',
#               'record/error_label70/loss.csv','record/error_label80/loss.csv','record/error_label90/loss.csv']
# labels = ['$A_{10}$','$A_{20}$','$A_{30}$','$A_{40}$','$A_{50}$','$A_{60}$','$A_{70}$','$A_{80}$','$A_{90}$']
# draw(path_array,labels=labels,xaxis='Training steps',yaxis='$Loss$',ratio=[8,4],dpi = 100)
#
# path_array = ['record/cnn_mlabel/1/KL_div.csv',
#               'record/cnn_mlabel/2/KL_div.csv']
# draw(path_array,['$A_0$','$A_1$'],xaxis='Training steps',yaxis='$\Delta S$')

#figure********************************* numerical verify
draw_contour('data_evaluation_minst/data_center_test/mark_C/0/RI.csv','data_evaluation_minst/data_center_test/mark_C/0/test_center_w.csv',
             'data_evaluation_minst/data_center_test/mark_C/repaint/')
#figure********************************************
# draw_scatter3D('MDS_test/output.csv')#test

# draw_scatter('MDS_test/output2.csv',axis=[-2,1,-2,2.6],range_select=[0,30])#test
# draw_scatter('MDS_test/output2.csv',axis=[-2,1,-2,2.6],range_select=[0,60])#test
# draw_scatter('MDS_test/output2.csv',axis=[-2,1,-2,2.6],range_select=[0,90])#test
# path = ['MDS_test/2DMDS/w.csv','MDS_test/2DMDS/w2.csv','MDS_test/2DMDS/w3.csv','MDS_test/2DMDS/w4.csv','MDS_test/2DMDS/w5.csv','MDS_test/2DMDS/w6.csv']
# path = ['MDS_test/2DMDS/gredient_and_weight/w_special.csv']
# draw_scatter(path,ratio=[4,4],open_color=False,axis = 'off')#test
# path = ['MDS_test/2DMDS/gredient_and_weight/w_special_gredient.csv']
# draw_scatter(path,ratio=[4,4],open_color=False,axis = 'off')

# path = ['MDS_test/w_special.csv']
# draw_scatter3D(path,ratio=[4,4])#test
# path = ['MDS_test/w_special_gredient.csv']
# draw_scatter3D(path,ratio=[4,4])

#
# path = ['MDS_test/2DMDS/w_error_label_gredient.csv']
# draw_scatter(path,ratio=[4,4],open_color=False,axis = 'off')
#
# path = ['MDS_test/2DMDS/w_error_label.csv','MDS_test/2DMDS/gredient_and_weight/w_special.csv']
# draw_scatter(path,ratio=[4,4],open_color=True,axis = 'off')
#test*------------------------------------------
# path = ['MDS_test/2DMDS/w4.csv','MDS_test/2DMDS/w5.csv','MDS_test/2DMDS/w6.csv']
# draw_scatter3D(path)
#test------------------------------------------
# path_array = ['Final_experiment/repeat_training_limitation/record/extreme/dis_limitation_acc.csv',
#               'Final_experiment/repeat_training_limitation/record/extreme_add/dis_limitation_acc.csv',
#               'Final_experiment/repeat_training_limitation/record/extreme_addt/dis_limitation_acc.csv']
# draw(path_array,['one layer','two layer','three layer'],xaxis='Training steps',yaxis='acc',range_select=[0,500])
#
# path_array = ['Final_experiment/repeat_training_limitation/record/extreme/dis_limitation_loss.csv',
#               'Final_experiment/repeat_training_limitation/record/extreme_add/dis_limitation_loss.csv',
#               'Final_experiment/repeat_training_limitation/record/extreme_addt/dis_limitation_loss.csv']
# draw(path_array,['one layer','two layer','three layer'],xaxis='Training steps',yaxis='loss',range_select=[0,500])
# #
# path_array = ['Final_experiment/repeat_training_limitation/record/extreme/dis_limitation_e.csv',
#               'Final_experiment/repeat_training_limitation/record/extreme_add/dis_limitation_e.csv',
#               'Final_experiment/repeat_training_limitation/record/extreme_addt/dis_limitation_e.csv']
# draw(path_array,['one layer','two layer','three layer'],xaxis='Training steps',yaxis='$\Delta S$',range_select=[0,500])
#
# path_array = ['Final_experiment/repeat_training_limitation/record/extreme/dis_limitation_eul.csv',
#               'Final_experiment/repeat_training_limitation/record/extreme_add/dis_limitation_eul.csv',
#               'Final_experiment/repeat_training_limitation/record/extreme_addt/dis_limitation_eul.csv']
# draw(path_array,['one layer','two layer','three layer'],xaxis='Training steps',yaxis='euclidean metric',range_select=[0,500])

#figure**************************************************
# path_array = ['record/cnn_sl_test/mlabel0/acc.csv',
#               'record/cnn_sl_test/mlabel1/acc.csv']
# draw(path_array,['zero','zero & one'],xaxis='Training steps',yaxis='acc')
#
# path_array = ['record/cnn_sl_test/mlabel0/E_dis.csv',
#               'record/cnn_sl_test/mlabel1/E_dis.csv']
# draw(path_array,['zero','zero & one'],xaxis='Training steps',yaxis='Edis')
# #
# path_array = ['record/cnn_sl_test/mlabel0/KL_div.csv',
#               'record/cnn_sl_test/mlabel1/KL_div.csv']
# draw(path_array,['zero','zero & one'],xaxis='Training steps',yaxis='KL_div')
#
# path_array = ['record/cnn_sl_test/mlabel0/KL_div_sigmoid.csv',
#               'record/cnn_sl_test/mlabel1/KL_div_sigmoid.csv']
# draw(path_array,['zero','zero & one'],xaxis='Training steps',yaxis='KL_div_sigmoid')
#
# path_array = ['record/cnn_sl_test/mlabel0/loss.csv',
#               'record/cnn_sl_test/mlabel1/loss.csv']
# draw(path_array,['zero','zero & one'],xaxis='Training steps',yaxis='loss')

#***********************************************************************Optimizer
# path_array = ['Final_experiment/repeat_training_limitation/record/extreme_add_adam/dis_limitation_e.csv',
#               'Final_experiment/repeat_training_limitation/record/extreme_add_sgd/dis_limitation_e.csv']
#
# draw(path_array,['Adam','SGD'],xaxis='Training Steps',yaxis='$\Delta S$',dpi = 100,ratio=[4.5,4])
#
# path_array = ['Final_experiment/repeat_training_limitation/record/extreme_add_adam/dis_limitation_loss.csv',
#               'Final_experiment/repeat_training_limitation/record/extreme_add_sgd/dis_limitation_loss.csv']
#
# draw(path_array,['Adam','SGD'],xaxis='Training Steps',yaxis='Loss',dpi = 100,ratio=[4.5,4])
#
# path_array = ['Final_experiment/repeat_training_limitation/record/extreme_add_adam/dis_limitation_acc.csv',
#               'Final_experiment/repeat_training_limitation/record/extreme_add_sgd/dis_limitation_acc.csv']
#
# draw(path_array,['Adam','SGD'],xaxis='Training Steps',yaxis='Accuracy',dpi = 100,ratio=[4.5,4])

#*********************************************************************learning rate
# path_array = ['learningrate_test_save/ac.csv']
# draw(path_array,xaxis='Learning Rate',yaxis='Accuracy',dpi = 100,ratio=[4.5,4],xoffset=[0.01,0.01])
#
# path_array = ['learningrate_test_save/loss.csv']
# draw(path_array,xaxis='Learning Rate',yaxis='Loss',dpi = 100,ratio=[4.5,4],xoffset=[0.01,0.01])
#
# path_array = ['learningrate_test_save/dis.csv']
# draw(path_array,labels=[],xaxis='Learning Rate',yaxis='$\Delta S$',dpi = 100,ratio=[4.5,4],xoffset=[0.01,0.01])
#*******************************************************************batchsize
# path_array = ['Final_experiment/batch_size/ac.csv']
# draw(path_array,xaxis='Batch Size',yaxis='Accuracy',dpi = 100,ratio=[4.5,4],xoffset=[1,1])

# path_array = ['Final_experiment/batch_size/loss.csv']
# draw(path_array,xaxis='Batch Size',yaxis='Loss',dpi = 100,ratio=[4.5,4],xoffset=[1,1])

# path_array = ['Final_experiment/batch_size/dis.csv']
# draw(path_array,labels=[],xaxis='Batch Size',yaxis='$\Delta S$',dpi = 100,ratio=[4.5,4],xoffset=[1,1])