import cv2
import numpy as np
import dataReader as dr



def creat_povray_mesh_of_numberical_verification(output_file):
    RI_raw = np.array(dr.read_csv('data_evaluation_minst/data_center_test/mark_C/0/RI.csv'),dtype=float)[:,0]
    weights_raw = np.array(dr.read_csv('data_evaluation_minst/data_center_test/mark_C/0/test_center_w.csv'),dtype=float)
    x_set = weights_raw[:,0]
    y_set = weights_raw[:, 1]
    z_set = weights_raw[:, 2]



    idx = x_set>=0
    idx2 = y_set>=0
    idx3 = z_set>=0
    idx = np.bitwise_and(idx,idx2)
    idx = np.bitwise_and(idx,idx3)
    RI = RI_raw[idx]
    x_set = x_set[idx]
    y_set = y_set[idx]
    z_set = z_set[idx]
    RI_norm =  np.array(RI / (np.max(RI) - np.min(RI)), dtype=float)
    RI_color = np.array(RI_norm*255, dtype=np.uint8)
    im_color = cv2.applyColorMap(RI_color, cv2.COLORMAP_RAINBOW)
    fw = open(output_file[0], 'w')  # 将要输出保存的文件地址
    for i in range(len(RI)):
        coord = [x_set[i], RI_norm[i], y_set[i]]
        color = im_color[i]
        insert_one_ellispon(fw, coord, color)
    fw.close()


def insert_one_ellispon(fw,coordinate,color):
    color = np.array(list(color[0]),dtype=float)
    coordinate = np.array(coordinate)
    fw.write('superellipsoid{ <1.00,1.00> texture{ pigment{ color rgbf<'+
             str(color[0])+','+str(color[1])+','+str(color[2])+
             ',0.5> } finish { phong 1 } } scale <0.01,0.01,0.01> translate<'+
             str(coordinate[0])+','+str(coordinate[1])+','+str(coordinate[2])+'>} \n')


# x = np.random.randint(0,255,(500,500),dtype=np.uint8)
# # cv2.applyColorMap()
# print(x)
# im_color = cv2.applyColorMap(x, cv2.COLORMAP_JET)
# print(im_color)
# cv2.imshow('11',im_color)
# cv2.waitKey()
creat_povray_mesh_of_numberical_verification(['figures/Mesh/povray/query_deal.inc'])

