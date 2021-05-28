import os
import MDS
from simple_model_mass_test import create_data_set
from simple_model_mass_test import create_network

#verify if the distribution of the weights is even
model = create_network(channals=10)
root_path = 'F:/chaotic_similarity/randominit_sameorder'
weights_path = root_path+'/weights'
file_list = os.listdir(weights_path)
final_list = []
init_list = []
for i in range(len(file_list)):
    final_list.append( weights_path+'/'+file_list[i]+'/19E238b.h5')
    init_list.append( weights_path+'/'+file_list[i]+'/0E200b.h5')

# MDS.mds_single_analysis(model,final_list,root_path+'/mds2d.csv',dim = 2)
# MDS.mds_single_analysis(model,final_list,root_path+'/mds3d.csv',dim = 3)
MDS.mds_single_analysis(model,init_list,root_path+'/mds2d_0E200b.csv',dim = 2)
MDS.mds_single_analysis(model,init_list,root_path+'/mds3d_0E200b.csv',dim = 3)