import numpy as np

def xxx_output():
    return [1,2,3]
def xxxx_o():
    return [1,1]
oa = []
oa.append(xxx_output())
oa.append(xxxx_o())

oa = np.array(oa)
oa= np.reshape(oa,(5,1))
print(oa)
# oa.append(np.array([1,2,3,4]))
# x = np.array((1,2),dtype=float)
# y = [1,2]
#
# print(x,type(x))
# print(y,type(y))
# y = (1,2,3)
# x = [(1,1),(2,2),(3,3)]
# x = np.reshape(x,[6,1])
#
# print(x,type(x),type(x[0]))
# print(y, type(y))
# oa.append(x)
# oa.append(np.array([1,2,3,4],dtype=float))
# oa = np.array(oa,dtype=float)
# # oa = np.reshape(oa,[9,1])
# print(oa)
# print(oa.shape, type(oa),len(oa))