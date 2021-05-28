import numpy as np
import matplotlib.pyplot as plt
def sigmoid(a):
    x = 1/(1+np.exp(-a))
    return x

def relu(a):
    if a>0:
        return a
    else:
        return 0

def count(input,boundary):
    output = np.zeros(len(boundary))
    for i in range(len(input[0])):
        for j in range(len(boundary)):
            if input[0][i] < boundary[j]:
                output[j] += input[1][i]
                break

    return output

def map(input, function):
    output = []
    for i in input:
        output.append(function(i))
    return np.array(output)

def pmf(pdf):
    output = pdf
    for i in range(len(output)-1):
        output[i+1]+=output[i]
    return output
ori_pdf = np.ones(1000)*0.001
ori_output = np.array(range(1000))/500
input_ori = [ori_output,ori_pdf]
input_sigmoid = [map(ori_output,sigmoid), ori_pdf]
boundary = np.array(range(1000))/1000

pdf_count = count(input_sigmoid,boundary)



plt.plot(range(len(boundary)),pmf(pdf_count),label = 'new')
plt.plot(range(len(ori_output)),pmf(ori_pdf),label = 'ori')
plt.legend()
plt.show()




