import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import cv2

x,y = np.mgrid[-5:5:200j,-5:5:200j]
sigma = 0.5
c1 = [0.5,0.5]
c2 = [4,1]
def nomal(x,y,sigma,mu_x,mu_y):
    return 1/(2 * np.pi * (sigma**2)) * np.exp(-((x-mu_x)**2+(y-mu_y)**2)/(2 * sigma**2))
# z = 1/(2 * np.pi * (sigma**2)) * np.exp(-(x**2+y**2)/(2 * sigma**2)) + 1/(2 * np.pi * (sigma**2)) * np.exp(-((x-2)**2+(y-1)**2)/(2 * sigma**2))

def draw_example_surface():
    z = nomal(x, y, 0.8, 0, 0)
    r_x = 5
    r_y = 5
    for i in range(30):
        cx = random.uniform(-r_x, r_x)
        cy = random.uniform(-r_y, r_y)
        sigma = random.uniform(0.5, 0.8)
        z += nomal(x, y, sigma, cx, cy)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='rainbow', alpha=0.9)
    ax = plt.gca()
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.axes.zaxis.set_ticklabels([])
    plt.show()

def draw_example_contour():
    z = nomal(x, y, 0.8, 0, 0)
    r_x = 5
    r_y = 5
    for i in range(30):
        cx = random.uniform(-r_x, r_x)
        cy = random.uniform(-r_y, r_y)
        sigma = random.uniform(0.5, 0.8)
        z += nomal(x, y, sigma, cx, cy)


    zoom = z.max() - z.min()
    rag = np.arange(0, 10, 0.001) * zoom / 10 + z.min() + 0.0001
    plt.contour(x, y, z, rag, cmap='rainbow')
    ax = plt.gca()
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    plt.show()

draw_example_contour()