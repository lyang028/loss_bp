import numpy as np
import random
import dataReader as dr
import matplotlib.pyplot as plt

def generate_centers(number):
    centers = []
    for i in range(number):
        centers.append(np.random.rand(2))
    return np.array(centers)

def draw_scatter(centers, radius):
    centers[:,]
    plt.scatter()