from copy import deepcopy
import numpy as np
import pandas as pd
import sys


'''
In this problem you write your own K-Means
Clustering code.

Your code should return a 2d array containing
the centers.

'''
# Import the dataset
data = np.loadtxt("data/data/iris.data", delimiter = ",", skiprows = 1, usecols = (0,1,2,3))

# Make 3  clusters
k = 3
# Initial Centroids
C = [[2.,  0.,  3.,  4.], [1.,  2.,  1.,  3.], [0., 2.,  1.,  0.]]
C = np.array(C)
print("Initial Centers")
print(C)

def k_means(C):
    converge = False
    while converge == False:
        class_pred_array = pred_class(C)
        C_final = update_center(class_pred_array)
        converge = converge_con(C_final, C)
        C = C_final

    return C_final


def pred_class(C):
    class_pred_list = []
    for sample in data:
        dist = np.sum(np.square(sample - C), axis=1)
        class_pred = np.argmin(dist)
        class_pred_list.append(class_pred)

    return np.array(class_pred_list)


def update_center(class_pred_array):
    new_center_list = []
    for class_index in range(k):
        sample_num = np.sum(class_pred_array == class_index)

        sample_index = np.where(class_pred_array == class_index)
        sample = data[sample_index]
        sample_sum = np.sum(sample, axis=0)

        new_center = sample_sum / sample_num
        new_center_list.append(new_center)

    return np.array(new_center_list)


def converge_con(new_center_array, old_center_array):
    converge = False
    diff_sum = np.sum(np.square(new_center_array - old_center_array))

    if diff_sum < 10e-3:
        converge = True

    return converge






