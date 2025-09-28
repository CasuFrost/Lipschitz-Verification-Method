'''
This file contains the function that create the set of the samples by dividing 
the hypercube in small disjoint hypercubes, taking the center as the samples.

A sample is discarded if his hypercube of side length 2epsilon doesn't intersects 
the unit sphere.
'''

import numpy as np
import itertools
import matplotlib.pyplot as plt
import os
from lib.utilities import file_to_array,array_to_file

def grid_on_cube(epsilon = 0.1, n = 4):
    '''
    This function consider the disjoint hypercubes of side length 2epsilon and return 
    their interval
    '''
    number_of_centers = 1/epsilon

    if number_of_centers%1!=0:
        print("Error : 1/epsilon should be a natural number.")
        exit(1)

    interval = []
    for i in range(int(number_of_centers)):
        interval.append(-1+epsilon+i*2*epsilon)

    return list(itertools.product(interval, repeat=n))

def check_sphere_intersection(point,epsilon=0.1):
    '''
    Given a point, this function return true if the hypercube of side length 2epsilon
    intersects the unit sphere
    '''
    n = len(point)
    distanza_minima_quadrata = 0
    distanza_massima_quadrata = 0
    for i in range(n):
        xi = point[i]
        dist_min_coord = max(0, abs(xi) - epsilon)
        distanza_minima_quadrata += dist_min_coord**2
        dist_max_coord = abs(xi) + epsilon
        distanza_massima_quadrata += dist_max_coord**2
    interseca = (distanza_minima_quadrata <= 1) and (distanza_massima_quadrata >= 1)
    return interseca

def get_samples(epsilon=0.1,n=4):
    path = 'samples/n_'+str(n)+'epsilon_'+str(epsilon)
    filename = 'samples.txt'
    if os.path.exists(path+'/'+filename):
        return file_to_array(path+'/'+filename)
    grid = grid_on_cube(epsilon,n)
    samples = []
    for p in grid:
        if check_sphere_intersection(p,epsilon):
            samples.append(list(p))
    array_to_file(samples,path,filename)
    return samples

