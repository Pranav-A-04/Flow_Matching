import torch as th
import numpy as np


def create_checkerboard(resolution):
    """
    Create a checkerboard pattern of given resolution.
    """
    
    N = 1000
    x_min, x_max = -4, 4
    y_min, y_max = -4, 4

    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y)

    length = 4
    checkerboard = np.indices((resolution, resolution)).sum(axis=0) % 2

    sample_points = []
    while len(sample_points) < N:
        x_sample = np.random.uniform(x_min, x_max, N)
        y_sample = np.random.uniform(y_min, y_max, N)
        
        i = int((x_sample - x_min)/(x_max - x_min) * length)
        j = int((y_sample - y_min)/(y_max - y_min) * length)
        
        if checkerboard[j, i] == 1:
            sample_points.append((x_sample, y_sample))

    sampled_points = np.array(sample_points).T
    return sampled_points

