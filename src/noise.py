'''
    This folder contains the noise augmentation function
'''
import numpy as np

def add_gaussian_noise(data, mu=0, sigma=0.05):
    data = data + np.random.normal(mu, sigma, [data.shape[0], data.shape[1]]) 
    data = np.minimum(1, data)
    data = np.maximum(data, 0)
    return data


def add_uniform_noise(data, low=0, high=0.05):
    data = data + np.random.uniform(low, high, [data.shape[0], data.shape[1]]) 
    data = np.minimum(1, data)
    data = np.maximum(data, 0)
    return data

def add_random_noise(data, max_noise=0.05):
    data = data + max_noise*np.random.rand((data.shape[0], data.shape[1])) 
    data = np.minimum(1, data)
    data = np.maximum(data, 0)
    return data

def add_none_noise(data):
    return data


noise_types = {
    'gaussian': add_gaussian_noise,
    'random': add_random_noise,
    'uniform': add_uniform_noise,
    'none': add_none_noise
}