'''
    This folder contains the noise augmentation function
'''


def add_gaussian_noise(data):
    return data


def add_uniform_noise(data):
    return data


def add_random_noise(data):
    return data

def add_none_noise(data):
    return data


noise_types = {
    'gaussian': add_gaussian_noise,
    'random': add_random_noise,
    'uniform': add_uniform_noise,
    'none': add_none_noise
}