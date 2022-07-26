'''
    This folder contains the noise augmentation function
'''


def add_gaussian_noise(data):
    return data


def add_uniform_noise(data):
    return data


def add_random_noise(data):
    return data


noise_types = {
    'Gaussian': add_gaussian_noise,
    'Random': add_random_noise,
    'Uniform': add_uniform_noise
}