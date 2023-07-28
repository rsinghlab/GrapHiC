import os
import json
import numpy as np
import scipy.sparse as sp
import pandas as pd
from typing import Union
from contextlib import suppress
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.spatial.distance import euclidean
from scipy.sparse.csgraph import laplacian
from numpy.linalg import eig


def invalid_comparisons(x, y):
    if np.all(x.flatten() == y.flatten()):
        return True
    if np.all(x == x[0]) or np.all(y == y[0]):
        return True
    return False

def mse(args):
    x, y = args

    if invalid_comparisons(x, y):
        return -100
    return mean_squared_error(x, y)

def mae(args):
    x, y = args

    if invalid_comparisons(x, y):
        return -100
    return mean_absolute_error(x, y)

def psnr(args):
    x, y = args
    if invalid_comparisons(x, y):
        return -100
    
    data_range = np.max(x) - np.min(x)
    err = mean_squared_error(x, y)
    if err == 0:
        return 100
    else:
        return 10*np.log10((data_range**2)/err)

def ssim(args):
    x, y = args
    if invalid_comparisons(x, y):
        return -100
    
    if (x.flatten() == y.flatten()).all():
        return -100
    return structural_similarity(x, y, data_range=2)


def pearsons(args):
    x, y = args

    if invalid_comparisons(x, y):
        return -100
    
    r, _ = pearsonr(x.flatten(), y.flatten())
    return r

def spearmans(args):
    x, y = args

    if invalid_comparisons(x, y):
        return -100
    
    r, _ = spearmanr(x.flatten(), y.flatten())
    return r



list_of_eval_funcs = {
    'MSE': mse,
    # 'MAE': mae,
    # 'PSNR': psnr,
    'SSIM': ssim,
    'PCC': pearsons,
    'SCC': spearmans,
}

def evaluate(y, y_bar, func):
    """
        A wrapper function on list of all chunks and we apply the func that can be 
            [mse, mae, psnr, ssim, pearsons, spearmans or same]
            
        @params: y <np.array>, Base Chunks
        @params: y_bar <np.array> Predicted Chunks
        @returns <np.array>, list : all errors and statistics of the errors.
    """

    y_list = [y[x, 0, :, :] for x in range(y.shape[0])]
    y_bar_list = [y_bar[x, 0, :, :] for x in range(y_bar.shape[0])]
    args = zip(y_list, y_bar_list)
    
    errors = list(map(func, args))

    return errors




def compute_correlation_metrics(
    y, 
    y_bar
):
    results = {}
    for key, eval_func in list_of_eval_funcs.items():
        results[key] = evaluate(y, y_bar, eval_func)
    
    return results



