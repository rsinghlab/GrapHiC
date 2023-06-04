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
    # x = x*255
    # y = y*255
    if invalid_comparisons(x, y):
        return -100
    return mean_squared_error(x, y)

def mae(args):
    x, y = args
    #x = x*255
    #y = y*255
    if invalid_comparisons(x, y):
        return -100
    return mean_absolute_error(x, y)

def psnr(args):
    x, y = args
    #x = x*255
    #y = y*255
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
    # x = x*255
    # y = y*255
    # if invalid_comparisons(x, y):
    #     return -100
    # #data_range = y.max()-y.min()
    # if (x.flatten() == y.flatten()).all():
    #     return -100
    # return structural_similarity(x, y)
    if invalid_comparisons(x, y):
        return -100
    data_range = x.max()-x.min()
    if (x.flatten() == y.flatten()).all():
        return -100
    return structural_similarity(x, y, data_range=data_range)


def pearsons(args):
    x, y = args
    #x = x*255
    #y = y*255
    if invalid_comparisons(x, y):
        return -100
    
    r, _ = pearsonr(x.flatten(), y.flatten())
    return r

def spearmans(args):
    x, y = args
    #x = x*255
    #y = y*255
    if invalid_comparisons(x, y):
        return -100
    
    r, _ = spearmanr(x.flatten(), y.flatten())
    return r

def hicrep(args):
    x, y = args
    if invalid_comparisons(x, y):
        return -100
    return hicrepSCC(x, y)


def hic_spector(args):
    x, y = args
    r = 20
    l = np.sqrt(2)

    x_laplacian = laplacian(x*255)
    y_laplacian = laplacian(y*255)
    _, x_eigen_vectors = eig(x_laplacian)
    _, y_eigen_vectors = eig(y_laplacian)
    
    score = 0.0

    for i in range(2):
        score = euclidean(x_eigen_vectors[-i], y_eigen_vectors[-i])

    score = score/r

    return score




'''
    Faster HiCRep Implementation
'''
def trimDiags(a: sp.coo_matrix, iDiagMax: int, bKeepMain: bool):
    """Remove diagonal elements whose diagonal index is >= iDiagMax
    or is == 0
    Args:
        a: Input scipy coo_matrix
        iDiagMax: Diagonal offset cutoff
        bKeepMain: If true, keep the elements in the main diagonal;
        otherwise remove them
    Returns:
        coo_matrix with the specified diagonals removed
    """
    gDist = np.abs(a.row - a.col)
    idx = np.where((gDist < iDiagMax) & (bKeepMain | (gDist != 0)))
    return sp.coo_matrix((a.data[idx], (a.row[idx], a.col[idx])),
                         shape=a.shape, dtype=a.dtype)


def resample(m: sp.coo_matrix, size: int):
    """Resample with replacement the input matrix so that the
    resulting matrix sum to the given size
    Args:
        m: `sp.coo_matrix` Input matrix
        size: Resulting matrix sum to this number
    Returns:
        resampled matrix
    """
    bins = np.arange(m.data.size)
    p = m.data / m.data.sum()
    samples = np.random.choice(bins, size=size, p=p)
    sampledData = np.bincount(samples, minlength=bins.size)
    ans = sp.coo_matrix((sampledData, (m.row, m.col)), shape=m.shape)
    ans.eliminate_zeros()
    return ans


def meanFilterSparse(a: sp.coo_matrix, h: int):
    """Apply a mean filter to an input sparse matrix. This convolves
    the input with a kernel of size 2*h + 1 with constant entries and
    subsequently reshape the output to be of the same shape as input
    Args:
        a: `sp.coo_matrix`, Input matrix to be filtered
        h: `int` half-size of the filter
    Returns:
        `sp.coo_matrix` filterd matrix
    """
    assert h > 0, "meanFilterSparse half-size must be greater than 0"
    assert sp.issparse(a) and a.getformat() == 'coo',\
        "meanFilterSparse input matrix is not scipy.sparse.coo_matrix"
    assert a.shape[0] == a.shape[1],\
        "meanFilterSparse cannot handle non-square matrix"
    fSize = 2 * h + 1
    # filter is a square matrix of constant 1 of shape (fSize, fSize)
    shapeOut = np.array(a.shape) + fSize - 1
    mToeplitz = sp.diags(np.ones(fSize),
                         np.arange(-fSize+1, 1),
                         shape=(shapeOut[1], a.shape[1]),
                         format='csr')
    ans = sp.coo_matrix((mToeplitz @ a) @ mToeplitz.T)
    # remove the edges since we don't care about them if we are smoothing
    # the matrix itself
    ansNoEdge = ans.tocsr()[h:(h+a.shape[0]), h:(h+a.shape[1])].tocoo()
    # Assign different number of neighbors to the edge to better
    # match what the original R implementation of HiCRep does
    rowDist2Edge = np.minimum(ansNoEdge.row, ansNoEdge.shape[0] - 1 - ansNoEdge.row)
    nDim1 = h + 1 + np.minimum(rowDist2Edge, h)
    colDist2Edge = np.minimum(ansNoEdge.col, ansNoEdge.shape[1] - 1 - ansNoEdge.col)
    nDim2 = h + 1 + np.minimum(colDist2Edge, h)
    nNeighbors = nDim1 * nDim2
    ansNoEdge.data /= nNeighbors
    return ansNoEdge

def upperDiagCsr(m: sp.coo_matrix, nDiags: int):
    """Convert an input sp.coo_matrix into a sp.csr_matrix where each row in the
    the output corresponds to one diagonal of the upper triangle of the input.
    Args:
        m (sp.coo_matrix): input matrix
        nDiags (int): output diagonals with index in the range [1, nDiags)
        as rows of the output matrix
    Returns: `sp.csr_matrix` whose rows are the diagonals of the input
    """
    row = m.col - m.row
    idx = np.where((row > 0) & (row < nDiags))
    idxRowp1 = row[idx]
    # the diagonal index becomes the row index
    idxRow = idxRowp1 - 1
    # offset in the original diagonal becomes the column index
    idxCol = m.col[idx] - idxRowp1
    ans = sp.csr_matrix((m.data[idx], (idxRow, idxCol)),
                        shape=(nDiags - 1, m.shape[1]), dtype=m.dtype)
    ans.eliminate_zeros()
    return ans

def meanFilterSparse(a: sp.coo_matrix, h: int):
    """Apply a mean filter to an input sparse matrix. This convolves
    the input with a kernel of size 2*h + 1 with constant entries and
    subsequently reshape the output to be of the same shape as input
    Args:
        a: `sp.coo_matrix`, Input matrix to be filtered
        h: `int` half-size of the filter
    Returns:
        `sp.coo_matrix` filterd matrix
    """
    assert h > 0, "meanFilterSparse half-size must be greater than 0"
    assert sp.issparse(a) and a.getformat() == 'coo',\
        "meanFilterSparse input matrix is not scipy.sparse.coo_matrix"
    assert a.shape[0] == a.shape[1],\
        "meanFilterSparse cannot handle non-square matrix"
    fSize = 2 * h + 1
    # filter is a square matrix of constant 1 of shape (fSize, fSize)
    shapeOut = np.array(a.shape) + fSize - 1
    mToeplitz = sp.diags(np.ones(fSize),
                         np.arange(-fSize+1, 1),
                         shape=(shapeOut[1], a.shape[1]),
                         format='csr')
    ans = sp.coo_matrix((mToeplitz @ a) @ mToeplitz.T)
    # remove the edges since we don't care about them if we are smoothing
    # the matrix itself
    ansNoEdge = ans.tocsr()[h:(h+a.shape[0]), h:(h+a.shape[1])].tocoo()
    # Assign different number of neighbors to the edge to better
    # match what the original R implementation of HiCRep does
    rowDist2Edge = np.minimum(ansNoEdge.row, ansNoEdge.shape[0] - 1 - ansNoEdge.row)
    nDim1 = h + 1 + np.minimum(rowDist2Edge, h)
    colDist2Edge = np.minimum(ansNoEdge.col, ansNoEdge.shape[1] - 1 - ansNoEdge.col)
    nDim2 = h + 1 + np.minimum(colDist2Edge, h)
    nNeighbors = nDim1 * nDim2
    ansNoEdge.data /= nNeighbors
    return ansNoEdge

def varVstran(n: Union[int, np.ndarray]):
    """
    Calculate the variance of variance-stabilizing transformed
    (or `vstran()` in the original R implementation) data. The `vstran()` turns
    the input data into ranks, whose variance is only a function of the input
    size:
        ```
        var(1/n, 2/n, ..., n/n) = (1 - 1/(n^2))/12
        ```
    or with Bessel's correction:
        ```
        var(1/n, 2/n, ..., n/n, ddof=1) = (1 + 1.0/n)/12
        ```
    See section "Variance stabilized weights" in reference for more detail:
    https://genome.cshlp.org/content/early/2017/10/06/gr.220640.117
    Args:
        n (Union(int, np.ndarray)): size of the input data
    Returns: `Union(int, np.ndarray)` variance of the ranked input data with Bessel's
    correction
    """
    with suppress(ZeroDivisionError), np.errstate(divide='ignore', invalid='ignore'):
        return np.where(n < 2, np.nan, (1 + 1.0 / n) / 12.0)


def resample(m: sp.coo_matrix, size: int):
    """Resample with replacement the input matrix so that the
    resulting matrix sum to the given size
    Args:
        m: `sp.coo_matrix` Input matrix
        size: Resulting matrix sum to this number
    Returns:
        resampled matrix
    """
    bins = np.arange(m.data.size)
    p = m.data
    p[p<0] = 0
    p = p / p.sum()

    samples = np.random.choice(bins, size=size, p=p)
    sampledData = np.bincount(samples, minlength=bins.size)
    ans = sp.coo_matrix((sampledData, (m.row, m.col)), shape=m.shape)
    ans.eliminate_zeros()
    return ans


def sccByDiag(m1: sp.coo_matrix, m2: sp.coo_matrix, nDiags: int):
    """Compute diagonal-wise hicrep SCC score for the two input matrices up to
    nDiags diagonals
    Args:
        m1 (sp.coo_matrix): input contact matrix 1
        m2 (sp.coo_matrix): input contact matrix 2
        nDiags (int): compute SCC scores for diagonals whose index is in the
        range of [1, nDiags)
    Returns: `float` hicrep SCC scores
    """
    # convert each diagonal to one row of a csr_matrix in order to compute
    # diagonal-wise correlation between m1 and m2
    m1D = upperDiagCsr(m1, nDiags)
    m2D = upperDiagCsr(m2, nDiags)
    nSamplesD = (m1D + m2D).getnnz(axis=1)
    rowSumM1D = m1D.sum(axis=1).A1
    rowSumM2D = m2D.sum(axis=1).A1
    # ignore zero-division warnings because the corresponding elements in the
    # output don't contribute to the SCC scores
    with np.errstate(divide='ignore', invalid='ignore'):
        cov = m1D.multiply(m2D).sum(axis=1).A1 - rowSumM1D * rowSumM2D / nSamplesD
        rhoD = cov / np.sqrt(
            (m1D.power(2).sum(axis=1).A1 - np.square(rowSumM1D) / nSamplesD ) *
            (m2D.power(2).sum(axis=1).A1 - np.square(rowSumM2D) / nSamplesD ))
        wsD = nSamplesD * varVstran(nSamplesD)
        # Convert NaN and Inf resulting from div by 0 to zeros.
        # posinf and neginf added to fix behavior seen in 4DN datasets
        # 4DNFIOQLTI9G and DNFIH7MQHOR at 5kb where inf would be reported
        # as an SCC score
        wsNan2Zero = np.nan_to_num(wsD, copy=True, posinf=0.0, neginf=0.0)
        rhoNan2Zero = np.nan_to_num(rhoD, copy=True, posinf=0.0, neginf=0.0)

    return rhoNan2Zero @ wsNan2Zero / wsNan2Zero.sum()


def hicrepSCC(mat1: np.ndarray, mat2: np.ndarray,
              h: int = 5, dBPMax: int = None, bDownSample: bool = False):
    """Compute hicrep score between two input Cooler contact matrices
    Args:
        cool1: `cooler.api.Cooler` Input Cooler contact matrix 1
        cool2: `cooler.api.Cooler` Input Cooler contact matrix 2
        h: `int` Half-size of the mean filter used to smooth the
        input matrics
        dBPMax `int` Only include contacts that are at most this genomic
        distance (bp) away
        bDownSample: `bool` Down sample the input with more contacts
        to the same number of contacts as in the other input
    Returns:
        `float` scc scores for each chromosome
    """
    assert mat1.shape == mat2.shape
    assert mat1.shape[-1] == mat1.shape[-2]
    if not dBPMax:
        # set the dBPMax to max size of the input matrices
        dBPMax = mat1.shape[-1]
    
    n1 = np.sum(mat1)
    n2 = np.sum(mat2)


    m1 = sp.coo_matrix(mat1)
    m2 = sp.coo_matrix(mat2)
    
    
   
    # remove major diagonal and all the diagonals >= nDiags
    # to save computation time
    m1 = trimDiags(m1, dBPMax, False)
    m2 = trimDiags(m2, dBPMax, False)

    if bDownSample:
        # do downsampling
        size1 = m1.sum()
        size2 = m2.sum()
        if size1 > size2:
            m1 = resample(m1, size2).astype(float)
        elif size2 > size1:
            m2 = resample(m2, size1).astype(float)
    else:
        # just normalize by total contacts
        m1 = m1.astype(float) / n1
        m2 = m2.astype(float) / n2

    if h > 0:
        # apply smoothing
        m1 = meanFilterSparse(m1, h)
        m2 = meanFilterSparse(m2, h)

    return sccByDiag(m1, m2, dBPMax)
    
















list_of_eval_funcs = {
    'MSE': mse,
    # 'MAE': mae,
    # 'PSNR': psnr,
    'SSIM': ssim,
    'PCC': pearsons,
    'SCC': spearmans,
    'HiCRep': hicrep,
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



