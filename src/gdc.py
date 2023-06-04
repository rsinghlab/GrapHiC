import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg




def gdc(matrix, alpha, eps=0.0001):
    '''
        A graph diffusion based pre-processing step 
    '''
    # ensure that matrix has shape (N, N)
    if len(matrix.shape) == 3:
        matrix = matrix[0, :, :]
    
    # Converting into a sparse matrix so I can directly use the code
    A = sp.csr_matrix(matrix)
    
    N = A.shape[0]

    # Self-loops
    A_loop = sp.eye(N) + A

    # Symmetric transition matrix
    D_loop_vec = A_loop.sum(0).A1
    D_loop_vec_invsqrt = 1 / np.sqrt(D_loop_vec)
    D_loop_invsqrt = sp.diags(D_loop_vec_invsqrt)
    T_sym = D_loop_invsqrt @ A_loop @ D_loop_invsqrt

    # PPR-based diffusion
    S = alpha * splinalg.inv(sp.eye(N) - (1 - alpha) * T_sym)

    # Sparsify using threshold epsilon
    S_tilde = S.multiply(S >= eps)

    # Column-normalized transition matrix on graph S_tilde
    D_tilde_vec = S_tilde.sum(0).A1
    T_S = S_tilde / D_tilde_vec
    
    return T_S







