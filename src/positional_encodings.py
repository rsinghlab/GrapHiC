'''
    This file contains the source code for all the positional encoding methods 
    we have at one point or the other used to evaluate the performance of GrapHiC
'''
import numpy as np
from scipy.sparse.csgraph import laplacian
from numpy.linalg import eig



def constant_positional_encoding(matrix):
    '''
        The most basic positional encoding method, we assign each node the same position of 1
        @params: matrix <np.array>, 2D Hi-C matrix array
        @returns: <np.array> 1D that contains the node positions
    '''
    return np.ones((matrix.shape[0], 1))


def monotonic_positional_encoding(matrix, type='fractional'):
    '''
        A positional encoding method that assigns monotonically increasing node position
        @params: matrix <np.array> 2D Hi-C matrix array
        @params: type <string>, fractional defines the max value to be 1 and integer defines the max 
                                value to be the matrix.shape[0].
        @params: <np.array> 1D that contains the node positions
    '''
    # Adding a small value so the starting value is not zero
    if type == 'fractional':
        return np.reshape(np.linspace(0.0, 1.0, num=matrix.shape[0]), (matrix.shape[0], 1))
    elif type == 'integer':
        return np.reshape(np.linspace(0.0, matrix.shape[0], num=matrix.shape[0]), (matrix.shape[0], 1))
    else:
        print('Invalid type provided to monotonic positional encoding function')
        exit(1)


def transformer_positional_encoding(matrix, encoding_dim=4, padding_idx=None):
    '''
        A positional encoding method that assigns transformer sine-cosine node position
        @params: matrix <np.array> 2D Hi-C matrix array
        @params: type <string>, fractional defines the max value to be 1 and integer defines the max 
                                value to be the matrix.shape[0].
        @params: <np.array> 1D that contains the node positions
    '''
    n_position = matrix.shape[0]
   
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / encoding_dim)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(encoding_dim)]

   

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return sinusoid_table


def graph_positional_encoding(matrix, encoding_dim=4, upscale=255):
    '''
        A positional encoding method that assigns eigen vectors of laplacian matrix as node position
        @params: matrix <np.array> 2D Hi-C matrix array
        @params: type <string>, fractional defines the max value to be 1 and integer defines the max 
                                value to be the matrix.shape[0].
        @params: <np.array> 1D that contains the node positions
    '''
    # Convert the matrix to laplacian form 
    L = (matrix*upscale).astype(np.int32)
    L = laplacian(L)
    # Get eigen values and eigen vectors 
    eigen_vals, eigen_vecs = eig(L)
    # Sort on eigen values
    idx = eigen_vals.argsort()

    eigen_vals, eigen_vecs = eigen_vals[idx], np.real(eigen_vecs[:,idx])

    return eigen_vecs[:,1:encoding_dim+1]




encoding_methods = {
    'constant': constant_positional_encoding,
    'monotonic': monotonic_positional_encoding,
    'transformer': transformer_positional_encoding,
    'graph': graph_positional_encoding
}