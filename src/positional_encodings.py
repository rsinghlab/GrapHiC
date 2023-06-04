'''
    This file contains the source code for all the positional encoding methods 
    we have at one point or the other used to evaluate the performance of GrapHiC
'''
import numpy as np
from scipy.sparse.csgraph import laplacian
from numpy.linalg import eig
import torch
import torch.nn.functional as F
from scipy.sparse.csgraph import laplacian
from scipy import sparse
from torch_geometric.utils import from_scipy_sparse_matrix, to_dense_adj, scatter, get_laplacian, to_scipy_sparse_matrix


def constant_positional_encoding(matrix, encoding_dim=4):
    '''
        The most basic positional encoding method, we assign each node the same position of 1
        @params: matrix <np.array>, 2D Hi-C matrix array
        @returns: <np.array> 1D that contains the node positions
    '''
    # Ensure the input adjacency matrix is of shape 2
    if len(matrix.shape) == 3:
        matrix = matrix[0]
    
    return np.ones((matrix.shape[0], encoding_dim))


def monotonic_positional_encoding(matrix, encoding_dim=4, type='fractional'):
    '''
        A positional encoding method that assigns monotonically increasing node position
        @params: matrix <np.array> 2D Hi-C matrix array
        @params: type <string>, fractional defines the max value to be 1 and integer defines the max 
                                value to be the matrix.shape[0].
        @params: <np.array> 1D that contains the node positions
    '''
    # Ensure the input adjacency matrix is of shape 2
    if len(matrix.shape) == 3:
        matrix = matrix[0]
    
    # Adding a small value so the starting value is not zero
    if type == 'fractional':
        monotonic_encoding = []
        for i in range(encoding_dim):
            monotonic_encoding.append([i for i in range(0, matrix.shape[0])])


        monotonic_encoding = np.array(monotonic_encoding)
        monotonic_encoding.astype(float)

        monotonic_encoding = monotonic_encoding/float(matrix.shape[0])

        return monotonic_encoding.T

    elif type == 'integer':
        monotonic_encoding = []
        for i in range(encoding_dim):
            monotonic_encoding.append([i for i in range(0, matrix.shape[0])])


        monotonic_encoding = np.array(monotonic_encoding)
        monotonic_encoding.astype(float)

        return monotonic_encoding.T
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
    # Ensure the input adjacency matrix is of shape 2
    if len(matrix.shape) == 3:
        matrix = matrix[0]
    
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
    sinusoid_table = np.array(sinusoid_table, dtype=float)
    return sinusoid_table


def graph_positional_encoding(matrix, encoding_dim=4, upscale=255):
    '''
        A positional encoding method that assigns eigen vectors of laplacian matrix as node position
        @params: matrix <np.array> 2D Hi-C matrix array
        @params: type <string>, fractional defines the max value to be 1 and integer defines the max 
                                value to be the matrix.shape[0].
        @params: <np.array> 1D that contains the node positions
    '''
    # Ensure the input adjacency matrix is of shape 2
    if len(matrix.shape) == 3:
        matrix = matrix[0]
    
    # Convert the matrix to laplacian form 
    L = (matrix*upscale).astype(np.int32)
    L = laplacian(L)
    # Get eigen values and eigen vectors 
    eigen_vals, eigen_vecs = eig(L)
    # Sort on eigen values
    idx = eigen_vals.argsort()

    eigen_vals, eigen_vecs = eigen_vals[idx], np.real(eigen_vecs[:,idx])
    eigen_vecs = np.array(eigen_vecs[:,1:encoding_dim+1], dtype=float)
    
    return eigen_vecs



def eigvec_normalizer(EigVecs, EigVals, normalization="L2", eps=1e-12):
    """
    Implement different eigenvector normalizations.
    """

    EigVals = EigVals.unsqueeze(0)

    if normalization == "L1":
        # L1 normalization: eigvec / sum(abs(eigvec))
        denom = EigVecs.norm(p=1, dim=0, keepdim=True)

    elif normalization == "L2":
        # L2 normalization: eigvec / sqrt(sum(eigvec^2))
        denom = EigVecs.norm(p=2, dim=0, keepdim=True)

    elif normalization == "abs-max":
        # AbsMax normalization: eigvec / max|eigvec|
        denom = torch.max(EigVecs.abs(), dim=0, keepdim=True).values

    elif normalization == "wavelength":
        # AbsMax normalization, followed by wavelength multiplication:
        # eigvec * pi / (2 * max|eigvec| * sqrt(eigval))
        denom = torch.max(EigVecs.abs(), dim=0, keepdim=True).values
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = denom * eigval_denom * 2 / np.pi

    elif normalization == "wavelength-asin":
        # AbsMax normalization, followed by arcsin and wavelength multiplication:
        # arcsin(eigvec / max|eigvec|)  /  sqrt(eigval)
        denom_temp = torch.max(EigVecs.abs(), dim=0, keepdim=True).values.clamp_min(eps).expand_as(EigVecs)
        EigVecs = torch.asin(EigVecs / denom_temp)
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = eigval_denom

    elif normalization == "wavelength-soft":
        # AbsSoftmax normalization, followed by wavelength multiplication:
        # eigvec / (softmax|eigvec| * sqrt(eigval))
        denom = (F.softmax(EigVecs.abs(), dim=0) * EigVecs.abs()).sum(dim=0, keepdim=True)
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = denom * eigval_denom

    else:
        raise ValueError(f"Unsupported normalization `{normalization}`")

    denom = denom.clamp_min(eps).expand_as(EigVecs)
    EigVecs = EigVecs / denom
    
    # Finally put it between 0 and 1 if abs max
    if normalization == 'abs-max':
        EigVecs = (EigVecs + 1)/2.0    
    
    
    return EigVecs


def graph_laplacian_pe_encodings(matrix, encoding_dim=4, lap_norm='sym', eig_norm='L2'):
    """Compute Laplacian eigen-decomposition-based PE stats of the given graph.
    Args:
        evals, evects: Precomputed eigen-decomposition
        max_freqs: Maximum number of top smallest frequencies / eigenvecs to use
        eigvec_norm: Normalization for the eigen vectors of the Laplacian
    Returns:
        Tensor (num_nodes, max_freqs, 1) eigenvalues repeated for each node
        Tensor (num_nodes, max_freqs) of eigenvector values per node
    """
    if len(matrix.shape) == 3:
        matrix  = matrix[0, :, :]
    N = matrix.shape[0]
    sparse_matrix = sparse.csr_matrix(matrix)
    edge_index, edge_weight = from_scipy_sparse_matrix(sparse_matrix)
    
    edge_index, edge_weight = get_laplacian(edge_index, edge_weight, normalization=lap_norm, num_nodes=N)
    L = to_scipy_sparse_matrix(edge_index, edge_weight, N)

    evals, evects = np.linalg.eigh(L.toarray())
    
    # Keep up to the maximum desired number of frequencies.
    idx = evals.argsort()[:encoding_dim]
    evals, evects = evals[idx], np.real(evects[:, idx])
    evals = torch.from_numpy(np.real(evals)).clamp_min(0)

    # Normalize and pad eigen vectors.
    evects = torch.from_numpy(evects).float()
    evects = eigvec_normalizer(evects, evals, normalization=eig_norm)
    if N < encoding_dim:
        EigVecs = F.pad(evects, (0, encoding_dim - N), value=float('nan'))
    else:
        EigVecs = evects

    # Pad and save eigenvalues.
    if N < encoding_dim:
        EigVals = F.pad(evals, (0, encoding_dim - N), value=float('nan')).unsqueeze(0)
    else:
        EigVals = evals.unsqueeze(0)
    EigVals = EigVals.repeat(N, 1).unsqueeze(2)
    
    return EigVecs.cpu().detach().numpy()



def get_rw_landing_probs(matrix, encoding_dim, space_dim=0):
    """Compute Random Walk landing probabilities for given list of K steps.
    Args:
        matrix: Adjacency matrix
        ksteps: List of k-steps for which to compute the RW landings
        space_dim: (optional) Estimated dimensionality of the space. Used to
            correct the random-walk diagonal by a factor `k^(space_dim/2)`.
            In euclidean space, this correction means that the height of
            the gaussian distribution stays almost constant across the number of
            steps, if `space_dim` is the dimension of the euclidean space.
    Returns:
        2D Tensor with shape (num_nodes, len(ksteps)) with RW landing probs
    """
    if len(matrix.shape) == 3:
        matrix  = matrix[0, :, :]
    ksteps = range(encoding_dim)
    num_nodes = matrix.shape[0]
    sparse_matrix = sparse.csr_matrix(matrix)
    edge_index, edge_weight = from_scipy_sparse_matrix(sparse_matrix)
    
    
    source, dest = edge_index[0], edge_index[1]
    deg = scatter(edge_weight, source, dim=0, dim_size=num_nodes, reduce='sum')  # Out degrees.
    deg_inv = deg.pow(-1.)
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)
    
    if edge_index.numel() == 0:
        P = edge_index.new_zeros((1, num_nodes, num_nodes))
    else:
        # P = D^-1 * A
        A = to_dense_adj(edge_index, max_num_nodes=num_nodes)  # 1 x (Num nodes) x (Num nodes)
        D_inv = torch.diag(deg_inv).to(torch.float)
        P = D_inv @ A
    
    rws = []
    if ksteps == list(range(min(ksteps), max(ksteps) + 1)):
        # Efficient way if ksteps are a consecutive sequence (most of the time the case)
        Pk = P.clone().detach().matrix_power(min(ksteps))
        for k in range(min(ksteps), max(ksteps) + 1):
            rws.append(torch.diagonal(Pk, dim1=-2, dim2=-1) * \
                       (k ** (space_dim / 2)))
            Pk = Pk @ P
    else:
        # Explicitly raising P to power k for each k \in ksteps.
        for k in ksteps:
            rws.append(torch.diagonal(P.matrix_power(k), dim1=-2, dim2=-1) * \
                       (k ** (space_dim / 2)))
    rw_landing = torch.cat(rws, dim=0).transpose(0, 1)  # (Num nodes) x (K steps)

    return rw_landing.cpu().detach().numpy()

def get_heat_kernels_diag(matrix, encoding_dim, lap_norm='sym', space_dim=0):
    """Compute Heat kernel diagonal.
    This is a continuous function that represents a Gaussian in the Euclidean
    space, and is the solution to the diffusion equation.
    The random-walk diagonal should converge to this.
    Args:
        evects: Eigenvectors of the Laplacian matrix
        evals: Eigenvalues of the Laplacian matrix
        kernel_times: Time for the diffusion. Analogous to the k-steps in random
            walk. The time is equivalent to the variance of the kernel.
        space_dim: (optional) Estimated dimensionality of the space. Used to
            correct the diffusion diagonal by a factor `t^(space_dim/2)`. In
            euclidean space, this correction means that the height of the
            gaussian stays constant across time, if `space_dim` is the dimension
            of the euclidean space.
    Returns:
        2D Tensor with shape (num_nodes, len(ksteps)) with RW landing probs
    """
    if len(matrix.shape) == 3:
        matrix  = matrix[0, :, :]
        
    N = matrix.shape[0]
    sparse_matrix = sparse.csr_matrix(matrix)
    edge_index, edge_weight = from_scipy_sparse_matrix(sparse_matrix)
    
    edge_index, edge_weight = get_laplacian(edge_index, edge_weight, normalization=lap_norm, num_nodes=N)
    L = to_scipy_sparse_matrix(edge_index, edge_weight, N)

    evals, evects = np.linalg.eigh(L.toarray())
    
    evals = torch.from_numpy(evals)
    evects = torch.from_numpy(evects)
    
    kernel_times = range(encoding_dim)
    
    
    heat_kernels_diag = []
    if len(kernel_times) > 0:
        evects = F.normalize(evects, p=2., dim=0)

        # Remove eigenvalues == 0 from the computation of the heat kernel
        idx_remove = evals < 1e-8
        evals = evals[~idx_remove]
        evects = evects[:, ~idx_remove]

        # Change the shapes for the computations
        evals = evals.unsqueeze(-1)  # lambda_{i, ..., ...}
        evects = evects.transpose(0, 1)  # phi_{i,j}: i-th eigvec X j-th node

        # Compute the heat kernels diagonal only for each time
        eigvec_mul = evects ** 2
        for t in kernel_times:
            # sum_{i>0}(exp(-2 t lambda_i) * phi_{i, j} * phi_{i, j})
            this_kernel = torch.sum(torch.exp(-t * evals) * eigvec_mul,
                                    dim=0, keepdim=False)

            # Multiply by `t` to stabilize the values, since the gaussian height
            # is proportional to `1/t`
            heat_kernels_diag.append(this_kernel * (t ** (space_dim / 2)))
        heat_kernels_diag = torch.stack(heat_kernels_diag, dim=0).transpose(0, 1)

    return heat_kernels_diag.cpu().detach().numpy()


    

encoding_methods = {
    'constant': constant_positional_encoding,
    'monotonic': monotonic_positional_encoding,
    'transformer': transformer_positional_encoding,
    'graph': graph_positional_encoding,
    'graph_lap_pe': graph_laplacian_pe_encodings,
    'rw_se': get_rw_landing_probs,
    'heat_kernel_se': get_heat_kernels_diag
}