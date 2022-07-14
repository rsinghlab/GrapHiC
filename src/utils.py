from concurrent.futures import process
from email.mime import base
import torch
import torch.nn as nn
from torch.nn import Conv1d
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader

import math
import random
import os, shutil
import numpy as np
import statistics
from scipy.stats import norm
from scipy.sparse.csgraph import laplacian
from numpy.linalg import eig

from matplotlib.colors import LinearSegmentedColormap
from matplotlib import pyplot as plt
from matplotlib import gridspec
import matplotlib.pyplot as plt 
from positional_encodings.torch_encodings import PositionalEncoding1D



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


def transformer_positional_encoding(matrix, encoding_dim=4):
    '''
        A positional encoding method that assigns transformer sine-cosine node position
        @params: matrix <np.array> 2D Hi-C matrix array
        @params: type <string>, fractional defines the max value to be 1 and integer defines the max 
                                value to be the matrix.shape[0].
        @params: <np.array> 1D that contains the node positions
    '''
    p = PositionalEncoding1D(matrix.shape[0])
    place_holder = torch.zeros(1, matrix.shape[0], encoding_dim)
    encoding = p(place_holder).detach().numpy()[0]

    return encoding

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


dataset_partitions = {
    'train': [1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15, 16, 17, 18],
    'valid': [8, 9, 10, 11],
    'test': [19, 20, 21, 22],
    'all': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
}



def plot_distribution_with_precentiles(values, graph_name):
    '''
        This function draws out a plot comparing the distribution of contact counts
        with respect to the percentiles
        @params: values <np.array> values or input distribution
        @returns None
    '''
    x_vals = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 99.5, 99.9, 99.95, 99.99]
    y_vals = list(map(lambda x: np.percentile(values, x), x_vals))

    x_vals = list(map(lambda x: str(x), x_vals))
    
    plt.xticks(rotation = 45) 
    plt.plot(x_vals, y_vals)
    plt.savefig('outputs/graphs/{}'.format(graph_name), format='png')
    plt.close()


def load_hic_file(path, format='.npz'):
    '''
        A wrapper that reads the HiC file and returns it, it essentially is created to 
        later support multiple HiC file formats, currently the only utilized file format is 
        .npz. 

        @params: 
    '''
    if format == '.npz':
        return np.load(path, allow_pickle=True)
    else:
        print('Provided file format is invalid')
        exit(1)


def normalize_hic_matrix(hic_matrix, params, cell_line='H1', chromosome='chr1'):
    '''
        This fuction performs chromosome wide normalization of the HiC matrices 
        @params: hic_matrix <np.array>, 2D array that contains all the intra-chromosomal contacts
        @params: params <dict>, A dictionary that contains all the required parameters to perform the normalization
        @returns: <np.array> A normalized HiC matrix
    '''
    # Do not perform any normalization (Not Recommended)
    if not params['norm']:
        return hic_matrix

    # Set diagonal zero 
    if params['set_diagonal_zero']:
        np.fill_diagonal(hic_matrix, 0)
    

    if params['cutoff'] == -1:
        return hic_matrix


    # Get the value distribution in a flattened matrix
    all_values = hic_matrix.flatten()
    
    # Remove zeros
    if params['remove_zeros']:
        all_values = all_values[all_values>0]
        


    # Draw distribution graphs for visualizations
    if params['draw_dist_graphs']:
        name_of_graph = 'c-{}:{}_sdz-{}_rz-{}_precentiles-vs-contacts.png'.format(
            cell_line, chromosome, params['set_diagonal_zero'], params['remove_zeros']
        )
        plot_distribution_with_precentiles(all_values, name_of_graph)
        

    # Compute and apply cutoff
    cutoff_value = np.percentile(all_values, params['cutoff'])

    hic_matrix = np.minimum(cutoff_value, hic_matrix)
    hic_matrix = np.maximum(hic_matrix, 0)

    # Rescale
    if params['rescale']:
        hic_matrix = hic_matrix / (np.max(cutoff_value) + 1)

    return hic_matrix



def process_compact_idxs(base_compact_idx, target_compact_idx, compact_method='intersection'):
    '''
        Processes the compact indexes based on the method type provided.
        @params: base_compact_idx <List>, Compaction indices of the base matrix
        @params: target_compact_idx <List>, Compaction indices of the target matrix
        @params: compact_method <string>, Compaction strategy, intersection is the default strategy  
        @returns: <List> Compact indices
    '''
    if compact_method == 'ignore':
        compact_indexes = []
            
    elif compact_method == 'intersection':
        compact_indexes = list(set.intersection(set(base_compact_idx), set(target_compact_idx)))
       
    elif compact_method == 'target':
        compact_indexes = target_compact_idx
        
    else:
        print("Invalid value for variable compact_type, please choose one of [ignore, intersection, target].")
        exit(1)

    return compact_indexes


def read_node_encoding_files(node_encoding_files, compact_idx=[]):

    pass



def create_dataset_from_hic_files(
                                path_to_base_files,
                                path_to_target_files, 
                                path_to_output_folder,
                                positional_encoding_method, 
                                node_encoding_files,
                                cropping_params={
                                'chunk_size':200,
                                'stride'    :200,
                                'bounds'    :190,
                                'padding'   :True
                                },
                                normalization_params={
                                'norm'              : False,
                                'remove_zeros'      : True,
                                'set_diagonal_zero' : False,
                                'cutoff'            : 99.0,
                                'rescale'           : False,
                                'chrom_wide'        : True, 
                                'draw_dist_graphs'  : False
                                },
                                noise='none',
                                compact='intersection',
                                datasets=[
                                'train', 
                                'valid', 
                                'test'
                                ],
                                verbose=True
    ):
    '''
        This function takes in path to the base chromosome files and target chromosome files 
        and pre-processes them and creates a dataset that can be used to train, validate and test 
        the HiC upscaling methods
        @params: path_to_base_file <string>, Absolute path to the base files
        @params: path_to_target_files <string>, Absolute path to the target chromosome files
        @params: path_to_output_folder <string>, Absolute path where to store the generated dataset file
        @params: positional_encoding_method <string>, what positional encoding method to use
        @params: node_encoding_files <string> path to the node encoding files, these files contain the epigenetic signals
        @params: cropping_params <dict> contains the cropping parameters necessary for cropping huge chromosome files
        @params: normalization_params <dict> normalization parameters defining how to normalize the Hi-C matrix
        @params: compact <string>, what matrix compaction method to use to remove rows (and columns) that are non informative. Defaults 'intersection'. 
        @params: dataset <list>, what dataset to construct 
        @returns: None
    '''

    create_entire_path_directory(path_to_output_folder)

    for dataset in datasets:
        if verbose: print('Creating {} dataset'.format(dataset))

        chromosomes = dataset_partitions[dataset]
        output_file = os.path.join(path_to_output_folder, '{}.npz'.format(dataset))
        
        results = []

        for chromosome in chromosomes:
            if verbose: print('Working with chromosome {} files...'.format(chromosome))

            base_chrom_path = os.path.join(path_to_base_files, 'chr{}.npz'.format(chromosome))
            target_chrom_path = os.path.join(path_to_target_files, 'chr{}.npz'.format(chromosome))
            base_data = load_hic_file(base_chrom_path)
            target_data = load_hic_file(target_chrom_path)

            base_chrom = base_data['hic']
            target_chrom = target_data['hic']
            
            if base_chrom.shape[0] == target_chrom.shape[0]:
                full_size = base_chrom.shape[0]
            else:
                print("Shapes of the datasets is not same, {} vs {}".format(base_data.shape, target_data.shape ))
                exit(1)
            
            # Find the non-informative rows and remove them from the Hi-C matrices
            if verbose: print('Removing non-infomative rows and columns')
            base_compact_idx = base_data['compact']
            target_compact_idx = target_data['compact'] 
            
            if len(base_compact_idx) == 0 or len(target_compact_idx) == 0:
                print('Chrom {} data not valid'.format(chromosome))
                continue
            

            compact_indexes = process_compact_idxs(base_compact_idx, target_compact_idx, compact_method=compact)
            
            base_chrom = compactM(base_chrom, compact_indexes)
            target_chrom = compactM(target_chrom, compact_indexes)
            
            
            if verbose: print('Processing Node Features')
            # Read Node feature files and compile them in a single array
            node_features = read_node_encoding_files(node_encoding_files, compact_indexes)
            
            
            # Divide the chromosome matrix
            if normalization_params['chrom_wide']:
                if verbose: print('Performing Chromosome wide normalization...')
            
                base_chrom = normalize_hic_matrix(base_chrom, normalization_params, chromosome='chr{}'.format(chromosome))
                target_chrom = normalize_hic_matrix(target_chrom, normalization_params, chromosome='chr{}'.format(chromosome))
            
            if verbose: print('Dividing the Chromosomes...')
            
            targets, inds = divide(target_chrom, chromosome, cropping_params)
            bases, _ = divide(base_chrom, chromosome, cropping_params)
            
            if verbose: print('Computing Positional Encodings')
            # Pos encodings needs to be computed once, they are the same for the base and targets (we dont really need those for the targets)
            pos_encodings = []
            for base in bases:
                pos_encoding = encoding_methods[positional_encoding_method](base[0])
                pos_encodings.append(pos_encoding)

            # Append the node encodings with positional encodings (TODO)
            # TODO
            encodings = pos_encodings
            
            # Store all the results for computations to be perfomed after all chromosomes
            results.append((chromosome, bases, targets, inds, encodings, compact_indexes, full_size))


        data = np.concatenate([r[1] for r in results])
        target = np.concatenate([r[2] for r in results])
        inds = np.concatenate([r[3] for r in results])
        encodings = np.concatenate([r[4] for r in results])
        
        compacts = {r[0]: r[5] for r in results}
        sizes = {r[0]: r[6] for r in results}

        
        print('Saving file:', output_file)
        np.savez_compressed(output_file, data=data, target=target, inds=inds, encodings=encodings, compacts=compacts, sizes=sizes)







def delete_files(folder_path):
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        try:
            shutil.rmtree(filepath)
        except OSError:
            os.remove(filepath)

def get_node_features(encoding):
    return torch.tensor(encoding, dtype=torch.float)

def get_edge_attrs(matrix):
    edge_indices = np.nonzero(matrix)    
    edge_features = matrix[edge_indices]
    edge_features = edge_features.reshape(-1, 1)

    return torch.tensor(edge_features, dtype=torch.float)

def get_edge_indexes(matrix):
    edge_indices = np.transpose(np.nonzero(matrix))
    edge_indices = torch.tensor(edge_indices, dtype=torch.long)
    edge_indices = edge_indices.t().to(torch.long).view(2, -1)

    return edge_indices


def create_graph_dataloader(base, targets, encodings, pos_indxs, batch_size, shuffle):
    graphs = []
    for idx in range(base.shape[0]):
        matrix = base[idx, 0, :, :].cpu().detach().numpy()
        target = targets[idx, 0, :, :].cpu().detach().numpy()
        encoding = encodings[idx, :, :].cpu().detach().numpy()
        pos_indx = pos_indxs[idx].cpu().detach().numpy()
        
        x = get_node_features(encoding)
        
        edge_attrs = get_edge_attrs(matrix)
        
        edge_indexes = get_edge_indexes(matrix)
        
        graph = Data(x=x, edge_attr=edge_attrs, edge_index=edge_indexes, 
                     pos_indx=torch.from_numpy(pos_indx), y=torch.from_numpy(target)
                    )
        
        graphs.append(graph)
    
    
    return DataLoader(graphs, batch_size=batch_size, shuffle=shuffle)


















except_chr = {'hsa': {'X': 23, 23: 'X'}, 'mouse': {'X': 20, 20: 'X'}}
# Matrix manips
def spreadM(c_mat, compact_idx, full_size, convert_int=True, verbose=False):
    """spreading matrix according to the index list (a reversed operation to compactM)."""
    result = np.zeros((full_size, full_size)).astype(c_mat.dtype)
    if convert_int: result = result.astype(np.int)
    if verbose: print('Spreading a', c_mat.shape, 'shaped matrix to', result.shape, 'shaped!' )
    for i, s_idx in enumerate(compact_idx):
        result[s_idx, compact_idx] = c_mat[i]
    return result

def compactM(matrix, compact_idx, verbose=False):
    """
        Compacting matrix according to the index list.
        @params: matrix <np.array> Full sized matrix, that needs to be compressed
        @params: compact_idx <list> Indexes of rows that contain data
        @params: verbose <boolean> Debugging print statements
        @returns: <np.array> Condesed matrix with zero arrays pruned 
    """
    compact_size = len(compact_idx)
    result = np.zeros((compact_size, compact_size)).astype(matrix.dtype)
    
    if verbose: print('Compacting a', matrix.shape, 'shaped matrix to', result.shape, 'shaped!')
    
    for i, idx in enumerate(compact_idx):
        result[i, :] = matrix[idx][compact_idx]
    
    return result

def together(matlist, indices, corp=0, species='hsa', tag='HiC'):
    chr_nums = sorted(list(np.unique(indices[:,0])))
    # convert last element to str 'X'
    if chr_nums[-1] in except_chr[species]: chr_nums[-1] = except_chr[species][chr_nums[-1]]
    print(f'{tag} data contain {chr_nums} chromosomes')
    _, h, w = matlist[0].shape
    results = dict.fromkeys(chr_nums)
    for n in chr_nums:
        # convert str 'X' to 23
        num = except_chr[species][n] if isinstance(n, str) else n
        loci = np.where(indices[:,0] == num)[0]
        sub_mats = matlist[loci]
        index = indices[loci]
        width = index[0,1]
        full_mat = np.zeros((width, width))
        for sub, pos in zip(sub_mats, index):
            i, j = pos[-2], pos[-1]
            if corp > 0:
                sub = sub[:, corp:-corp, corp:-corp]
                _, h, w = sub.shape
            full_mat[i:i+h, j:j+w] = sub
        results[n] = full_mat
    return results

def divide(mat, chr_num, cropping_params, verbose=False):
    """
        @params: mat <np.array> HiC matrix that needs to be chunked up
        @params: chr_num <string> Chromosome number of the input HiC matrix
        @params: cropping_params <dict> contains the required parameters to crop the matrix
        @params: verbose <boolean> Debugging print statements
        @returns: list<np.array>, list first return is chunked up matrices in a list and second 
                return contains the positions 
    """
    chr_str = str(chr_num)
    result = []
    index = []
    size = mat.shape[0]
    
    stride = cropping_params['stride']
    chunk_size = cropping_params['chunk_size']
    bound = cropping_params['bounds']
    padding = cropping_params['padding']

    if (stride < chunk_size and padding):
        pad_len = (chunk_size - stride) // 2
        mat = np.pad(mat, ((pad_len,pad_len), (pad_len,pad_len)), 'constant')
    # mat's shape changed, update!
    height, width = mat.shape
    assert height == width, 'Now, we just assumed matrix is squared!'
    
    for i in range(0, height, stride):
        for j in range(0, width, stride):
            if abs(i-j)<=bound and i+chunk_size<height and j+chunk_size<width:
                subImage = mat[i:i+chunk_size, j:j+chunk_size]
                result.append([subImage])
                index.append((chr_num, size, i, j))
    result = np.array(result)
    if verbose: print(f'[Chr{chr_str}] Dividing HiC matrix ({size}x{size}) into {len(result)} samples with chunk={chunk_size}, stride={stride}, bound={bound}')
    index = np.array(index)
    return result, index



def create_entire_path_directory(path):
    """
        Given a path, creates all the missing directories on the path and 
        ensures that the given path is always a valid path
        @params: <string> path, full absolute path, this function can act wonkily if given a relative path
        @returns: None
    """
    
    path = path.split('/')
    curr_path = '/'
    for dir in path:
        curr_path = os.path.join(curr_path, dir)
        if os.path.exists(curr_path):
            continue
        else:
            os.mkdir(curr_path)



def visualize_matrix(input_path, output_path, idx, size):
    data = np.load(input_path, allow_pickle=True)['hic']
    print(np.max(data), np.mean(data))

    plt.matshow(data[idx:idx+size, idx:idx+size], cmap='hot', interpolation='none')
    plt.savefig(output_path)