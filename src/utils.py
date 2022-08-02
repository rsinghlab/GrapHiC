# import torch
# import torch.nn as nn
# from torch.nn import Conv1d
# from torch_geometric.data import Dataset, Data
# from torch_geometric.loader import DataLoader

import math
import random
import os, shutil
import numpy as np
import statistics
# from scipy.stats import norm
# from scipy.sparse.csgraph import laplacian

# from matplotlib.colors import LinearSegmentedColormap
# from matplotlib import pyplot as plt
# from matplotlib import gridspec
# import matplotlib.pyplot as plt 

from src.matrix_operations import compactM, divide
from src.epigentic_encodings import read_node_encoding_files
from src.positional_encodings import encoding_methods






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
                     pos_indx=torch.from_numpy(pos_indx), y=torch.from_numpy(target),
                     input=torch.from_numpy(matrix)
                    )
        
        graphs.append(graph)
    
    
    return DataLoader(graphs, batch_size=batch_size, shuffle=shuffle)





















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





def graph_random_walks(adjacency_matrix):
    pass