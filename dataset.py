import pandas as pd
import torch
import torch_geometric
from torch_geometric.data import Dataset, Data
import numpy as np 
import os
from tqdm import tqdm
from src import globals



print(f"Torch version: {torch.__version__}")
print(f"Cuda available: {torch.cuda.is_available()}")
print(f"Torch geometric version: {torch_geometric.__version__}")


class HiCDataset(Dataset):
    def __init__(self, root, noise, cell_line, chr_id, required_datasets, percentile = 99.95, 
                 test=False, regenerate=False, transform=None, pre_transform=None,
                 verbose=True):
        '''
        @params: root <string>, root directory that contails all the dataset files
        @params: chr_id <int>, chromosome id, whose .npz files that need to be processed
        @params: percentile <float>, percentile value to use for the preprocessing phase
        @params: test <bool>, whether this is a test dataset or a training dataset 
        '''
        self.test = test
        self.required_datasets = required_datasets
        self.noise = noise
        self.cell_line = cell_line
        self.chr_id = chr_id
        self.percentile = percentile
        self.verbose = verbose
        self.regenerate = regenerate

        super(HiCDataset, self).__init__(root, transform, pre_transform)
        
    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)  
        """
        dataset_ids = globals.DATASETS[self.required_datasets][self.cell_line]
        
        files_required = []

        for dataset_id  in dataset_ids:
            file_name = '{}_{}_{}_chr{}.npz'.format(
                self.noise,
                self.cell_line,
                dataset_id,
                self.chr_id
            )
            files_required.append(file_name)

        return files_required

    @property
    def processed_file_names(self):
        
        if self.regenerate:
            if self.verbose: print('Reprocessing the files')
            return ['regenerate']
        
        dataset_ids = globals.DATASETS[self.required_datasets][self.cell_line]
        
        files_required = []

        for dataset_id  in dataset_ids:

            for n_submat in range(0, int((globals.CHROMOSOME_SIZES[str(self.chr_id)]/globals.RESOLUTION)), globals.SUB_MATRIX_SIZE):
                file_name = 'n-{}_c-{}_d-{}_chr-{}_s-{}.pt'.format(
                    self.noise,
                    self.cell_line,
                    dataset_id,
                    self.chr_id,
                    n_submat
                )
                files_required.append(file_name)
        
        print(files_required)

        return files_required

    def download(self):
        pass

    def process(self):
        for file_path in self.raw_paths:
            file_name = file_path.split('/')[-1].split('.')[0]

            raw_data = self.read_npz_file(file_path)
            
            cutoff = self.get_cutoff_value(file_name)

            if self.verbose: print('Working on file {}, that has shape {} and cutoff value {}'.format(file_path, raw_data.shape, cutoff))

            raw_data = self.normalize(raw_data, cutoff)
            
            file_name = file_name.split('_')

            for n_submat in range(0, raw_data.shape[0], globals.SUB_MATRIX_SIZE,):
                print(n_submat, n_submat+globals.SUB_MATRIX_SIZE)
                
                # Out of bounds check 
                if n_submat + globals.SUB_MATRIX_SIZE >= raw_data.shape[0]:
                    submat = np.zeros((globals.SUB_MATRIX_SIZE, globals.SUB_MATRIX_SIZE))
                    submat[0:(raw_data.shape[0] - n_submat), 0:(raw_data.shape[0] - n_submat)] = raw_data[n_submat: raw_data.shape[0], n_submat: raw_data.shape[0]]
                    node_feature_mask = np.zeros((globals.SUB_MATRIX_SIZE))
                    node_feature_mask[0:(raw_data.shape[0] - n_submat)] = 1
                    
                else:
                    submat = raw_data[n_submat: n_submat+globals.SUB_MATRIX_SIZE, n_submat: n_submat+globals.SUB_MATRIX_SIZE]
                    node_feature_mask = np.ones((globals.SUB_MATRIX_SIZE))

                node_features = self._get_node_features(submat, node_feature_mask)
                edge_features = self._get_edge_features(submat)
                adjacency_matrix = self._get_adjacency_info(submat)

                data = Data(x=node_features, 
                            edge_index=adjacency_matrix,
                            edge_attr=edge_features,
                            path=file_path) 
                

                save_file = 'n-{}_c-{}_d-{}_chr-{}_s-{}.pt'.format(
                    self.noise,
                    self.cell_line,
                    file_name[-2],
                    self.chr_id,
                    n_submat
                )

                torch.save(data, 
                        os.path.join(self.processed_dir, 
                                    save_file))

            

    # Preprocessing utilities
    def read_npz_file(self, file):
        '''
            @params: file <string>, path to the file that we need to read
            @returns: <np.array> HiC data in a dense matrix format        
        '''
        return np.load(file)['hic']

    def get_cutoff_value(self, file_name):
        dataset_id = file_name.split('_')[2]
        file_data = list(map(lambda x: x.split(','),open(globals.PATH_TO_DATASET_STATISTICS_FILE).read().split('\n')))[:-1]
        cutoff = list(filter(lambda x: dataset_id in x, file_data))[0][2]
        
        return float(cutoff)


    def normalize(self, data, cutoff):
            data_norm = np.minimum(cutoff, data)
            data_norm = np.maximum(data_norm, 0)

            data_norm = data_norm / cutoff

            return data_norm

   

    def _get_node_features(self, matrix, node_feature_mask):
        """ 
        This will return a matrix / 2d array of the shape
        [Number of Nodes, Node Feature size]
        """
        # Since for now we do not have node feature information, I am going to return a constant value for all node features 
        node_features = np.asarray(np.ones((matrix.shape[0], 1))) * (node_feature_mask.reshape(-1, 1))

        return torch.tensor(node_features, dtype=torch.float)



    def _get_edge_features(self, matrix):
        """ 
        This will return a matrix / 2d array of the shape
        [Number of edges, Edge Feature size]
        """
        edge_features = []
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if i > j: 
                    continue
                
                edge_features += [[matrix[i][j]], [matrix[i][j]]]
        
        edge_features = np.asarray(edge_features)
        return torch.tensor(edge_features, dtype=torch.long)

        


    def _get_adjacency_info(self, matrix):
        """
            We could also use rdmolops.GetAdjacencyMatrix(mol)
            but we want to be sure that the order of the indices
            matches the order of the edge features
        """
        edge_indices = []

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if i > j: 
                    continue
                
                edge_indices += [[i, j], [j, i]]
        
        edge_indices = np.asarray(edge_indices)
        edge_indices = torch.tensor(edge_indices, dtype=torch.long)
        edge_indices = edge_indices.t().to(torch.long).view(2, -1)
        
        return edge_indices
        

    def _get_labels(self, label):
        pass

    def len(self):
        return len(os.listdir(self.processed_dir))


    def get(self, dataset_id, submat_num):
        files = list(map(lambda x: os.path.join(self.processed_dir, x),os.listdir(self.processed_dir)))
        
        required_file_path = list(filter(lambda x: dataset_id in x, files))
        
        dataset_file_path = ''
        for rfp in required_file_path:
            rfp_submat_num = rfp.split('/')[-1].split('.')[0].split('_')[-1].split('-')[-1]
            if rfp_submat_num == str(submat_num):
                dataset_file_path = rfp

        if dataset_file_path == '':
            print('Trying to access file that doesnt exist')
            exit(1)

        data = torch.load(dataset_file_path)

        return data































