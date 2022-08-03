import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.nn import Conv2d
from src.utils import WEIGHTS_DIRECTORY
from operator import itemgetter
from src.matrix_operations import create_graph_dataloader

torch.manual_seed(0)






class HiCPlus(nn.Module):
    def __init__(self, HYPERPARAMETERS, device, model_name, 
                input_size=40, base_dir=WEIGHTS_DIRECTORY):
        super(HiCPlus, self).__init__()
        # Initialize the control parameters
        self.model_type = 'Image'
        self.hyperparameters = HYPERPARAMETERS
        self.device = device
        self.model_name = model_name
        self.weights_dir = os.path.join(base_dir, model_name)

        # Initialize the Loss Function
        self.loss_function = nn.MSELoss()
        if not os.path.exists(self.weights_dir):
            os.mkdir(self.weights_dir)

        # Initialize layers
        self.conv1 = Conv2d(1, 8, 9)
        self.conv2 = Conv2d(8, 8, 1)
        self.conv3 = Conv2d(8, 1, 5)
        

        

    def forward(self, data):
        '''
            Forward pass of the model

            @params: x <torch.tensor> input to the model with shape (batch_size, 1, 40 ,40)
            @returns <torch.tensor> out of the model with shape (batch_size, 1, 28, 28)
        '''
        x = self.process_graph_batch(data.input, data.batch)
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return F.relu(self.conv3(x))
    
    
    def load_data(self, file_path):
        '''
            Loads the data file in a dataloader that is appropriate for the model
            @params file_path <string> path to the file that we need to load for the model
            @returns <torch.DataLoader> dataloader object 
        '''
        # Decompress the file
        data = np.load(file_path, allow_pickle=True)
        
        # Get the individual data objects
        bases = torch.tensor(data['data'], dtype=torch.float32)
        targets = torch.tensor(data['target'], dtype=torch.float32)
        encodings = torch.tensor(data['encodings'], dtype=torch.float32)
        indxs = torch.tensor(data['inds'], dtype=torch.long)

        # Targets need to be reshaped to compare the output
        targets_reshaped = np.zeros((targets.shape[0], 1, 28, 28))
        
        for i in range(targets.shape[0]):
            targets_reshaped[i] = targets[i, 0, 6:34, 6:34]
        
        targets_reshaped = torch.tensor(targets_reshaped, dtype=torch.float32)

        data_loader = create_graph_dataloader(bases, targets_reshaped, encodings, indxs, self.hyperparameters['batch_size'], False)

        return data_loader
    
    def create_optimizer(self):
        if self.hyperparameters['optimizer_type'] == 'ADAM':
            self.optimizer = optim.Adam(self.parameters(), lr=self.hyperparameters['learning_rate'])
            return self.optimizer
        else:
            print('Optimizer {} not currently support!'.format(self.hyperparameters['optimizer_type']))
            exit(1)
    
    
    def process_graph_batch(self, graph_batch, batch_idx):
        num_targets = int(batch_idx.max()) + 1
        return graph_batch.reshape(num_targets, int(graph_batch.shape[0]/num_targets), graph_batch.shape[1])

    
    def loss(self, preds, target):
        return self.loss_function(preds.float(), target.float())
        
    def load_weights(self, scheme='min-valid-loss'):
        if scheme not in ['min-valid-loss', 'last-epoch']:
            print('Weight loading scheme not supported!')
            exit(1)
        

        req_weights = list(filter(lambda x: '99-epoch' in x, os.listdir(self.weights_dir)))[0]
        req_weights = os.path.join(self.weights_dir, req_weights)
        # weights = list(map(lambda x: (float(x.split('_')[1].split('-')[0]) ,os.path.join(self.weights_dir, x)), os.listdir(self.weights_dir)))
        # req_weights = min(weights,key=itemgetter(0))[1]
        
        self.load_state_dict(torch.load(req_weights, map_location=self.device))
