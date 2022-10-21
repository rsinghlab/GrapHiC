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
        

        

    def forward(self, x):
        '''
            Forward pass of the model

            @params: x <torch.tensor> input to the model with shape (batch_size, 1, 40 ,40)
            @returns <torch.tensor> out of the model with shape (batch_size, 1, 28, 28)
        '''
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return F.relu(self.conv3(x))
    
    
    def load_data(self, file_path, batch_size=-1, shuffle=True):
        '''
            Loads the data file in a dataloader that is appropriate for the model
            @params file_path <string> path to the file that we need to load for the model
            @returns <torch.DataLoader> dataloader object 
        '''
        batch_size = self.hyperparameters['batch_size'] if batch_size == -1 else batch_size
        # Decompress the file
        data = np.load(file_path, allow_pickle=True)
        
        # Get the individual data objects
        base = torch.tensor(data['data'], dtype=torch.float32)
        target = torch.tensor(data['target'], dtype=torch.float32)
        indxs = torch.tensor(data['inds'], dtype=torch.long)

        # Targets need to be reshaped to compare the output
        target_reshaped = np.zeros((target.shape[0], 1, 28, 28))
        
        for i in range(target.shape[0]):
            target_reshaped[i] = target[i, 0, 6:34, 6:34]
        
        # Create the dataloader object
        data_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(base, torch.from_numpy(target_reshaped), indxs, target), 
            batch_size=batch_size, shuffle=shuffle
        )

        return data_loader
    
    def create_optimizer(self):
        if self.hyperparameters['optimizer_type'] == 'ADAM':
            self.optimizer = optim.Adam(self.parameters(), lr=self.hyperparameters['learning_rate'])
            return self.optimizer
        else:
            print('Optimizer {} not currently support!'.format(self.hyperparameters['optimizer_type']))
            exit(1)
    
    def load_weights(self, scheme='last-epoch'):
        if scheme not in ['min-valid-loss', 'last-epoch']:
            print('Weight loading scheme not supported!')
            exit(1)
        if scheme == 'min-valid-loss':
            weights = list(map(lambda x: (float(x.split('_')[1].split('-')[0]) ,os.path.join(self.weights_dir, x)), os.listdir(self.weights_dir)))
            req_weights = min(weights, key=itemgetter(0))[1]

            print("Loading: {}".format(req_weights))

            self.load_state_dict(torch.load(req_weights, map_location=self.device))
        
        if scheme == 'last-epoch':
            weights = list(map(lambda x: (float(x.split('_')[0].split('-')[0]) ,os.path.join(self.weights_dir, x)), os.listdir(self.weights_dir)))
            req_weights = max(weights, key=itemgetter(0))[1]
            print("Loading: {}".format(req_weights))

            self.load_state_dict(torch.load(req_weights, map_location=self.device))

    def loss(self, preds, target):
        return self.loss_function(preds.float(), target.float())
