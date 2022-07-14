import torch
import torch.nn as nn
from torch.nn import Conv2d
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
import os
from operator import itemgetter

torch.manual_seed(0)






class HiCNN(nn.Module):
    def __init__(self, HYPERPARAMETERS, device, dir_model ='weights/hicnn/'):
        super(HiCNN, self).__init__()
        # Initializing variables
        self.hyperparameters = HYPERPARAMETERS
        self.device = device
        self.dir_model = dir_model
        if not os.path.exists(self.dir_model):
            os.mkdir(self.dir_model)
        self.loss_function = nn.MSELoss()
        
        
        # Initializing layers
        self.conv1 = nn.Conv2d(1, 8, 13)
        self.conv2 = nn.Conv2d(8, 1, 1)
        self.conv3 = nn.Conv2d(1, 128, 3, padding=1, bias=False)
        self.conv4R = nn.Conv2d(128, 128, 3, padding=1, bias=False)
        self.conv5 = nn.Conv2d(128, 1, 3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        # He initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        

        

    def forward(self, x):
        '''
            Forward pass of the model

            @params: x <torch.tensor> input to the model with shape (batch_size, 1, 40 ,40)
            @returns <torch.tensor> out of the model with shape (batch_size, 1, 28, 28)
        '''
        x = x.float()
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        residual = x
        x2 = self.conv3(x)
        out = x2
        for _ in range(5):
            out = self.conv4R(self.relu(self.conv4R(self.relu(out))))
            out = torch.add(out, x2)

        out = self.conv5(self.relu(out))
        out = torch.add(out, residual)
        return out
    
    
    def load_data(self, file_path):
        '''
            Loads the data file in a dataloader that is appropriate for the model
            @params file_path <string> path to the file that we need to load for the model
            @returns <torch.DataLoader> dataloader object 
        '''
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
            torch.utils.data.TensorDataset(base, torch.from_numpy(target_reshaped), indxs), 
            batch_size=self.hyperparameters['batch_size'], shuffle=True
        )

        return data_loader
    
    def create_optimizer(self):
        if self.hyperparameters['optimizer_type'] == 'ADAM':
            self.optimizer = optim.Adam(self.parameters(), lr=self.hyperparameters['learning_rate'])
            return self.optimizer
        else:
            print('Optimizer {} not currently support!'.format(self.hyperparameters['optimizer_type']))
            exit(1)
    
    def load_weights(self, scheme='min-valid-loss'):
        if scheme not in ['min-valid-loss', 'last-epoch']:
            print('Weight loading scheme not supported!')
            exit(1)
        
        weights = list(map(lambda x: (float(x.split('_')[1].split('-')[0]) ,os.path.join(self.dir_model, x)), os.listdir(self.dir_model)))
        req_weights = max(weights,key=itemgetter(0))[1]
        
        self.load_state_dict(torch.load(req_weights, map_location=self.device))


    def loss(self, preds, target):
        return self.loss_function(preds.float(), target.float())
        