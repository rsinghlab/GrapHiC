import torch
import torch.nn as nn
from torchvision.transforms import GaussianBlur
import numpy as np
from operator import itemgetter

torch.manual_seed(0)






class Smoothing(nn.Module):
    def __init__(self, HYPERPARAMETERS, device, filter_size=(17,17), sigma=(7,7)):
        super(Smoothing, self).__init__()
        self.hyperparameters = HYPERPARAMETERS
        self.device = device
        self.smooth = GaussianBlur(kernel_size=filter_size, sigma=sigma)

        

    def forward(self, x):
        '''
            Forward pass of the model

            @params: x <torch.tensor> input to the model with shape (batch_size, 1, 40 ,40)
            @returns <torch.tensor> out of the model with shape (batch_size, 1, 28, 28)
        '''
        return self.smooth(x)
    
    
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

        # Create the dataloader object
        data_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(base, target, indxs), 
            batch_size=self.hyperparameters['batch_size'], shuffle=True
        )

        return data_loader
    
    def load_weights(self, scheme='min-valid-loss'):
        pass