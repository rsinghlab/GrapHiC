
import torch
import torch.nn as nn
from torch.nn import Conv2d
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from operator import itemgetter

import numpy as np
import os

torch.manual_seed(0)

class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.conv(x))


class HiCNN2(nn.Module):
    def __init__(self, HYPERPARAMETERS, device, dir_model ='weights/hicnn2/'):
        super(HiCNN2, self).__init__()
        # Initializing variables
        self.hyperparameters = HYPERPARAMETERS
        self.device = device
        self.dir_model = dir_model
        if not os.path.exists(self.dir_model):
            os.mkdir(self.dir_model)
        self.loss_function = nn.MSELoss()


        # Initializing layers
        self.net1_conv1 = nn.Conv2d(1, 64, 13)
        self.net1_conv2 = nn.Conv2d(64, 64, 1)
        self.net1_conv3 = nn.Conv2d(64, 128, 3, padding=1, bias=False)
        self.net1_conv4R = nn.Conv2d(128, 128, 3, padding=1, bias=False)
        self.net1_conv5 = nn.Conv2d(128*25, 1000, 1, padding=0, bias=True)
        self.net1_conv6 = nn.Conv2d(1000,64,1,padding=0, bias=True)
        self.net1_conv7 = nn.Conv2d(64,1,3,padding=1, bias=False)

        self.net2_conv1 = nn.Conv2d(1, 8, 13)
        self.net2_conv2 = nn.Conv2d(8, 1, 1)
        self.residual_layer_vdsr = self.make_layer(Conv_ReLU_Block, 18)
        self.input_vdsr = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.output_vdsr = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)

        self.net3_conv1 = nn.Conv2d(1, 8, 9)
        self.net3_conv2 = nn.Conv2d(8, 8, 1)
        self.net3_conv3 = nn.Conv2d(8, 1, 5) 

        self.relu = nn.ReLU(inplace=True)

        self.weights = nn.Parameter((torch.ones(1, 3)/3), requires_grad=True)
        # He initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers) 
    
    def forward(self, input):
        '''
            Forward pass of the model

            @params: x <torch.tensor> input to the model with shape (batch_size, 1, 40 ,40)
            @returns <torch.tensor> out of the model with shape (batch_size, 1, 28, 28)
        '''
        # ConvNet1
        x = self.relu(self.net1_conv1(input))
        x = self.relu(self.net1_conv2(x))
        residual = x
        x2 = self.net1_conv3(x)
        output1 = x2
        outtmp = []
        for i in range(25):
          output1 = self.net1_conv4R(self.relu(self.net1_conv4R(self.relu(output1))))
          output1 = torch.add(output1, x2)
          outtmp.append(output1)
        output1 = torch.cat(outtmp, 1)
        output1 = self.net1_conv5(output1) 
        output1 = self.net1_conv6(output1)
        output1 = torch.add(output1, residual)
        output1 = self.net1_conv7(output1)

        # ConvNet2
        x_vdsr = self.relu(self.net2_conv1(input))
        x_vdsr = self.relu(self.net2_conv2(x_vdsr))
        residual2 = x_vdsr
        output2 = self.relu(self.input_vdsr(x_vdsr))
        output2 = self.residual_layer_vdsr(output2)
        output2 = self.output_vdsr(output2)
        output2 = torch.add(output2, residual2)

        # ConvNet3
        output3 = self.net3_conv1(input)
        output3 = F.relu(output3)
        output3 = self.net3_conv2(output3)
        output3 = F.relu(output3)
        output3 = self.net3_conv3(output3)
        output3 = F.relu(output3) 

        # w1*output1 + w2*output2 + w3*output3
        w_sum = self.weights.sum(1)
        output = (output1*self.weights.data[0][0]/w_sum) + (output2*self.weights.data[0][1]/w_sum) + (output3*self.weights.data[0][2]/w_sum)
        #output = (output2*self.weights.data[0][0]/w_sum) + (output3*self.weights.data[0][1]/w_sum)

        return output    
    
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
    

    def loss(self, preds, target):
        return self.loss_function(preds.float(), target.float())
  

    def load_weights(self, scheme='min-valid-loss'):
        if scheme not in ['min-valid-loss', 'last-epoch']:
            print('Weight loading scheme not supported!')
            exit(1)
        
        weights = list(map(lambda x: (float(x.split('_')[1].split('-')[0]) ,os.path.join(self.dir_model, x)), os.listdir(self.dir_model)))
        req_weights = max(weights,key=itemgetter(0))[1]
        
        self.load_state_dict(torch.load(req_weights, map_location=self.device))

