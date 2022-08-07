import torch
import torch.nn.functional as F 
import torch.nn as nn

from torch.nn import L1Loss
from torch_geometric.nn import GCNConv
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from operator import itemgetter
import numpy as np
import os
from src.matrix_operations import create_graph_dataloader
from src.utils import WEIGHTS_DIRECTORY
from src.models.ContactCNN import ContactCNN
from src.models.TVLoss import TVLoss

torch.manual_seed(0)


class GrapHiCLoss(torch.nn.Module):
    def __init__(self):
        super(GrapHiCLoss, self).__init__()
        self.mse_lambda = 1
        self.tv_lambda = 0.0001
        self.mse = L1Loss()
        self.tvloss = TVLoss(self.tv_lambda)

    def forward(self, output, target):
        l1_loss = self.mse_lambda*self.mse(output, target)
        tv_loss = self.tvloss(output)
        return l1_loss + tv_loss



class GrapHiC(torch.nn.Module):
    def __init__(self, HYPERPARAMETERS,  device, model_name, input_embedding_size=4, base_dir=WEIGHTS_DIRECTORY):
        super(GrapHiC, self).__init__()
        self.model_type = 'Graph'
        self.hyperparameters = HYPERPARAMETERS
        self.device = device
        self.model_name = model_name
        self.weights_dir = os.path.join(base_dir, model_name)

        if not os.path.exists(self.weights_dir):
            os.mkdir(self.weights_dir)
        
        self.loss_function = GrapHiCLoss()

        self.conv0 = GCNConv(input_embedding_size, 64, 1, normalize=True)
        self.conv1 = GCNConv(64, 64, 1, normalize=True)
        self.conv2 = GCNConv(64, 64, 1, normalize=True)

        self.contact_cnn = ContactCNN(64, 64)

        

    def encode(self, x, edge_index, edge_attr, batch_index):
        x = torch.relu(self.conv0(x=x, edge_index=edge_index, edge_weight=edge_attr))
        x = torch.relu(self.conv1(x=x, edge_index=edge_index, edge_weight=edge_attr))
        x = torch.relu(self.conv2(x=x, edge_index=edge_index, edge_weight=edge_attr))
        x = self.process_graph_batch(x, batch_index)
        return x


    def decode(self, Z):
        Z = self.contact_cnn(Z, Z)
        return Z        

    def forward(self, data):
        Z = self.encode(data.x, data.edge_index, data.edge_attr, data.batch)

        return self.decode(Z)


    def load_data(self, file_path):
        data = np.load(file_path, allow_pickle=True)
        bases = torch.tensor(data['data'], dtype=torch.float32)
        targets = torch.tensor(data['target'], dtype=torch.float32)
        encodings = torch.tensor(data['encodings'], dtype=torch.float32)
        indxs = torch.tensor(data['inds'], dtype=torch.long)

        data_loader = create_graph_dataloader(bases, targets, encodings, indxs, self.hyperparameters['batch_size'], False)
        
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
        

    def process_graph_batch(self, graph_batch, batch_idx):
        num_targets = int(batch_idx.max()) + 1
        return graph_batch.reshape(num_targets, int(graph_batch.shape[0]/num_targets), graph_batch.shape[1])


    def load_weights(self, scheme='min-valid-loss'):
        if scheme not in ['min-valid-loss', 'last-epoch']:
            print('Weight loading scheme not supported!')
            exit(1)
        
        weights = list(map(lambda x: (float(x.split('_')[1].split('-')[0]) ,os.path.join(self.weights_dir, x)), os.listdir(self.weights_dir)))
        req_weights = min(weights,key=itemgetter(0))[1]

        print("Loading: {}".format(req_weights))

        self.load_state_dict(torch.load(req_weights, map_location=self.device))

        


























