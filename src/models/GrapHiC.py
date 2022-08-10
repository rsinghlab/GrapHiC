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
    def __init__(self, variational):
        super(GrapHiCLoss, self).__init__()
        self.variational = variational
        self.mse_lambda = 1
        self.tv_lambda = 0.0001
        self.kld_lambda = 0.1
        self.mse = L1Loss()
        self.tvloss = TVLoss(self.tv_lambda)

    def kld(self, mu, logvar):
        return (-0.5 / mu.shape[1] * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1)))

    def forward(self, output, target, mu, logvar):
        l1_loss = self.mse_lambda*self.mse(output, target)
        tv_loss = self.tvloss(output)
        kld_loss = self.kld_lambda * self.kld(mu, logvar) if self.variational else 0

        return l1_loss + tv_loss + kld_loss



class GrapHiC(torch.nn.Module):
    def __init__(self, HYPERPARAMETERS,  device, model_name, input_embedding_size=4, base_dir=WEIGHTS_DIRECTORY):
        super(GrapHiC, self).__init__()
        self.model_type = 'Graph'
        self.hyperparameters = HYPERPARAMETERS
        self.device = device
        self.model_name = model_name
        self.weights_dir = os.path.join(base_dir, model_name)
        self.variational = False


        if not os.path.exists(self.weights_dir):
            os.mkdir(self.weights_dir)
        
        self.loss_function = GrapHiCLoss(self.variational)
        
        self.conv0 = GCNConv(input_embedding_size, 32, 1, normalize=True)
        self.conv1 = GCNConv(32, 32, 1, normalize=True)
        self.mu = GCNConv(32, 32, 1, normalize=True)
        self.logvar = GCNConv(32, 32, 1, normalize=True)


        self.contact_cnn = ContactCNN(32, 32)
        self.mu_keeper = None
        self.logvar_keeper = None
        self.training = True
        

    def encode(self, x, edge_index, edge_attr, batch_index):
        x = torch.relu(self.conv0(x=x, edge_index=edge_index, edge_weight=edge_attr))
        x = torch.relu(self.conv1(x=x, edge_index=edge_index, edge_weight=edge_attr))
        mu = torch.relu(self.mu(x=x, edge_index=edge_index, edge_weight=edge_attr))
        logvar = torch.relu(self.logvar(x=x, edge_index=edge_index, edge_weight=edge_attr))

        mu = self.process_graph_batch(mu, batch_index)
        logvar = self.process_graph_batch(logvar, batch_index)

        return mu, logvar


    def decode(self, Z):
        Z = self.contact_cnn(Z, Z)
        return Z        

    def reparametrize(self, mu, logvar):
        if self.training and self.variational:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu


    def forward(self, data):
        mu, logvar = self.encode(data.x, data.edge_index, data.edge_attr, data.batch)
        self.mu_keeper = mu
        self.logvar_keeper = logvar

        Z = self.reparametrize(mu, logvar)

        return self.decode(Z)


    def load_data(self, file_path, batch_size=-1, shuffle=False):
        data = np.load(file_path, allow_pickle=True)
        bases = torch.tensor(data['data'], dtype=torch.float32)
        targets = torch.tensor(data['target'], dtype=torch.float32)
        print(data['encodings'].shape)

        encodings = torch.tensor(data['encodings'], dtype=torch.float32)
        indxs = torch.tensor(data['inds'], dtype=torch.long)
        batch_size = self.hyperparameters['batch_size'] if batch_size == -1 else batch_size
        data_loader = create_graph_dataloader(bases, targets, encodings, indxs, batch_size , shuffle)
        
        return data_loader



    def create_optimizer(self):
        if self.hyperparameters['optimizer_type'] == 'ADAM':
            self.optimizer = optim.Adam(self.parameters(), lr=self.hyperparameters['learning_rate'])
            return self.optimizer
        else:
            print('Optimizer {} not currently support!'.format(self.hyperparameters['optimizer_type']))
            exit(1)

    def loss(self, preds, target):
        return self.loss_function(preds.float(), target.float(), self.mu_keeper, self.logvar_keeper)
        

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

        


























