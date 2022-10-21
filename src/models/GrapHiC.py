import torch
import torch.nn.functional as F 
import torch.nn as nn
from torchvision.models.vgg import vgg16

from torch.nn import Linear, BatchNorm1d, ModuleList
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.nn import GCNConv, TransformerConv, GraphConv, GATConv
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader
from operator import itemgetter
import numpy as np
import os
from src.matrix_operations import create_graph_dataloader
from src.utils import WEIGHTS_DIRECTORY
from src.models.ContactCNN import ContactCNN
from src.models.TVLoss import TVLoss
from src.models.Insulation import InsulationLoss

torch.manual_seed(0)

class SpectralLoss(nn.Module):
    def __init__(self, n=10):
        super(SpectralLoss, self).__init__()
        self.eigen_values = torch.linalg.eigvals
        self.n = n

    def forward(self, outputs, targets):
        outputs_eigen_values = self.eigen_values(outputs) 
        targets_eigen_values = self.eigen_values(targets) 
        
        return torch.dist(outputs_eigen_values[:self.n], targets_eigen_values[:self.n])


class GCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=False, add_self_loops=False, act=torch.relu):
        super(GCNBlock, self).__init__()
        self.conv = GCNConv(in_channels, out_channels, 
                            normalize=normalize, 
                            add_self_loops=add_self_loops
                        )

        self.bn= GraphNorm(out_channels)
        self.act = act

    def forward(self, x, edge_index, edge_attr, batch_index):
        x = self.conv(x, edge_index, edge_attr)
        x = self.act(x)
        x = self.bn(x, batch_index)
        return x

class GraphConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act=torch.relu):
        super(GraphConvBlock, self).__init__()
        self.conv = GraphConv(in_channels, out_channels)
        self.bn= GraphNorm(out_channels)
        self.act = act

    def forward(self, x, edge_index, edge_attr, batch_index):
        x = self.conv(x, edge_index, edge_attr)
        x = self.act(x)
        x = self.bn(x, batch_index)
        return x

class GATConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, heads=4, edge_dim=1, act=torch.relu):
        super(GATConvBlock, self).__init__()
        self.conv = GATConv(
                                    in_channels, 
                                    out_channels, 
                                    heads=heads, 
                                    edge_dim=edge_dim,
                                    concat=True
                    ) 
        
        self.bn= GraphNorm(out_channels)
        self.act = act

    def forward(self, x, edge_index, edge_attr, batch_index):
        x = self.conv(x, edge_index, edge_attr)
        x = self.bn(x, batch_index)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, in_channels, out_channels, heads=4, edge_dim=1, act=torch.relu):
        super(TransformerBlock, self).__init__()
        # Transformation layer
        self.conv = TransformerConv(
                                    in_channels, 
                                    out_channels, 
                                    heads=heads, 
                                    edge_dim=edge_dim,
                                    beta=True,
                                    dropout=0.3
                                ) 

        self.linear = Linear(
            out_channels*heads, 
            out_channels
        )
        self.bn= GraphNorm(out_channels)
        self.act = act

    def forward(self, x, edge_index, edge_attr, batch_index):
        x = self.conv(x, edge_index, edge_attr)
        x = self.act(self.linear(x))
        x = self.bn(x, batch_index)
        return x





class GrapHiCLoss(nn.Module):
    def __init__(self, device):
        super(GrapHiCLoss, self).__init__()
        self.mse_loss = nn.MSELoss().to(device)

    def forward(self, out_images, target_images):
        image_loss = self.mse_loss(out_images, target_images)
        
        return image_loss 





class InnerProductDecoder(torch.nn.Module):
    def __init__(self, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.act = act

    def forward(self, z_0, z_1):
        z_1 = torch.transpose(z_1, 1, 2)
        adj = self.act(torch.matmul(z_0, z_1))
        adj = adj.reshape(adj.shape[0], 1, adj.shape[1], adj.shape[2])
        return adj



class GrapHiC(torch.nn.Module):
    def __init__(self, HYPERPARAMETERS,  device, model_name, input_embedding_size=4, base_dir=WEIGHTS_DIRECTORY):
        super(GrapHiC, self).__init__()
        self.hyperparameters = HYPERPARAMETERS
        self.device = device
        self.model_name = model_name
        self.weights_dir = os.path.join(base_dir, model_name)
        self.num_transform_blocks = HYPERPARAMETERS['transformblocks']
        
        if not os.path.exists(self.weights_dir):
            os.mkdir(self.weights_dir)
        
        self.loss_function = GrapHiCLoss(device)
        
        self.input_block = TransformerBlock(
            input_embedding_size,
            self.hyperparameters['embedding_size']
        )
        self.transform_blocks = [
            TransformerBlock(
                self.hyperparameters['embedding_size'], self.hyperparameters['embedding_size']
            ) 
            for _ in range(self.num_transform_blocks - 1)
        ]


        # Pytorch for some reasons dont move the class to the required device
        for transform_block in self.transform_blocks:
            transform_block.to(device)


        # Decoder
        self.contact_cnn = ContactCNN(
            self.hyperparameters['embedding_size'], 
            self.hyperparameters['embedding_size'],
            residual_blocks=HYPERPARAMETERS['resblocks']
        )


    def encode(self, x, edge_index, edge_attr, batch_index):
        x = self.input_block(x, edge_index, edge_attr, batch_index)
        for i in range(len(self.transform_blocks) - 1):
            x = self.transform_blocks[i](x, edge_index, edge_attr, batch_index)
        Z = self.process_graph_batch(x, batch_index)
        
        return Z

    def decode(self, Z):
        Z = self.contact_cnn(Z, Z)
        return Z


    def forward(self, data):
        Z = self.encode(data.x, data.edge_index, data.edge_attr, data.batch)
        Z = self.decode(Z)
        return Z


    def load_data(self, file_path, batch_size=-1, shuffle=False):
        data = np.load(file_path, allow_pickle=True)
        bases = torch.tensor(data['data'], dtype=torch.float32)
        targets = torch.tensor(data['target'], dtype=torch.float32)
        encodings = torch.tensor(data['encodings'], dtype=torch.float32)
        indxs = torch.tensor(data['inds'], dtype=torch.long)
        batch_size = self.hyperparameters['batch_size'] if batch_size == -1 else batch_size
        data_loader = create_graph_dataloader(bases, targets, encodings, indxs, batch_size , shuffle)
        
        return data_loader



    def create_optimizer(self):
        if self.hyperparameters['optimizer_type'] == 'ADAM':
            self.optimizer = optim.Adam(
                self.parameters(), 
                lr=self.hyperparameters['learning_rate']
            )
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

        


























