import torch
import torch.nn.functional as F 
import torch.nn as nn

from torch.nn import ConvTranspose2d, L1Loss, Dropout2d
from torch_geometric.nn import GCNConv, Set2Set, global_mean_pool
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from operator import itemgetter
import numpy as np
import os
from src.utils import create_graph_dataloader


class FullyConnected(nn.Module):
    """
    Performs part 1 of Contact Prediction Module. Takes embeddings from Projection module and produces broadcast tensor.

    Input embeddings of dimension :math:`d` are combined into a :math:`2d` length MLP input :math:`z_{cat}`, where :math:`z_{cat} = [z_0 \\ominus z_1 | z_0 \\odot z_1]`

    :param embed_dim: Output dimension of `dscript.models.embedding <#module-dscript.models.embedding>`_ model :math:`d` [default: 100]
    :type embed_dim: int
    :param hidden_dim: Hidden dimension :math:`h` [default: 50]
    :type hidden_dim: int
    :param activation: Activation function for broadcast tensor [default: torch.nn.ReLU()]
    :type activation: torch.nn.Module
    """

    def __init__(self, embed_dim, hidden_dim, activation=nn.ReLU()):
        super(FullyConnected, self).__init__()

        self.D = embed_dim
        self.H = hidden_dim
        #self.conv = nn.Conv2d(2 * self.D, self.H, 1)
        self.conv = nn.Conv2d(self.D, self.H, 1)
        #self.batchnorm = nn.BatchNorm2d(self.H)
        self.activation = activation

    def forward(self, z0, z1):
        """
        :param z0: Projection module embedding :math:`(b \\times N \\times d)`
        :type z0: torch.Tensor
        :param z1: Projection module embedding :math:`(b \\times M \\times d)`
        :type z1: torch.Tensor
        :return: Predicted broadcast tensor :math:`(b \\times N \\times M \\times h)`
        :rtype: torch.Tensor
        """

        # z0 is (b,N,d), z1 is (b,M,d)
        z0 = z0.transpose(1, 2)
        z1 = z1.transpose(1, 2)
        # z0 is (b,d,N), z1 is (b,d,M)

        # Removing the z_diff because in our case the embeddings should be the same
        # z_dif = torch.abs(z0.unsqueeze(3) - z1.unsqueeze(2))
        # print(z_dif.shape)
        
        z_mul = z0.unsqueeze(3) * z1.unsqueeze(2)
        
        # z_cat = torch.cat([z_dif, z_mul], 1)
        # print(z_cat.shape)

        c = self.conv(z_mul)
        c = self.activation(c)
        #c = self.batchnorm(c)

        return c


class ContactCNN(nn.Module):
    """
    Residue Contact Prediction Module. Takes embeddings from Projection module and produces contact map, output of Contact module.

    :param embed_dim: Output dimension of `dscript.models.embedding <#module-dscript.models.embedding>`_ model :math:`d` [default: 100]
    :type embed_dim: int
    :param hidden_dim: Hidden dimension :math:`h` [default: 50]
    :type hidden_dim: int
    :param width: Width of convolutional filter :math:`2w+1` [default: 7]
    :type width: int
    :param activation: Activation function for final contact map [default: torch.nn.Sigmoid()]
    :type activation: torch.nn.Module
    """

    def __init__(
        self, embed_dim, hidden_dim=50, width=7, activation=nn.Sigmoid()
    ):
        super(ContactCNN, self).__init__()

        self.hidden = FullyConnected(embed_dim, hidden_dim)

        self.conv = nn.Conv2d(hidden_dim, 1, width, padding=width // 2)
        #self.batchnorm = nn.BatchNorm2d(1)
        self.activation = activation
        #self.clip()

    def clip(self):
        """
        Force the convolutional layer to be transpose invariant.

        :meta private:
        """
        w = self.conv.weight
        self.conv.weight.data[:] = 0.5 * (w + w.transpose(2, 3))

    def forward(self, z0, z1):
        """
        :param z0: Projection module embedding :math:`(b \\times N \\times d)`
        :type z0: torch.Tensor
        :param z1: Projection module embedding :math:`(b \\times M \\times d)`
        :type z1: torch.Tensor
        :return: Predicted contact map :math:`(b \\times N \\times M)`
        :rtype: torch.Tensor
        """
        # print('z0', z0)
        # print('z1', z1)

        C = self.cmap(z0, z1)
        return self.predict(C)

    def cmap(self, z0, z1):
        """
        Calls `dscript.models.contact.FullyConnected <#module-dscript.models.contact.FullyConnected>`_.

        :param z0: Projection module embedding :math:`(b \\times N \\times d)`
        :type z0: torch.Tensor
        :param z1: Projection module embedding :math:`(b \\times M \\times d)`
        :type z1: torch.Tensor
        :return: Predicted contact broadcast tensor :math:`(b \\times N \\times M \\times h)`
        :rtype: torch.Tensor
        """
        C = self.hidden(z0, z1)
        return C

    def predict(self, C):
        """
        Predict contact map from broadcast tensor.

        :param B: Predicted contact broadcast :math:`(b \\times N \\times M \\times h)`
        :type B: torch.Tensor
        :return: Predicted contact map :math:`(b \\times N \\times M)`
        :rtype: torch.Tensor
        """

        # S is (b,N,M)
        s = self.conv(C)
        #s = self.batchnorm(s)
        s = self.activation(s)
        return s




class GrapHiCLoss(torch.nn.Module):
    def __init__(self):
        super(GrapHiCLoss, self).__init__()
        self.mse_lambda = 1

    def forward(self, output, target):
        # Computing the insulation loss
        l1_loss = self.mse_lambda*self.mse(output, target)
        return l1_loss










class GraphConvGrapHiC(torch.nn.Module):
    def __init__(self, HYPERPARAMETERS,  device, model_name, weight_dir ='weights/'):
        super(GraphConvGrapHiC, self).__init__()
        
        self.hyperparameters = HYPERPARAMETERS
        self.device = device
        self.model_name = model_name
        self.dir_model = os.path.join(weight_dir, model_name)

        if not os.path.exists(self.dir_model):
            os.mkdir(self.dir_model)
        
        self.loss_function = GrapHiCLoss()

        self.conv0 = GCNConv(HYPERPARAMETERS['input_shape'], 16, 1, normalize=True)
        self.conv1 = GCNConv(16, 32, 1, normalize=True)
        self.conv2 = GCNConv(32, 32, 1, normalize=True)

        self.contact_cnn = ContactCNN(32, 32)

        

    def encode(self, x, edge_index, edge_attr, batch_index):
        x1 = torch.relu(self.conv0(x=x, edge_index=edge_index, edge_weight=edge_attr))
        x2 = torch.relu(self.conv1(x=x1, edge_index=edge_index, edge_weight=edge_attr))
        x3 = torch.relu(self.conv2(x=x2, edge_index=edge_index, edge_weight=edge_attr))
        x3 = self.process_graph_batch(x3, batch_index)

        return x3


    def decode(self, Z):
        Z = self.contact_cnn(Z, Z)
        return Z        

    def forward(self, x, edge_index, edge_attr, batch_index):
        Z = self.encode(x, edge_index, edge_attr, batch_index)

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
        
        weights = list(map(lambda x: (float(x.split('_')[1].split('-')[0]) ,os.path.join(self.dir_model, x)), os.listdir(self.dir_model)))
        req_weights = min(weights,key=itemgetter(0))[1]

        print("Loading: {}".format(req_weights))

        self.load_state_dict(torch.load(req_weights, map_location=self.device))

        


























