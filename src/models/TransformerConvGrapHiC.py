import torch
import torch.nn.functional as F 
from torch.nn import Linear, BatchNorm1d, ModuleList, Dropout
from torch_geometric.nn import TransformerConv
from torch_geometric.nn import Set2Set
from src.globals import SUB_MATRIX_SIZE

HYPERPARAMETERS = {
    'feature_size':1, 
    'embedding_size':64, 
    'n_heads':4, 
    'n_layers':5, 
    'dropout_rate':0.5, 
    'edge_dim':1, 
    'latent_dim':128,
    'set2set_steps': 4,
    'decoder_0': 256,
    'decoder_1': 512,
    'decoder_dropout': 0.5
}


 


class TransformerConvGrapHiC(torch.nn.Module):
    def __init__(self, HYPERPARAMETERS):
        
        super(TransformerConvGrapHiC, self).__init__()
        self.hyperparameters = HYPERPARAMETERS
        self.conv_layers = ModuleList([])
        self.transf_layers = ModuleList([])
        self.bn_layers = ModuleList([])

        # First layer of the network
        self.conv1 = TransformerConv(self.hyperparameters['feature_size'], 
                                    self.hyperparameters['embedding_size'], 
                                    heads=self.hyperparameters['n_heads'], 
                                    dropout=self.hyperparameters['dropout_rate'],
                                    edge_dim=self.hyperparameters['edge_dim'],
                                    beta=True) 

        self.transf1 = Linear(self.hyperparameters['embedding_size']*self.hyperparameters['n_heads'], 
                              self.hyperparameters['embedding_size'])

        self.bn1 = BatchNorm1d(self.hyperparameters['embedding_size'])

        # Remaining layers
        for i in range(self.hyperparameters['n_layers'] - 1):
            self.conv_layers.append(TransformerConv(self.hyperparameters['embedding_size'], 
                                                    self.hyperparameters['embedding_size'], 
                                                    heads=self.hyperparameters['n_heads'], 
                                                    dropout=self.hyperparameters['dropout_rate'],
                                                    edge_dim=self.hyperparameters['edge_dim'],
                                                    beta=True))

            self.transf_layers.append(Linear(self.hyperparameters['embedding_size']*self.hyperparameters['n_heads'], 
                                      self.hyperparameters['embedding_size']))
            self.bn_layers.append(BatchNorm1d(self.hyperparameters['embedding_size']))

        # Pooling to a graph representation
        self.pooling = Set2Set(self.hyperparameters['embedding_size'], 
                               processing_steps=self.hyperparameters['set2set_steps'])

        # Encoder latent variables
        self.mu_transform = Linear(2*self.hyperparameters['embedding_size'], 
                                            self.hyperparameters['latent_dim'])
        self.logvar_transform = Linear(2*self.hyperparameters['embedding_size'], 
                                            self.hyperparameters['latent_dim'])

        # Decoder
        self.decoder_0 = Linear(self.hyperparameters['latent_dim'], self.hyperparameters['decoder_0'])
        self.decoder_1 = Linear(self.hyperparameters['latent_dim'], self.hyperparameters['decoder_1'])
        self.decoder_2 = Linear(self.hyperparameters['latent_dim'], SUB_MATRIX_SIZE)

        decoder_dropout = Dropout(p=self.hyperparameters['decoder_dropout'])


    def decoder(self, z):
        output = torch.relu(self.decoder_dropout(self.decoder_0(z)))
        output = torch.relu(self.decoder_dropout(self.decoder_1(output)))
        output = torch.relu(self.decoder_dropout(self.decoder_2(output)))

        # Inner product Decoder
        output = torch.sigmoid(torch.mm(output, output.t()))

        return output



    def encoder(self, x, edge_attr, edge_index, batch_index):
        x = self.bn1(torch.relu(self.transf1(self.conv1(x, edge_index, edge_attr))))

        for i in range(self.hyperparameters['n_layers'] -1):
            x = self.bn_layers[i](torch.relu(self.transf_layers[i](self.conv_layers[i](x, edge_index, edge_attr))))
        
        x = self.pooling(x, batch_index)

        mu = self.mu_transform(x)
        logvar = self.logvar_transform(x)

        return mu, logvar

    def reparametrize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add(mu)
        else:
            return mu
        

    def forward(self, x, edge_attr, edge_index, batch_index):
        
        mu, logvar = self.encoder(x, edge_attr, edge_index, batch_index)
        z = self.reparametrize(mu, logvar)

        output = self.decoder(z)

        return output

    def encode(self, x, edge_attr, edge_index, batch_index):
        mu, logvar = self.encoder(x, edge_attr, edge_index, batch_index)
        z = self.reparametrize(mu, logvar)
        return z
    
    def decode(self, z):
        return self.decoder(z)