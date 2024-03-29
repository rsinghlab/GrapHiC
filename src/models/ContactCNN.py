import torch
import torch.nn.functional as F 
import torch.nn as nn



def swish(x):
    return x * torch.sigmoid(x)

class residualBlock(nn.Module):
    def __init__(self, channels, k=3, s=1):
        super(residualBlock, self).__init__()

        self.conv1 = nn.Conv2d(channels, channels, k, stride=s, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        # a swish layer here
        self.conv2 = nn.Conv2d(channels, channels, k, stride=s, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = swish(self.bn1(self.conv1(x)))
        residual =       self.bn2(self.conv2(residual))
        return x + residual



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

    def __init__(self, embed_dim, hidden_dim, activation=nn.LeakyReLU(0.2)):
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

        z0 = z0.transpose(1, 2)
        z1 = z1.transpose(1, 2)

        z_mul = z0.unsqueeze(3) * z1.unsqueeze(2)
        
        c = self.conv(z_mul)
        c = self.activation(c)
        
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
        self, embed_dim, hidden_dim=32, 
        residual_blocks=5, width=7, 
        activation=nn.Sigmoid()
    ):
        super(ContactCNN, self).__init__()

        self.hidden = FullyConnected(embed_dim, hidden_dim)
        
        resblocks = [residualBlock(hidden_dim) for _ in range(residual_blocks)]
        self.resblocks = nn.Sequential(*resblocks)

        self.conv = nn.Conv2d(hidden_dim, 1, width, padding=width // 2)
        self.clip()
        
        self.activation = activation
        
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
        C = self.resblocks(self.cmap(z0, z1))
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

        s = self.conv(C)
        s = self.activation(s)
        return s


