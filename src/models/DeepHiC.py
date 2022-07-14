import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vgg import vgg16
import numpy as np
import os, sys
from operator import itemgetter


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
    
class DeepHiC(nn.Module):
    def __init__(self, HYPERPARAMETERS, device, dir_model='weights/deephic', in_channel=1, resblock_num=5):
        super(DeepHiC, self).__init__()
        self.hyperparameters = HYPERPARAMETERS
        self.device = device
        self.dir_model = dir_model

        if not os.path.exists(self.dir_model):
            os.mkdir(self.dir_model)


        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=9, stride=1, padding=4)
        # have a swish here in forward
        
        resblocks = [residualBlock(64) for _ in range(resblock_num)]
        self.resblocks = nn.Sequential(*resblocks)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # have a swish here in forward

        self.conv3 = nn.Conv2d(64, in_channel, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        emb = swish(self.conv1(x))
        x   =       self.resblocks(emb)
        x   = swish(self.bn2(self.conv2(x)))
        x   =       self.conv3(x + emb)
        return (torch.tanh(x) + 1) / 2

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
        if scheme not in ['min-valid-loss', 'last-epoch']:
            print('Weight loading scheme not supported!')
            exit(1)
        
        weights = list(map(lambda x: (float(x.split('_')[1].split('-')[0]) ,os.path.join(self.dir_model, x)), os.listdir(self.dir_model)))
        req_weights = max(weights,key=itemgetter(0))[1]
        print(req_weights)

        self.load_state_dict(torch.load(req_weights, map_location=self.device))




class Discriminator(nn.Module):
    def __init__(self, in_channel=3):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 64, 3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        # Replaced original paper FC layers with FCN
        self.conv7 = nn.Conv2d(256, 1, 1, stride=1, padding=0)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size = x.size(0)

        x = swish(self.conv1(x))
        x = swish(self.bn2(self.conv2(x)))
        x = swish(self.bn3(self.conv3(x)))
        x = swish(self.bn4(self.conv4(x)))
        x = swish(self.bn5(self.conv5(x)))
        x = swish(self.bn6(self.conv6(x)))

        x = self.conv7(x)
        x = self.avgpool(x)
        return torch.sigmoid(x.view(batch_size))

class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()

    def forward(self, out_labels, out_images, target_images):
        # Adversarial Loss
        adversarial_loss = 1 - out_labels
        # Perception Loss
        out_feat = self.loss_network(out_images.repeat([1,3,1,1]))
        target_feat = self.loss_network(target_images.repeat([1,3,1,1]))
        perception_loss = self.mse_loss(out_feat.reshape(out_feat.size(0),-1), target_feat.reshape(target_feat.size(0),-1))
        # Image Loss
        image_loss = self.mse_loss(out_images, target_images)
        tv_loss = self.tv_loss(out_images)
        return image_loss + 0.001 * adversarial_loss + 0.006 * perception_loss + 2e-8 * tv_loss

class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        b, c, h, w = x.shape
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h-1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w-1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / b

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


