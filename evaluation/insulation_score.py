import torch
import numpy as np

class InsulationLoss(torch.nn.Module):
    def __init__(self, window_radius=20, deriv_size=20):
        super(InsulationLoss, self).__init__()
        self.deriv_size     = deriv_size
        self.window_radius  = window_radius
        self.di_pool        = torch.nn.AvgPool2d(kernel_size=window_radius, stride=1)
        self.top_pool       = torch.nn.AvgPool1d(kernel_size=deriv_size, stride=1)
        self.bottom_pool    = torch.nn.AvgPool1d(kernel_size=deriv_size, stride=1)
        self.mse = torch.nn.MSELoss()

    def indivInsulation(self, x):
        iv     = self.di_pool(x)
        iv     = torch.diagonal(iv, dim1=2, dim2=3)       
        iv     = torch.log2(iv/torch.mean(iv))
        top    = self.top_pool(iv[:,:,self.deriv_size:])
        bottom = self.bottom_pool(iv[:,:,:-self.deriv_size])
        dv     = (top-bottom)
        return dv

    def forward(self, output, target):
        out_dv = self.indivInsulation(output)
        tar_dv = self.indivInsulation(target)
        loss  = self.mse(tar_dv, out_dv).detach().numpy()
        return loss
        



def compute_insulation_score(
    args
):
    
    x, y = args
    x = x*255
    y = y*255
    
    x = x.reshape(1, 1, x.shape[0], x.shape[1])
    y = y.reshape(1, 1, y.shape[0], y.shape[1])
    
    insulation_score = InsulationLoss()
    
    return insulation_score(torch.from_numpy(x), torch.from_numpy(y))

            
    
    