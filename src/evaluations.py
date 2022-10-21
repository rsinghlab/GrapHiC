'''
    This file contains the evaluation functions
    1. MSE, L1Norm 
    2. SSIM (Image Similarity Metric)
    3. PCC, SCC (Statistical correlations)
    4. HiCRep (HiC Similarity)
    5. Insulation Score (TAD profile Similarity)
'''
import numpy as np
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity
from scipy.stats import pearsonr


def MSE(outputs, targets):
    batch_mse = 0.0
    for i in range(outputs.shape[0]):
        batch_mse += mean_squared_error(
            outputs[i, 0, :, :].detach().to('cpu').numpy().astype(float),
            targets[i, 0, :, :].to('cpu').numpy().astype(float)
        )
    
    return batch_mse/outputs.shape[0]

    
def SSIM(outputs, targets):
    batch_structural_similarity = 0.0
    for i in range(outputs.shape[0]):
        x = outputs[i, 0, :, :].detach().to('cpu').numpy().astype(float)
        y = targets[i, 0, :, :].to('cpu').numpy().astype(float)
        
        batch_structural_similarity += structural_similarity(
            x,
            y,
            data_range=x.max() - x.min()
        )
    
    return batch_structural_similarity/outputs.shape[0]

def PCC(outputs, targets):
    batch_pcc = 0.0
    for i in range(outputs.shape[0]):
        r, _ = pearsonr(
            outputs[i, 0, :, :].detach().to('cpu').numpy().flatten().astype(float),
            targets[i, 0, :, :].to('cpu').numpy().flatten().astype(float)
        )
        batch_pcc += r
    
    return batch_pcc/outputs.shape[0]

