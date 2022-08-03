from src import utils
import torch
from tqdm import tqdm
import numpy as np
from src.matrix_operations import spreadM, together
from src.utils import create_entire_path_directory
import os
import time
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr as _pearsonr

def pearsonr(x, y):
    r, _ = _pearsonr(x.flatten(), y.flatten())
    return r




REDMAP = LinearSegmentedColormap.from_list("bright_red", [(1,1,1),(1,0,0)])

def data_info(data):
    indices = data['inds']
    compacts = data['compacts'][()]
    sizes = data['sizes'][()]
    return indices, compacts, sizes

def save_data(predicted_hic, compact, size, file):
    hic = spreadM(predicted_hic, compact, size, convert_int=False, verbose=True)
    np.savez_compressed(file, hic=hic, compact=compact)
    print('Saving file:', file)



def predict(model, dataset_file, output_directory, clean_existing_chromfiles=False, debug=True):
    start = time.time()
    create_entire_path_directory(output_directory)
    chromosome_output_dir = os.path.join(output_directory, 'chromosomes')
    visualizations_output_dir = os.path.join(output_directory, 'visualizations')
    create_entire_path_directory(chromosome_output_dir)
    create_entire_path_directory(visualizations_output_dir)
    


    # Clean the existing chromosomes
    if clean_existing_chromfiles:
        utils.delete_files(output_directory)
    
    if debug: print('Initializing the model parameters')
    
    # Move model to the defined device
    model.to(model.device)
    # Load the best weights
    model.load_weights()
    #model.eval()
    
    # Dataloader function, this is defined by the model
    if debug: print('Loading the dataset')
    
    indices, compacts, sizes = data_info(np.load(dataset_file, allow_pickle=True))

    dataset_loader = model.load_data(dataset_file)
    

    baseline_mse_score = 0.0
    baseline_ssim_score = 0.0
    baseline_pcc_score = 0.0


    upscaled_mse_score = 0.0
    upscaled_ssim_score = 0.0
    upscaled_pcc_score = 0.0

    total_samples = 0
    
    for data in tqdm(dataset_loader, desc='Predicting: '):
        data = data.to(model.device)
        outputs = model(data)
        
        targets = model.process_graph_batch(data.y, data.batch)
        inputs = model.process_graph_batch(data.input, data.batch)
        
        for idx in range(outputs.shape[0]):
            input = inputs[idx, :, :].to('cpu').numpy()
            target = targets[idx, :, :].to('cpu').numpy()
            output = outputs[idx, 0, :, :].detach().to('cpu').numpy()

            baseline_mse_score += mean_squared_error(input, target)
            upscaled_mse_score += mean_squared_error(output, target)

            baseline_ssim_score += ssim(input, target)
            upscaled_ssim_score += ssim(output, target)
            
            baseline_pcc_score += pearsonr(input, target)
            upscaled_pcc_score += pearsonr(output, target)
            total_samples += 1
            
            
            
            plt.matshow(input, cmap=REDMAP)
            plt.axis('off')
            plt.savefig(os.path.join(visualizations_output_dir, 'idx:{}_input.png'.format(idx)))
            plt.close()
            plt.matshow(target, cmap=REDMAP)
            plt.axis('off')
            plt.savefig(os.path.join(visualizations_output_dir, 'idx:{}_targets.png'.format(idx)))
            plt.close()
            plt.matshow(output, cmap=REDMAP)
            plt.axis('off')
            plt.savefig(os.path.join(visualizations_output_dir, 'idx:{}_outputs.png'.format(idx)))
            plt.close()


    print('Mean Squared Error --- Baseline: {}, Generated: {}'.format(
        (baseline_mse_score/total_samples), (upscaled_mse_score/total_samples)
    ))
    print('SSIM --- Baseline: {}, Generated: {}'.format(
        (baseline_ssim_score/total_samples), (upscaled_ssim_score/total_samples)
    ))
    print('Pearson\'s Correlation Coefficient --- Baseline: {}, Generated: {}'.format(
        (baseline_pcc_score/total_samples), (upscaled_pcc_score/total_samples)
    ))




    # result_data = np.concatenate(result_data, axis=0)
    # result_inds = np.concatenate(result_inds, axis=0).reshape(-1, 4)

    # predicted = together(result_data, result_inds, tag='Reconstructing: ')

    # def save_data_n(key):
    #     file = os.path.join(chromosome_output_dir, f'chr{key}.npz')
    #     save_data(predicted[key], compacts[key], sizes[key], file)

    # if debug: print(f'Saving predicted data as individual chromosome files')

    # for key in compacts.keys():
    #     save_data_n(key)

    # print(f'All data saved. Running cost is {(time.time()-start)/60:.1f} min.')
