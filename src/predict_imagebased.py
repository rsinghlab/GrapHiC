from src import utils
import torch
from tqdm import tqdm
import numpy as np
from src.utils import spreadM, together, create_entire_path_directory
import os
import time

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

    # Clean the existing chromosomes
    if clean_existing_chromfiles:
        utils.delete_files(output_directory)
    
    if debug: print('Initializing the model parameters')
    
    # Move model to the defined device
    model.to(model.device)
    # Load the best weights
    model.load_weights()
    model.eval()
    
    # Dataloader function, this is defined by the model
    if debug: print('Loading the dataset')
    
    indices, compacts, sizes = data_info(np.load(dataset_file, allow_pickle=True))

    dataset_loader = model.load_data(dataset_file)
    
    result_data = []
    result_inds = []

    with torch.no_grad():
        for data in tqdm(dataset_loader, desc='Predicting: '):
            x = data[0]
            x = x.to(model.device)
            ind = data[2]

            output = model(x)
            result_data.append(output.to('cpu').numpy())
            result_inds.append(ind)
        
        
    result_data = np.concatenate(result_data, axis=0)
    result_inds = np.concatenate(result_inds, axis=0).reshape(-1, 4)

    predicted = together(result_data, result_inds, tag='Reconstructing: ')

    def save_data_n(key):
        file = os.path.join(output_directory, f'chr{key}.npz')
        save_data(predicted[key], compacts[key], sizes[key], file)

    if debug: print(f'Saving predicted data as individual chromosome files')

    for key in compacts.keys():
        save_data_n(key)

    print(f'All data saved. Running cost is {(time.time()-start)/60:.1f} min.')
