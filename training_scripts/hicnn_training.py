import sys
sys.path.append('../GrapHiC/')
import os
import torch
import json, hashlib

from src.utils import hic_file_paths
from src.utils import DATASET_DIRECTORY, PARSED_HIC_FILES_DIRECTORY, PREDICTED_FILES_DIRECTORY
from src.parse_hic_files import parse_hic_file, download_file
from related_work.hicnn import create_dataset, run
from src.models.HiCNN import HiCNN

# These hyperparameters go to HiCNN model controlling the training batch size, optimizer parameters 
HYPERPARAMETERS = {
    'epochs'            : 250, 
    'batch_size'        : 64, # Change to control per batch GPU memory requirement
    'optimizer_type'    : 'ADAM',
    'learning_rate'     : 0.0001,
    'momentum'          : 0.9
}

# Constants 
hic_data_resolution = 10000


# Step 1: Download and parse the HiC files
for hic_file_key in hic_file_paths.keys():
    if not os.path.exists(hic_file_paths[hic_file_key]['local_path']):
        download_file(hic_file_paths[hic_file_key])
    
    parse_hic_file(
        hic_file_paths[hic_file_key]['local_path'], 
        os.path.join(PARSED_HIC_FILES_DIRECTORY, hic_file_key),
        hic_data_resolution
    )

# These parameters are used by the dataset creator function to describe how to divide the chromosome matrices
cropping_params = {
    'sub_mat'   :40,
    'stride'    :28,
    'bounds'    :201,
    'padding'   :True
}

# These parameters are also used by the dataset creator function but this set contains the normalization parameters
normalization_params = {
    'norm'              : True,   # To normalize or not
    'remove_zeros'      : True,   # Remove zero before percentile calculation
    'set_diagonal_zero' : False,  # Remove the diagonal before percentile calculation
    'percentile'        : 99.90,  # Percentile 
    'rescale'           : True,   # After applying cutoff, rescale between 0 and 1
    'chrom_wide'        : True,   # Apply it on chromosome scale #TODO: Sample wise normalization isn't implemented
    'draw_dist_graphs'  : False,  # Visualize the distribution of the chromosome
    'edge_culling'      : -1,
}
non_informative_row_resolution_method = 'intersection' # This finds common non-informative rows in both dataset and then removes them. Can take ['ignore', 'target', 'intersection']

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# First we construct a string that is the dataset path
target = 'GM12878-geo-raoetal'



for base in ['GM12878-encode-0', 'GM12878-encode-1', 'GM12878-encode-2', 'GM12878-geo-026', 'GM12878-geo-033']:
    
    dataset_name = 'base:{}_target:{}_c:{}_s:{}_b:{}_n:{}_rz:{}_sdz:{}_p:{}_r:{}_nirrm:{}/'.format(
        base,
        target,
        cropping_params['sub_mat'],
        cropping_params['stride'],    
        cropping_params['bounds'],
        normalization_params['norm'],
        normalization_params['remove_zeros'],
        normalization_params['set_diagonal_zero'],
        normalization_params['percentile'],
        normalization_params['rescale'],
        non_informative_row_resolution_method,
    )
    
    # We create a hash because the path length is too long, we remember the hash to name correspondances in an external JSON. 

    dataset_name_hash = hashlib.sha1(dataset_name.encode('ascii')).hexdigest()

    # save hash in the external json for later references
    datasets_database = json.load(open('datasets.json'))

    if not datasets_database.get(dataset_name_hash):
        datasets_database[dataset_name_hash] = dataset_name
        with open("datasets.json", "w") as outfile:
            json.dump(datasets_database, outfile, indent=4)

    else:
        print('Entry already exists in the database')
    dataset_path = os.path.join(DATASET_DIRECTORY, dataset_name_hash)

    
    if not os.path.exists(os.path.join(dataset_path, 'train.npz')):
        create_dataset(
            os.path.join(PARSED_HIC_FILES_DIRECTORY, base ,'resolution_{}'.format(hic_data_resolution)),
            os.path.join(PARSED_HIC_FILES_DIRECTORY, target ,'resolution_{}'.format(hic_data_resolution)),
            dataset_path,
            cropping_params,
            normalization_params,
            non_informative_row_resolution_method,
            ['test', 'valid', 'train'] 
        )
    else:
        print('Dataset already exists!')

    model_name = 'hicnn-{}/'.format(base)
    hicnn_model = HiCNN(HYPERPARAMETERS, device, model_name)
    
    run(
        hicnn_model,
        os.path.join(dataset_path, 'train.npz'),
        os.path.join(dataset_path, 'valid.npz'),
        os.path.join(dataset_path, 'test.npz'),
        base,
        True
    )
    
