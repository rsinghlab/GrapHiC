import sys
sys.path.append('../GrapHiC/')

import os
import json
import torch
import hashlib

from parameters import *

from src.parse_hic_files import download_all_hic_datasets
from src.utils import  get_required_node_encoding_files_paths
from src.utils import PARSED_HIC_FILES_DIRECTORY, DATASET_DIRECTORY
from src.epigentic_encodings import download_all_epigenetic_datasets

from src.run import run
from src.models.GrapHiC import GrapHiC
from src.dataset_creator import create_dataset_from_hic_files



# Setup training device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# Step 1: Make sure all the datasets are downloaded locally
download_all_hic_datasets()
download_all_epigenetic_datasets()

def train_grapHiC(base, target, epi_factors, model_name):
    dataset_name = '{}:{}_target:{}_nenc:{}/'.format(
        model_name,
        base,
        target,
        epi_factors
    )
    dataset_path = os.path.join(DATASET_DIRECTORY, dataset_name)
    
    cell_line = target.split('-')[0]
    
    node_encoding_files = get_required_node_encoding_files_paths(
        cell_line, 
        epigenetic_features[epi_factors]
    )
    
    model_name = '{}-{}-{}'.format(
        model_name,
        base,
        epi_factors
    )
    
    print('Training Setup: base dataset -- {}, feature set -- {} , model -- {}'.format(
        base, 
        epi_factors,
        model_name
    ))
    
    input_embedding_size = dataset_creation_parameters['positional_encoding_dim'] + len(epigenetic_features[epi_factors])
    
    print('Input embedding size: {}'.format(input_embedding_size))
    
    # Step 2: Setup the experiment by creating the datasets and required directories
    if not os.path.exists(os.path.join(dataset_path, 'train.npz')):
        create_dataset_from_hic_files(
            os.path.join(PARSED_HIC_FILES_DIRECTORY, base ,'resolution_{}'.format(hic_data_resolution)),
            os.path.join(PARSED_HIC_FILES_DIRECTORY, target ,'resolution_{}'.format(hic_data_resolution)),
            dataset_path,
            node_encoding_files,
            dataset_creation_parameters
        )
    else:
        print('Dataset exists')
    
    # Step 3: Create the model and initialize the weights
    graphic_model = GrapHiC(
        PARAMETERS, 
        device, 
        model_name, 
        input_embedding_size=input_embedding_size
    )

    # Step 4: Run the main training loop with the model and the dataset
    run(
        graphic_model,
        os.path.join(dataset_path, 'train.npz'),
        os.path.join(dataset_path, 'valid.npz'),
        os.path.join(dataset_path, 'test.npz'),
        base,
        target,
        True
    )


train_grapHiC('GM12878-encode-0', 'GM12878-geo-raoetal', 'GrapHiC-Trimmed', 'graphic-final')
