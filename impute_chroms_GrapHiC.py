import os
import torch
from parameters import *

from src.parse_hic_files import download_all_hic_datasets
from src.utils import  get_required_node_encoding_files_paths
from src.utils import PARSED_HIC_FILES_DIRECTORY, DATASET_DIRECTORY
from src.epigentic_encodings import download_all_epigenetic_datasets

from src.run import run
from src.models.GrapHiC import GrapHiC
from src.dataset_creator import create_dataset_from_hic_files

# Step 1: Make sure all the datasets are downloaded locally
download_all_hic_datasets()
download_all_epigenetic_datasets()

# Setup training device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


def upscale(model, target, base, epi_feature_set, experiment):
    dataset_name = '{}_base:{}_target:{}_nenc:{}/'.format(
            experiment,
            base,
            target,
            epi_feature_set
    )
    
    cell_line = target.split('-')[0]
    
    if 'grch38' in base:
        cell_line = target.split('-')[0] + '-GRCH38'
    
    if 'MM10' in base:
        cell_line = target.split('-')[0] + '-MM10'
    
    
    # Step 1: Create the dataset
    dataset_path = os.path.join(DATASET_DIRECTORY, dataset_name)
    node_encoding_files = get_required_node_encoding_files_paths(
        cell_line, 
        epigenetic_features[epi_feature_set]
    )
    if dataset_creation_parameters['node_embedding_concat_method'] == 'concat':
        input_embedding_size = dataset_creation_parameters['positional_encoding_dim'] + len(epigenetic_features[epi_feature_set])
    elif dataset_creation_parameters['node_embedding_concat_method'] == 'positional':
        input_embedding_size = dataset_creation_parameters['positional_encoding_dim']
    elif dataset_creation_parameters['node_embedding_concat_method'] == 'epigenetic':
        input_embedding_size = len(epigenetic_features[epi_feature_set])
    else:
        input_embedding_size = dataset_creation_parameters['positional_encoding_dim']   
    
    
    print('Input embedding size: {}'.format(input_embedding_size))
    
    print(base, target, dataset_path, node_encoding_files)
    
    # Step 2: Setup the experiment by creating the datasets
    create_dataset_from_hic_files(
        os.path.join(PARSED_HIC_FILES_DIRECTORY, base ,'resolution_{}'.format(hic_data_resolution)),
        os.path.join(PARSED_HIC_FILES_DIRECTORY, target ,'resolution_{}'.format(hic_data_resolution)),
        dataset_path,
        node_encoding_files,
        dataset_creation_parameters,
        datasets=['test']
    )
    
    # Step 3: Create the model and initialize the weights
    graphic_model = GrapHiC(
        PARAMETERS, 
        device, 
        model, 
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
        False
    )    


datasets = {
    # GM12878 Datasets
    'GM12878-encode-0'  : 'GM12878-geo-raoetal',
    'GM12878-encode-1'  : 'GM12878-geo-raoetal',
    'GM12878-encode-2'  : 'GM12878-geo-raoetal',
    'GM12878-geo-026'   : 'GM12878-geo-raoetal',
    'GM12878-geo-033'   : 'GM12878-geo-raoetal',
    # Cross-celltype
    'IMR90-geo-057'     : 'IMR90-geo-raoetal'  ,
    'K562-geo-073'      : 'K562-geo-raoetal'   ,
}

models = {
    'graphic-final-encode-0-GrapHiC-Trimmed'
}



def upscale_all_datasets(models, datasets):
    PARAMETERS['decoderstyle'] = 'Unet'

    for model, epi_set in models.items():
        if 'pos' in model:
            dataset_creation_parameters['node_embedding_concat_method'] = 'positional'
        if 'prior' in model:
            dataset_creation_parameters['replace_with_expected_hic'] = True
        
        
        for input, target in datasets.items():
            if 'IMR90' in input:
                dataset_creation_parameters['add_expected_hic'] = True
            if 'K562' in input:
                dataset_creation_parameters['add_expected_hic'] = True
            
            upscale(model, target, input, epi_set, 'new-exps')

            if 'IMR90' in input:
                dataset_creation_parameters['add_expected_hic'] = False
            if 'K562' in input:
                dataset_creation_parameters['add_expected_hic'] = False
        
        if 'pos' in model:
            dataset_creation_parameters['node_embedding_concat_method'] = 'concat'
        if 'prior' in model:
            dataset_creation_parameters['replace_with_expected_hic'] = False


upscale_all_datasets(models, datasets)