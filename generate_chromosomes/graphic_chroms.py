import sys
sys.path.append('../GrapHiC/')

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


# GrapHiC-Large
graphic_large_models = [
    'graphic-GM12878-encode-0-All', 
    # 'graphic-GM12878-encode-1-All', 
    # 'graphic-GM12878-encode-2-All', 
    # 'graphic-GM12878-geo-026-All', 
    # 'graphic-GM12878-geo-033-All'
]

# GrapHiC-Med
graphic_medium_models = [
    'graphic-hyperparameter-tuning-epi-factors-GrapHiC-Med',
]

# GrapHiC-Small
graphic_small_models = [
    'graphic-GM12878-encode-0-RAD-21',
]


# GrapHiC-Ablations
graphic_ablations =[
    'graphic-GM12878-encode-0-constant',
    'graphic-hyperparameter-tuning-pe-size-8'
]





def upscale(model, target, bases, epi_feature_set):
    cell_line = target.split('-')[0]
    for base in bases:        
        # Step 1: Create the dataset
        dataset_name = 'base:{}_target:{}_nenc:{}/'.format(
                base,
                target,
                epi_feature_set
        )
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
        
        # Step 2: Setup the experiment by creating the datasets and required directories
        if not os.path.exists(os.path.join(dataset_path, 'train.npz')):
            create_dataset_from_hic_files(
                os.path.join(PARSED_HIC_FILES_DIRECTORY, base ,'resolution_{}'.format(hic_data_resolution)),
                os.path.join(PARSED_HIC_FILES_DIRECTORY, target ,'resolution_{}'.format(hic_data_resolution)),
                dataset_path,
                node_encoding_files,
                dataset_creation_parameters,
                datasets=['test']
            )
        else:
            print('Dataset exists')
        
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

###################################################################################################
# # Upscale GM12878 Cell line datasets
# target = 'GM12878-geo-raoetal'
# bases = ['GM12878-encode-0'] #, 'GM12878-encode-1', 'GM12878-encode-2', 'GM12878-geo-026', 'GM12878-geo-033']



# # Upscale with graphic-large
# for model in graphic_large_models:
#     upscale(model, target, bases, 'All')


# # Upscale with graphic-medium
# for model in graphic_medium_models:
#     upscale(model, target, bases, 'GrapHiC-Med')

# # Upscale with graphic-small
# for model in graphic_small_models:
#     upscale(model, target, bases, 'RAD-21')

# # Upscale with graphic-ablations
# dataset_creation_parameters['positional_encoding_method'] = 'constant'
# dataset_creation_parameters['node_embedding_concat_method'] = 'positional'
# upscale('graphic-GM12878-encode-0-constant', target, bases, 'All')
# dataset_creation_parameters['positional_encoding_method'] = 'graph'
# dataset_creation_parameters['node_embedding_concat_method'] = 'concat'


# dataset_creation_parameters['node_embedding_concat_method'] = 'positional'
# dataset_creation_parameters['positional_encoding_dim'] = 13
# upscale('graphic-hyperparameter-tuning-pe-size-8', target, bases, 'All')
# dataset_creation_parameters['positional_encoding_dim'] = 8
# dataset_creation_parameters['node_embedding_concat_method'] = 'concat'




###################################################################################################
# # Upscale IMR90 datasets
target = 'IMR90-geo-raoetal'
bases = ['IMR90-geo-057']

dataset_creation_parameters['non_informative_row_resolution_method'] = 'target'
dataset_creation_parameters['add_expected_hic'] = True
# Upscale with graphic-large
for model in graphic_large_models:
    upscale(model, target, bases, 'All')

# Upscale with graphic-medium
for model in graphic_medium_models:
    upscale(model, target, bases, 'GrapHiC-Med')

# Upscale with graphic-small
for model in graphic_small_models:
    upscale(model, target, bases, 'RAD-21')
dataset_creation_parameters['add_expected_hic'] = False


# Upscale in cross cell type setting
bases = ['GM12878-encode-1']
# Upscale with graphic-large
for model in graphic_large_models:
    upscale(model, target, bases, 'All')

# Upscale with graphic-medium
for model in graphic_medium_models:
    upscale(model, target, bases, 'GrapHiC-Med')

# Upscale with graphic-small
for model in graphic_small_models:
    upscale(model, target, bases, 'RAD-21')

dataset_creation_parameters['non_informative_row_resolution_method'] = 'intersection'


# ###################################################################################################
# Upscale K562 datasets
target = 'K562-geo-raoetal'
bases = ['K562-geo-073']
dataset_creation_parameters['non_informative_row_resolution_method'] = 'target'

dataset_creation_parameters['add_expected_hic'] = True
# Upscale with graphic-large
for model in graphic_large_models:
    upscale(model, target, bases, 'All')

# Upscale with graphic-medium
for model in graphic_medium_models:
    upscale(model, target, bases, 'GrapHiC-Med')

# Upscale with graphic-small
for model in graphic_small_models:
    upscale(model, target, bases, 'RAD-21')

dataset_creation_parameters['add_expected_hic'] = False


# Upscale in cross cell type setting
bases = ['GM12878-encode-1']
# Upscale with graphic-large
for model in graphic_large_models:
    upscale(model, target, bases, 'All')

# Upscale with graphic-medium
for model in graphic_medium_models:
    upscale(model, target, bases, 'GrapHiC-Med')

# Upscale with graphic-small
for model in graphic_small_models:
    upscale(model, target, bases, 'RAD-21')
dataset_creation_parameters['non_informative_row_resolution_method'] = 'intersection'









#dataset_creation_parameters['replace_with_expected_hic'] = True

# target = 'IMR90-geo-raoetal'

# for model_name in ['graphic-hyperparameter-tuning-epi-factors-GrapHiC-Med']: #, 'graphic-GM12878-encode-1-All', 'graphic-GM12878-encode-2-All', 'graphic-GM12878-geo-026-All', 'graphic-GM12878-geo-033-All']:
#     for base in ['GM12878-encode-1']:
        
#         dataset_name = 'base:{}_target:{}_nenc:{}/'.format(
#                 base,
#                 target,
#                 'GrapHiC-Med' 
#         )
        
#         dataset_path = os.path.join(DATASET_DIRECTORY, dataset_name)
#         node_encoding_files = get_required_node_encoding_files_paths(
#             'IMR90', 
#             epigenetic_features['GrapHiC-Med']
#         )
        
#         input_embedding_size = dataset_creation_parameters['positional_encoding_dim'] + len(epigenetic_features['GrapHiC-Med'])
        
#         print('Input embedding size: {}'.format(input_embedding_size))
        
#         # Step 2: Setup the experiment by creating the datasets and required directories
#         #if not os.path.exists(os.path.join(dataset_path, 'train.npz')):
#         create_dataset_from_hic_files(
#             os.path.join(PARSED_HIC_FILES_DIRECTORY, base ,'resolution_{}'.format(hic_data_resolution)),
#             os.path.join(PARSED_HIC_FILES_DIRECTORY, target ,'resolution_{}'.format(hic_data_resolution)),
#             dataset_path,
#             node_encoding_files,
#             dataset_creation_parameters,
#             datasets=['test']
#         )
#         #else:
#         #   print('Dataset exists')
        
#         # Step 3: Create the model and initialize the weights
#         graphic_model = GrapHiC(
#             PARAMETERS, 
#             device, 
#             model_name, 
#             input_embedding_size=input_embedding_size
#         )
        
#         # Step 4: Run the main training loop with the model and the dataset
#         run(
#             graphic_model,
#             os.path.join(dataset_path, 'train.npz'),
#             os.path.join(dataset_path, 'valid.npz'),
#             os.path.join(dataset_path, 'test.npz'),
#             base,
#             False
#         )
        
# dataset_creation_parameters['add_expected_hic'] = True

# target = 'K562-geo-raoetal'
# for model_name in ['graphic-hyperparameter-tuning-epi-factors-GrapHiC-Med']:
#     for base in ['K562-geo-073']:
#         dataset_name = 'base:{}_target:{}_nenc:{}/'.format(
#                 base,
#                 target,
#                 'GrapHiC-Med'
#         )
        
#         dataset_path = os.path.join(DATASET_DIRECTORY, dataset_name)
#         node_encoding_files = get_required_node_encoding_files_paths(
#             'K562', 
#             epigenetic_features['GrapHiC-Med']
#         )
        
#         input_embedding_size = dataset_creation_parameters['positional_encoding_dim'] + len(epigenetic_features['GrapHiC-Med'])
        
#         print('Input embedding size: {}'.format(input_embedding_size))
        
#         # Step 2: Setup the experiment by creating the datasets and required directories
#         #if not os.path.exists(os.path.join(dataset_path, 'train.npz')):
#         create_dataset_from_hic_files(
#             os.path.join(PARSED_HIC_FILES_DIRECTORY, base ,'resolution_{}'.format(hic_data_resolution)),
#             os.path.join(PARSED_HIC_FILES_DIRECTORY, target ,'resolution_{}'.format(hic_data_resolution)),
#             dataset_path,
#             node_encoding_files,
#             dataset_creation_parameters,
#             datasets=['test']
#         )
#         #else:
#         #    print('Dataset exists')
        
#         # Step 3: Create the model and initialize the weights
#         graphic_model = GrapHiC(
#             PARAMETERS, 
#             device, 
#             model_name, 
#             input_embedding_size=input_embedding_size
#         )
        
#         # Step 4: Run the main training loop with the model and the dataset
#         run(
#             graphic_model,
#             os.path.join(dataset_path, 'train.npz'),
#             os.path.join(dataset_path, 'valid.npz'),
#             os.path.join(dataset_path, 'test.npz'),
#             base,
#             False
#         )
