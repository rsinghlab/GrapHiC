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


PARAMETERS['epochs'] = 250

# Experiment Name
experiment = 'encoding-merging-eval'

# Concerned cell lines
base = 'GM12878-encode-1'
target = 'GM12878-geo-raoetal'

# Epigenetic features set
epi_features_set = 'All'

# Retrain?
retrain = False

# Step 1: Make sure all the datasets are downloaded locally
download_all_hic_datasets()
download_all_epigenetic_datasets()

for encoding_merging in ['epigenetic']:
    experiment = encoding_merging + '-' + experiment
    dataset_creation_parameters['node_embedding_concat_method'] = encoding_merging
    


    ############################################## TRAIN AND RUN GRAPHIC MODEL ###############################################################
    # Step 2: Setup the experiment by creating the datasets and required directories
    dataset_name = 'exp:{}_base:{}_target:{}_nenc:{}/'.format(
            experiment,
            base,
            target,
            'All'
    )
    dataset_path = os.path.join(DATASET_DIRECTORY, dataset_name)
    print(dataset_path)

    node_encoding_files = get_required_node_encoding_files_paths('GM12878', epigenetic_features[epi_features_set])

    print(node_encoding_files)

    if not os.path.exists(os.path.join(dataset_path, 'train.npz')):
        create_dataset_from_hic_files(
            os.path.join(PARSED_HIC_FILES_DIRECTORY, base ,'resolution_{}'.format(hic_data_resolution)),
            os.path.join(PARSED_HIC_FILES_DIRECTORY, target ,'resolution_{}'.format(hic_data_resolution)),
            dataset_path,
            node_encoding_files,
            dataset_creation_parameters
        )
    else:
        print('Dataset exists!')
        
    #Step3: Create the Model

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")


    model_name = 'graphic_{}/'.format(
        experiment
    )
    print(model_name)

    # Determine the input_embedding_size based on the node_embedding_concatenation method
    if dataset_creation_parameters['node_embedding_concat_method'] != 'concat':
        input_embedding_size = len(node_encoding_files)
    else:
        input_embedding_size = 2*len(node_encoding_files)

    graphic_model = GrapHiC(
        PARAMETERS, 
        device, 
        model_name, 
        input_embedding_size=input_embedding_size
    )


    # Step 4: Run the main training and evaluation loop
    run(
        graphic_model,
        os.path.join(dataset_path, 'train.npz'),
        os.path.join(dataset_path, 'valid.npz'),
        os.path.join(dataset_path, 'test.npz'),
        base,
        retrain
    )
    # ############################################################################################################################################








































