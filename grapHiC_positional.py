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



# Experiment Name
experiment = 'graphic-positional'

# Concerned cell lines
base = 'GM12878-encode-0'
target = 'GM12878-geo-raoetal'

# Epigenetic features set
epi_features_set = 'All'

# Retrain?
retrain = False

# Step 1: Make sure all the datasets are downloaded locally
download_all_hic_datasets()
download_all_epigenetic_datasets()

dataset_creation_parameters['node_embedding_concat_method'] = 'positional'


for base in ['GM12878-encode-0', 'GM12878-encode-1', 'GM12878-encode-2', 'GM12878-geo-026', 'GM12878-geo-033']:
    ############################################## TRAIN AND RUN GRAPHIC MODEL ###############################################################
    # Step 2: Setup the experiment by creating the datasets and required directories
    dataset_name = 'exp:{}_base:{}_target:{}_nenc:{}/'.format(
            experiment,
            base,
            target,
            'All'
    )
    if base == 'GM12878-encode-0':
        retrain = True
    else:
        retrain = False

    # Change target when doing cross cell type predictions
    # if base == 'K562-geo-073':
    #     target = 'K562-geo-raoetal'
    # if base == 'HUVEC-geo-056':
    #     target = 'HUVEC-geo-raoetal'



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
        print('Dataset exists')

    #Step3: Create the Model

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")


    model_name = 'graphic-positional'

    print(model_name)

    input_embedding_size = 13
    
    
    
    print('Input embedding size: {}'.format(input_embedding_size))

    graphic_model = GrapHiC(
        PARAMETERS, 
        device, 
        model_name, 
        input_embedding_size=input_embedding_size
    )


    #Step 4: Run the main training and evaluation loop
    run(
        graphic_model,
        os.path.join(dataset_path, 'train.npz'),
        os.path.join(dataset_path, 'valid.npz'),
        os.path.join(dataset_path, 'test.npz'),
        base,
        retrain
    )
   ############################################################################################################################################








































