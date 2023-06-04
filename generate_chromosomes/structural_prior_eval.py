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
# download_all_hic_datasets()
# download_all_epigenetic_datasets()

# Setup training device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# GrapHiC-Structural Priors:

dataset_creation_parameters['sub_mat'] = 256


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






datasets = {
    # GM12878 Datasets
    # 'GM12878-encode-0'  : 'GM12878-geo-raoetal',
    # 'GM12878-encode-1'  : 'GM12878-geo-raoetal',
    # 'GM12878-encode-2'  : 'GM12878-geo-raoetal',
    # 'GM12878-geo-026'   : 'GM12878-geo-raoetal',
    # 'GM12878-geo-033'   : 'GM12878-geo-raoetal',
    # Cross-celltype
    # 'IMR90-geo-057'     : 'IMR90-geo-raoetal'  ,
    # 'K562-geo-073'      : 'K562-geo-raoetal'   ,
    # 'HUVEC-geo-059'     : 'HUVEC-geo-raoetal'  ,
    'GM12878-encode-grch38-lrc-0'   : 'GM12878-encode-grch38-hrc-0',
    'K562-4dn-grch38-lrc-0'         : 'K562-encode-grch38-hrc-0',
}

models = {
    # 'graphic-final-GM12878-encode-0-All'                : 'All',
    'graphic-final-GM12878-encode-0-GrapHiC-Trimmed'    : 'GrapHiC-Trimmed',
    # 'graphic-final-GM12878-encode-0-CTCF'               : 'CTCF',
    # 'graphic-final-pos-GM12878-encode-0-CTCF'           : 'CTCF',
    # 'graphic-basic-GM12878-encode-0-CTCF'               : 'CTCF',
    
}



def upscale_all_datasets(models, datasets):
    PARAMETERS['decoderstyle'] = 'Unet'

    for model, epi_set in models.items():
        if 'pos' in model:
            dataset_creation_parameters['node_embedding_concat_method'] = 'positional'
        if 'basic' in model:
            PARAMETERS['decoderstyle'] = 'ContactCNN'
            dataset_creation_parameters['positional_encoding_method'] = 'constant'
            dataset_creation_parameters['node_embedding_concat_method'] = 'positional'

        for input, target in datasets.items():
            if 'IMR90' in input:
                dataset_creation_parameters['add_expected_hic'] = True
            if 'HUVEC' in input:
                dataset_creation_parameters['add_expected_hic'] = True
            if 'K562' in input:
                dataset_creation_parameters['add_expected_hic'] = True
            
            
            upscale(model, target, input, epi_set, 'new-exps')

            if 'IMR90' in input:
                dataset_creation_parameters['add_expected_hic'] = False
            if 'HUVEC' in input:
                dataset_creation_parameters['add_expected_hic'] = False
            if 'K562' in input:
                dataset_creation_parameters['add_expected_hic'] = False
        
        if 'pos' in model:
            dataset_creation_parameters['node_embedding_concat_method'] = 'concat'
        if 'basic' in model:
            PARAMETERS['decoderstyle'] = 'Unet'
            dataset_creation_parameters['positional_encoding_method'] = 'concat'
            dataset_creation_parameters['node_embedding_concat_method'] = 'graph'
    


upscale_all_datasets(models, datasets)





# models = {
#     # 'graphic-simple-prior-final-GM12878-encode-0-All'                    : 'All',
#     'graphic-simple-prior-final-GM12878-encode-0-GrapHiC-Trimmed'        : 'GrapHiC-Trimmed',
#     # 'graphic-simple-prior-final-GM12878-encode-0-CTCF'                   : 'CTCF',
# }





# datasets = [
#     # 'GM12878-geo-raoetal',  'IMR90-geo-raoetal', 'K562-geo-raoetal', 'HUVEC-geo-raoetal' , 
#     'H1-geo-raoetal', 'HFF-encode-grch38-hrc-0'
# ]




# def upscale_with_structural_prior(models, datasets):
#     dataset_creation_parameters['replace_with_expected_hic'] = True
#     dataset_creation_parameters['non_informative_row_resolution_method'] = 'target'
    
#     base = 'GM12878-encode-0'


#     for model, epi_set in models.items():
#         for target in datasets:
#             if 'H1' in target:
#                 base = 'H1-4dn-0'
#             if 'HFF' in target:
#                 base = 'H1-4dn-0'
            
            
#             upscale(model, target, base, epi_set, 'simple-prior-new-exps')

#             if 'H1' in target:
#                 base = 'GM12878-encode-0'
#             if 'HFF' in target:
#                 base = 'GM12878-encode-0'

            
# upscale_with_structural_prior(models, datasets)















# ### GRCH38 datasets
# models = {
#     
# }

datasets = {
    # GM12878 Datasets
    'GM12878-encode-grch38-lrc-0'   : 'GM12878-encode-grch38-hrc-0',
    'K562-4dn-grch38-lrc-0'         : 'K562-encode-grch38-hrc-0',
}




# def upscale_grch38_datasets(models, datasets):
#     PARAMETERS['decoderstyle'] = 'Unet'

#     for model, epi_set in models.items():
#         for input, target in datasets.items():
            
#             upscale(model, target, input, epi_set, 'grch38-exps')

            
#     PARAMETERS['decoderstyle'] = 'ContactCNN'


#upscale_grch38_datasets(models, datasets)




# models = {
#     'graphic-grch38-prior-GM12878-encode-grch38-lrc-0-GrapHiC-Trimmed' : 'GrapHiC-Trimmed'
# }

# datasets = [
#     'H1-encode-grch38-hrc-0', 'HFF-encode-grch38-hrc-0'
# ]


# def upscale_prior_grch38_datasets(models, datasets):
#     dataset_creation_parameters['replace_with_expected_hic'] = True
#     dataset_creation_parameters['non_informative_row_resolution_method'] = 'target'
#     base = 'GM12878-encode-grch38-lrc-0'
    
#     for model, epi_set in models.items():
#         for target in datasets:
#             upscale(model, target, base, epi_set, 'simple-prior-grch38-exps')




# upscale_prior_grch38_datasets(models, datasets)







# ###################################################################################################

# dataset_creation_parameters['replace_with_expected_hic'] = True
# # Upscale GM12878 Cell line datasets
# target = 'GM12878-geo-raoetal'
# bases = ['GM12878-geo-033'] #, 'GM12878-encode-1', 'GM12878-encode-2', 'GM12878-geo-026', 'GM12878-geo-033']

# for model, epi_set in zip(models, epi_sets):
#     upscale(model, target, bases, epi_set)
    
# # target = 'H1-geo-raoetal'
# # bases = ['GM12878-geo-033'] #, 'GM12878-encode-1', 'GM12878-encode-2', 'GM12878-geo-026', 'GM12878-geo-033']

# # # Upscale with graphic-large
# # for model, epi_set in zip(models, epi_sets):
# #     upscale(model, target, bases, epi_set)

# # target = 'IMR90-geo-raoetal'
# # bases = ['GM12878-geo-033'] #, 'GM12878-encode-1', 'GM12878-encode-2', 'GM12878-geo-026', 'GM12878-geo-033']

# # # Upscale with graphic-large
# # for model, epi_set in zip(models, epi_sets):
# #     upscale(model, target, bases, epi_set)

# # target = 'HUVEC-geo-raoetal'
# # bases = ['GM12878-geo-033'] #, 'GM12878-encode-1', 'GM12878-encode-2', 'GM12878-geo-026', 'GM12878-geo-033']

# # # Upscale with graphic-large
# # for model, epi_set in zip(models, epi_sets):
# #     upscale(model, target, bases, epi_set)

# # target = 'K562-geo-raoetal'
# # bases = ['GM12878-geo-033'] #, 'GM12878-encode-1', 'GM12878-encode-2', 'GM12878-geo-026', 'GM12878-geo-033']

# # # Upscale with graphic-large
# # for model, epi_set in zip(models, epi_sets):
# #     upscale(model, target, bases, epi_set)

# dataset_creation_parameters['replace_with_expected_hic'] = False
