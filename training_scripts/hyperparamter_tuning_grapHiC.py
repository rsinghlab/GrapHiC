import sys
sys.path.append('../GrapHiC/')

import os
import torch
from src.run import run
from parameters import *
from src.models.GrapHiC import GrapHiC
from src.parse_hic_files import download_all_hic_datasets
from src.dataset_creator import create_dataset_from_hic_files
from src.utils import  get_required_node_encoding_files_paths
from src.utils import PARSED_HIC_FILES_DIRECTORY, DATASET_DIRECTORY
from src.epigentic_encodings import download_all_epigenetic_datasets

# Concerned cell lines
base = 'GM12878-encode-0'
target = 'GM12878-geo-raoetal'

# Experiment Name
experiment = 'graphic-hyperparameter-tuning'

# Retrain?
retrain = True

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")



def download_datasets():
    download_all_hic_datasets()
    download_all_epigenetic_datasets()

def create_dataset(dataset_name, epi_features_set):
    dataset_path = os.path.join(DATASET_DIRECTORY, dataset_name)
    node_encoding_files = get_required_node_encoding_files_paths('GM12878', epigenetic_features[epi_features_set])
    #if not os.path.exists(os.path.join(dataset_path, 'train.npz')):
    create_dataset_from_hic_files(
        os.path.join(PARSED_HIC_FILES_DIRECTORY, base ,'resolution_{}'.format(hic_data_resolution)),
        os.path.join(PARSED_HIC_FILES_DIRECTORY, target ,'resolution_{}'.format(hic_data_resolution)),
        dataset_path,
        node_encoding_files,
        dataset_creation_parameters
    )
    # else:
    #   print('Dataset exists')
    return dataset_path


# # optimize the pe-method
# for pe_method in ['constant', 'monotonic', 'transformer', 'graph']:
#     dataset_name = experiment + '-{}'.format(pe_method)
#     # Update dataset creation parameters
#     dataset_creation_parameters['positional_encoding_method'] = pe_method
#     dataset_creation_parameters['node_embedding_concat_method'] = 'positional'
#     dataset_path = create_dataset(dataset_name, 'All')
        
#     # Determine the input_embedding_size based on the node_embedding_concatenation method
#     input_embedding_size =  dataset_creation_parameters['positional_encoding_dim']
#     print('Input embedding size: {}'.format(input_embedding_size))

#     model_name = experiment + '-{}'.format(pe_method)
#     # Step 3: Create model
#     graphic_model = GrapHiC(
#         PARAMETERS, 
#         device, 
#         model_name, 
#         input_embedding_size=input_embedding_size
#     )
    
#     run(
#         graphic_model,
#         os.path.join(dataset_path, 'train.npz'),
#         os.path.join(dataset_path, 'valid.npz'),
#         os.path.join(dataset_path, 'test.npz'),
#         base,
#         retrain
#     )

# # Reset the mutated parameters to their original values explicitly 
# dataset_creation_parameters['positional_encoding_method']   =   'graph'
# dataset_creation_parameters['node_embedding_concat_method'] =   'concat'


# # optimize for pe-encoding-size
# for pe_encoding_size in [2, 4, 8, 13]:
#     dataset_name = experiment + '-pe-size-{}'.format(pe_encoding_size)
#     # Update dataset creation parameters
#     dataset_creation_parameters['node_embedding_concat_method'] = 'positional'
#     dataset_path = create_dataset(dataset_name, 'All')
        
#     # Determine the input_embedding_size based on the node_embedding_concatenation method
#     input_embedding_size =  dataset_creation_parameters['positional_encoding_dim']
#     print('Input embedding size: {}'.format(input_embedding_size))

#     model_name = experiment + '-pe-size-{}'.format(pe_encoding_size)
#     # Step 3: Create model
#     graphic_model = GrapHiC(
#         PARAMETERS, 
#         device, 
#         model_name, 
#         input_embedding_size=input_embedding_size
#     )
    
#     run(
#         graphic_model,
#         os.path.join(dataset_path, 'train.npz'),
#         os.path.join(dataset_path, 'valid.npz'),
#         os.path.join(dataset_path, 'test.npz'),
#         base,
#         retrain
#     )
    


dataset_creation_parameters['node_embedding_concat_method'] = 'concat'

# optimize for a single-epi-factor
for epi_factors in epigenetic_features.keys():
    if epi_factors == 'None':
        continue

    dataset_name = experiment + '-epi-factors-{}'.format(epi_factors)
    # Update dataset creation parameters
    dataset_path = create_dataset(dataset_name, epi_factors)
        
    # Determine the input_embedding_size based on the node_embedding_concatenation method
    input_embedding_size =  dataset_creation_parameters['positional_encoding_dim'] + len(epigenetic_features[epi_factors])
    print('Input embedding size: {}'.format(input_embedding_size))

    model_name = experiment + '-epi-factors-{}'.format(epi_factors)
    # Step 3: Create model
    graphic_model = GrapHiC(
        PARAMETERS, 
        device, 
        model_name, 
        input_embedding_size=input_embedding_size
    )
    
    run(
        graphic_model,
        os.path.join(dataset_path, 'train.npz'),
        os.path.join(dataset_path, 'valid.npz'),
        os.path.join(dataset_path, 'test.npz'),
        base,
        retrain
    )


# # optimize for merging operator
# for merging_operator in ['concat', 'mean', 'sum', 'epigenetic']:
    
#     dataset_name = experiment + '-merging-operator-{}'.format(merging_operator)
    
    
#     # Update dataset creation parameters
#     dataset_creation_parameters['node_embedding_concat_method'] = merging_operator

    
#     dataset_path = create_dataset(dataset_name, 'All')
    
#     # Determine the input_embedding_size based on the node_embedding_concatenation method
#     if merging_operator == 'concat':
#         input_embedding_size =  dataset_creation_parameters['positional_encoding_dim'] + len(epigenetic_features['All'])
#     else:
#         input_embedding_size = len(epigenetic_features['All'])
    
#     print('Input embedding size: {}'.format(input_embedding_size))

#     model_name = experiment + '-merging-operator-{}'.format(merging_operator)
#     # Step 3: Create model
#     graphic_model = GrapHiC(
#         PARAMETERS, 
#         device, 
#         model_name, 
#         input_embedding_size=input_embedding_size
#     )
    
#     run(
#         graphic_model,
#         os.path.join(dataset_path, 'train.npz'),
#         os.path.join(dataset_path, 'valid.npz'),
#         os.path.join(dataset_path, 'test.npz'),
#         base,
#         retrain
#     )

# dataset_creation_parameters['node_embedding_concat_method'] = 'concat'

# # optimize for graph conv operation
# for conv_operation in ['GAT', 'GCN', 'GConv']:
#     dataset_name = experiment + '-conv-algo-{}'.format(conv_operation)
    
#     # Update dataset creation parameters
#     PARAMETERS['graphconvalgo'] = conv_operation

    
#     dataset_path = create_dataset(dataset_name, 'All')
    
#     # Determine the input_embedding_size based on the node_embedding_concatenation method
    
#     input_embedding_size =  dataset_creation_parameters['positional_encoding_dim'] + len(epigenetic_features['All'])
    
#     print('Input embedding size: {}'.format(input_embedding_size))

#     model_name = experiment + '-conv-algo-{}'.format(conv_operation)
#     # Step 3: Create model
#     graphic_model = GrapHiC(
#         PARAMETERS, 
#         device, 
#         model_name, 
#         input_embedding_size=input_embedding_size
#     )
    
#     run(
#         graphic_model,
#         os.path.join(dataset_path, 'train.npz'),
#         os.path.join(dataset_path, 'valid.npz'),
#         os.path.join(dataset_path, 'test.npz'),
#         base,
#         retrain
#     )

# # optimize for number of graph conv operations 
# PARAMETERS['graphconvalgo'] = 'Transformer'
# for num_conv_operation in [1, 2, 3]:
#     dataset_name = experiment + '-num-conv-layers-{}'.format(num_conv_operation)
    
    
#     # Update dataset creation parameters
#     PARAMETERS['graphconvblocks'] = num_conv_operation

    
#     dataset_path = create_dataset(dataset_name, 'All')
    
#     # Determine the input_embedding_size based on the node_embedding_concatenation method
    
#     input_embedding_size =  dataset_creation_parameters['positional_encoding_dim'] + len(epigenetic_features['All'])
    
#     print('Input embedding size: {}'.format(input_embedding_size))

#     model_name = experiment + '-num-conv-layers-{}'.format(num_conv_operation)
#     # Step 3: Create model
#     graphic_model = GrapHiC(
#         PARAMETERS, 
#         device, 
#         model_name, 
#         input_embedding_size=input_embedding_size
#     )
    
#     run(
#         graphic_model,
#         os.path.join(dataset_path, 'train.npz'),
#         os.path.join(dataset_path, 'valid.npz'),
#         os.path.join(dataset_path, 'test.npz'),
#         base,
#         retrain
#     )

# PARAMETERS['graphconvblocks'] = 1


# # optimize for number of residual layers 
# for num_resblocks_operation in [1, 2, 3, 5]:
#     dataset_name = experiment + '-num-res-layers-{}'.format(num_resblocks_operation)
    
    
#     # Update dataset creation parameters
#     PARAMETERS['resblocks'] = num_resblocks_operation

    
#     dataset_path = create_dataset(dataset_name, 'All')
    
#     # Determine the input_embedding_size based on the node_embedding_concatenation method
    
#     input_embedding_size =  dataset_creation_parameters['positional_encoding_dim'] + len(epigenetic_features['All'])
    
#     print('Input embedding size: {}'.format(input_embedding_size))

#     model_name = experiment + '-num-res-layers-{}'.format(num_resblocks_operation)
#     # Step 3: Create model
#     graphic_model = GrapHiC(
#         PARAMETERS, 
#         device, 
#         model_name, 
#         input_embedding_size=input_embedding_size
#     )
    
#     run(
#         graphic_model,
#         os.path.join(dataset_path, 'train.npz'),
#         os.path.join(dataset_path, 'valid.npz'),
#         os.path.join(dataset_path, 'test.npz'),
#         base,
#         retrain
#     )
    

# PARAMETERS['resblocks'] = 5

# # optimize for loss function 

# for loss_func in ['MSE', 'L1']:
#     dataset_name = experiment + '-loss-func-{}'.format(loss_func)
    
    
#     # Update dataset creation parameters
#     PARAMETERS['loss_func'] = loss_func

    
#     dataset_path = create_dataset(dataset_name, 'All')
    
#     # Determine the input_embedding_size based on the node_embedding_concatenation method
    
#     input_embedding_size =  dataset_creation_parameters['positional_encoding_dim'] + len(epigenetic_features['All'])
    
#     print('Input embedding size: {}'.format(input_embedding_size))

#     model_name = experiment + '-loss-func-{}'.format(loss_func)
#     # Step 3: Create model
#     graphic_model = GrapHiC(
#         PARAMETERS, 
#         device, 
#         model_name, 
#         input_embedding_size=input_embedding_size
#     )
    
#     run(
#         graphic_model,
#         os.path.join(dataset_path, 'train.npz'),
#         os.path.join(dataset_path, 'valid.npz'),
#         os.path.join(dataset_path, 'test.npz'),
#         base,
#         retrain
#     )



























































# # Step 4: Run the main training and evaluation loop







    # if base == 'GM12878-encode-0':
    #     retrain = True
    # else:
    #     retrain = False

    # # Change target when doing cross cell type predictions
    # if base == 'K562-geo-073':
    #     target = 'K562-geo-raoetal'
    # if base == 'HUVEC-geo-056':
    #     target = 'HUVEC-geo-raoetal'



    #Step3: Create the Model
   

   
   ############################################################################################################################################

