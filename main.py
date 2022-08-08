import os
import json
import torch
import hashlib

from src.run import run

from src.models.GrapHiC import GrapHiC
from src.models.HiCPlus import HiCPlus
from src.parse_hic_files import parse_hic_file, download_hic_file
from src.dataset_creator import create_dataset_from_hic_files
from src.utils import HIC_FILES_DIRECTORY, PARSED_HIC_FILES_DIRECTORY, DATASET_DIRECTORY


### These are default parameters, to run interesting experiments change these parameters ###
hic_data_resolution = 10000

# These hyperparameters go to GrapHiC model controlling the training batch size, optimizer parameters 
HYPERPARAMETERS = {
    'epochs'            : 100, 
    'batch_size'        : 128, # Change to control per batch GPU memory requirement
    'optimizer_type'    : 'ADAM',
    'learning_rate'     : 0.001,
    'momentum'          : 0.9,
    'num_heads'         : 4,
}

# These parameters are used by the dataset creator function to describe how to divide the chromosome matrices
cropping_params = {
    'sub_mat'   :400,
    'stride'    :50,
    'bounds'    :190,
    'padding'   :True
}

# These parameters are also used by the dataset creator function but this set contains the normalization parameters
normalization_params = {
    'norm'              : True,  # To normalize or not
    'remove_zeros'      : True,  # Remove zero before percentile calculation
    'set_diagonal_zero' : False, # Remove the diagonal before percentile calculation
    'percentile'        : 99.0,  # Percentile 
    'rescale'           : True,  # After applying cutoff, rescale between 0 and 1
    'chrom_wide'        : True,  # Apply it on chromosome scale #TODO: Sample wise normalization isn't implemented
    'draw_dist_graphs'  : False  # Visualize the distribution of the chromosome
}

# Some other dataset creation paramters
positional_encoding_method = 'graph' # Required for GrapHiC, can take values between ['constant', 'monotonic', 'transformer', 'graph']
non_informative_row_resolution_method = 'intersection' # This finds common non-informative rows in both dataset and then removes them. Can take ['ignore', 'target', 'intersection']
noise_augmentation_method = 'none' # This function adds noise to all the input samples should improve training in certain cases. Can take ['none', 'random', 'uniform', 'gaussian']


base_files = ['hic026', 'encode0', 'hic033']



# These files should exist, (currently not using all of them but would at some point)
hic_file_paths = {
    'GM12878-geo-raoetal': {
            'local_path': os.path.join(HIC_FILES_DIRECTORY, 'GM12878', 'geo-raoetal.hic'),
            'remote_path': 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE63525&format=file&file=GSE63525%5FGM12878%5Finsitu%5Fprimary%2Breplicate%5Fcombined%5F30%2Ehic'
    },
    'GM12878-encode-0': {
            'local_path': os.path.join(HIC_FILES_DIRECTORY, 'GM12878', 'encode-0.hic'),
            'remote_path': 'https://www.encodeproject.org/files/ENCFF799QGA/@@download/ENCFF799QGA.hic'
    },
    'GM12878-encode-1': {
            'local_path': os.path.join(HIC_FILES_DIRECTORY, 'GM12878', 'encode-1.hic'),
            'remote_path': 'https://www.encodeproject.org/files/ENCFF473CAA/@@download/ENCFF473CAA.hic'
    },
    'GM12878-geo-026': {
            'local_path': os.path.join(HIC_FILES_DIRECTORY, 'GM12878', 'geo-026.hic'),
            'remote_path': 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM1551575&format=file&file=GSM1551575%5FHIC026%5F30%2Ehic'
    },
}


# Step 1: Download and parse the HiC files
for hic_file_key in hic_file_paths.keys():
    if not os.path.exists(hic_file_paths[hic_file_key]['local_path']):
        download_hic_file(hic_file_paths[hic_file_key])
    
    parse_hic_file(
        hic_file_paths[hic_file_key]['local_path'], 
        os.path.join(PARSED_HIC_FILES_DIRECTORY, hic_file_key),
        hic_data_resolution
    )


# Step 2: Create Dataset (Currently I am only creating dataset with base:encode-0 and target:rao-et-al)

# First we construct a string that is the dataset path
base = 'GM12878-encode-0'
target = 'GM12878-geo-raoetal'


dataset_name = 'base:{}_target:{}_c:{}_s:{}_b:{}_n:{}_rz:{}_sdz:{}_p:{}_r:{}_enc:{}_noi:{}_nirrm:{}/'.format(
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
    positional_encoding_method,
    noise_augmentation_method,
    non_informative_row_resolution_method
)
# We create a hash because the path length is too long, we remember the hash to name correspondances in an external JSON. 

dataset_name_hash = hashlib.sha1(dataset_name.encode('ascii')).hexdigest()

# save hash in the external json for later references
datasets_database = json.load(open('datasets.json'))

if not datasets_database.get(dataset_name_hash):
    datasets_database[dataset_name_hash] = dataset_name
    with open("datasets.json", "w") as outfile:
        json.dump(datasets_database, outfile)

else:
    print('Entry already exists in the database')




dataset_path = os.path.join(DATASET_DIRECTORY, dataset_name_hash)


if not os.path.exists(os.path.join(dataset_path, 'train.npz')):
    create_dataset_from_hic_files(
        os.path.join(PARSED_HIC_FILES_DIRECTORY, base ,'resolution_{}'.format(hic_data_resolution)),
        os.path.join(PARSED_HIC_FILES_DIRECTORY, target ,'resolution_{}'.format(hic_data_resolution)),
        dataset_path,
        positional_encoding_method,
        [],
        cropping_params,
        normalization_params,
        noise_augmentation_method,
        non_informative_row_resolution_method,
        ['train', 'test', 'valid']
    )
else:
    print('Dataset already exists!')

############################################## TRAIN AND RUN GRAPHIC MODEL ###############################################################
#Step3: Create the Model

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


model_name = 'graphic(embd-64)_l1loss+tvloss(e-04)_dhash:{}/'.format(
    dataset_name_hash
)

graphic_model = GrapHiC(HYPERPARAMETERS, device, model_name)

# Step 4: Run the main training and evaluation loop
run(
    graphic_model,
    os.path.join(dataset_path, 'train.npz'),
    os.path.join(dataset_path, 'valid.npz'),
    os.path.join(dataset_path, 'test.npz')
)
#############################################################################################################################################

############################################### TRAIN AND RUN HiCPLUS MODEL ###############################################################
# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda:0" if use_cuda else "cpu")


# model_name = 'hicplus_dhash:{}/'.format(
#     dataset_name_hash
# )
# hicplus_model = HiCPlus(HYPERPARAMETERS, device, model_name)

# run(
#     hicplus_model,
#     os.path.join(dataset_path, 'train.npz'),
#     os.path.join(dataset_path, 'valid.npz'),
#     os.path.join(dataset_path, 'test.npz')
# )

#############################################################################################################################################


















# precentiles = [99.9]

# for percentile in precentiles:
#     normalization_params['cutoff'] = percentile

#     for base_file in base_files:

#         HYPERPARAMETERS['input_shape'] = 4

#         model_name = 'graphic_l1loss_cell:{}_base:{}_c:{}_s:{}_b:{}_n:{}_rz:{}_sdz:{}_p:{}_r:{}_cw:{}_enc:{}'.format(
#             'GM12878',
#             base_file,
#             cropping_params['chunk_size'],
#             cropping_params['stride'],    
#             cropping_params['bounds'],
#             normalization_params['norm'],
#             normalization_params['remove_zeros'],
#             normalization_params['set_diagonal_zero'],
#             normalization_params['cutoff'],
#             normalization_params['rescale'],
#             normalization_params['chrom_wide'],
#             'graph'
#         )
#         print(model_name)

#         dataset_path = '/home/murtaza/Documents/GrapHiC/data/datasets/real/{}/cell:{}_base:{}_c:{}_s:{}_b:{}_n:{}_rz:{}_sdz:{}_p:{}_r:{}_cw:{}_enc:{}/'.format(
#             'GM12878',
#             'GM12878',
#             base_file,
#             cropping_params['chunk_size'],
#             cropping_params['stride'],    
#             cropping_params['bounds'],
#             normalization_params['norm'],
#             normalization_params['remove_zeros'],
#             normalization_params['set_diagonal_zero'],
#             normalization_params['cutoff'],
#             normalization_params['rescale'],
#             normalization_params['chrom_wide'],
#             'graph'
#         )
#         print(dataset_path)

#         #if not os.path.exists(os.path.join(dataset_path, 'train.npz')):
#         create_dataset_from_hic_files(
#             '/media/murtaza/ubuntu2/hic_data/chromosome_files/real/{}_{}'.format('GM12878', base_file),
#             '/media/murtaza/ubuntu2/hic_data/chromosome_files/real/{}_rao_et_al'.format('GM12878'),
#             dataset_path,
#             'graph',
#             [],
#             cropping_params,
#             normalization_params,
#             None,
#             'intersection',
#             ['train', 'valid', 'test']
#         )

#         #else:
#         #   print('Dataset already exists!')





      


#         graph_train(graphic_model, 
#                     os.path.join(dataset_path, 'train.npz'),
#                     os.path.join(dataset_path, 'valid.npz'),
#                     model_name,
#                     clean_existing_weights=True, debug=True
#         )

#         


# chromosomes = ['chr1', 'chr10',model_name = 'graphic_insulation+l1loss_c:{}_s:{}_b:{}_n:{}_rz:{}_sdz:{}_p:{}_r:{}_cw:{}_enc:{}'.format(
#             cropping_params['chunk_size'],
#             cropping_params['stride'],    
#             cropping_params['bounds'],
#             normalization_params['norm'],
#             normalization_params['remove_zeros'],
#             normalization_params['set_diagonal_zero'],
#             normalization_params['cutoff'],
#             normalization_params['rescale'],
#             normalization_params['chrom_wide'],
#             encoding
#         ) 'chr19']
# rzs = [True, False]
# sdzs = [True, False]


# for cell_line in cell_lines:
#     for chromosome in chromosomes:
#         matrix = load_hic_npz_file('/media/murtaza/ubuntu2/hic_data/chromosome_files/real/{}_rao_et_al/{}.npz'.format(cell_line, chromosome))['hic']
#         for rz in rzs:
#             for sdz in sdzs:
#                 normalization_params['remove_zeros'] = rz
#                 normalization_params['set_diagonal_zero'] = sdz
                
#                 normalized_matrix = normalize_hic_matrix(matrix, normalization_params, cell_line, chromosome)

# precentiles = [95]

# for percentile in precentiles:
#     normalization_params['cutoff'] = percentile

#     for encoding in pos_encoding:

#         HYPERPARAMETERS['input_shape'] = input_shape[encoding]

#         model_name = 'graphic_insulation+l1loss_c:{}_s:{}_b:{}_n:{}_rz:{}_sdz:{}_p:{}_r:{}_cw:{}_enc:{}'.format(
#             cropping_params['chunk_size'],
#             cropping_params['stride'],    
#             cropping_params['bounds'],
#             normalization_params['norm'],
#             normalization_params['remove_zeros'],
#             normalization_params['set_diagonal_zero'],
#             normalization_params['cutoff'],
#             normalization_params['rescale'],
#             normalization_params['chrom_wide'],
#             encoding
#         )
#         print(model_name)

#         dataset_path = '/users/gmurtaza/GrapHiC/data/datasets/real/{}/c:{}_s:{}_b:{}_n:{}_rz:{}_sdz:{}_p:{}_r:{}_cw:{}_enc:{}/'.format(
#             cell_lines[cell_line_idx],
#             cropping_params['chunk_size'],
#             cropping_params['stride'],    
#             cropping_params['bounds'],
#             normalization_params['norm'],
#             normalization_params['remove_zeros'],
#             normalization_params['set_diagonal_zero'],
#             normalization_params['cutoff'],
#             normalization_params['rescale'],
#             normalization_params['chrom_wide'],
#             encoding
#         )
#         print(dataset_path)

#         if not os.path.exists(os.path.join(dataset_path, 'train.npz')):
#             create_dataset_from_hic_files(
#                 '/users/gmurtaza/data/gmurtaza/parsed_hic_datasets/H1/resolution_10000'.format(cell_lines[cell_line_idx]),
#                 '/users/gmurtaza/data/gmurtaza/parsed_hic_datasets/H1/resolution_10000'.format(cell_lines[cell_line_idx]),
#                 dataset_path,
#                 encoding,
#                 [],
#                 cropping_params,
#                 normalization_params,
#                 'none',
#                 'intersection',
#                 ['train', 'valid', 'test']
#             )

#         else:
#             print('Dataset already exists!')





# embedding = torch.rand(64, 200, 32)

# fullly_connected = FullyConnected(32, 32)
# contact_pred = ContactCNN(32, 32)


# print(contact_pred.forward(embedding, embedding).shape)



# model_name = 'graphic-200-no-cutoff'
# graphic_model = GraphConvGrapHiC(HYPERPARAMETERS, device, 'weights/{}'.format(model_name))

# graph_train(graphic_model, 'data/datasets/real/{}/h1/train.npz'.format(model_name), 
#             'data/datasets/real/{}/h1/valid.npz'.format(model_name)
#             ,model_name,
#             clean_existing_weights=True, debug=True)



# graphic_model.load_weights()
# graphic_model.eval()


# graph_predict(graphic_model, 
#               '/home/murtaza/Documents/GrapHiC/data/datasets/real/{}/h1/test.npz'.format(model_name), 
#               '/home/murtaza/Documents/GrapHiC/data/predicted/H1_{}_H1/'.format(model_name), 
#               True
#             )


# visualize_matrix(
#     '/home/murtaza/Documents/GrapHiC/data/predicted/H1_{}_H1/chr19.npz'.format(model_name),
#     '/home/murtaza/Documents/GrapHiC/outputs/visualizations/{}_chr19_1000-1200.png'.format(model_name),
#     1000,
#     200
# )





# image_predict(
#     hicplus_model,
#     'data/datasets/real/c40_s28/encode_0/test.npz',
#     '/home/murtaza/Documents/GrapHiC/data/predicted/GM12878_hicplus-16_encode0/'
# )

# image_predict(
#     hicnn_model,
#     'data/datasets/real/c40_s28/encode_0/test.npz',
#     '/home/murtaza/Documents/GrapHiC/data/predicted/GM12878_hicnn-16_encode0/'
# )

# image_predict(
#     hicnn2_model,
#     'data/datasets/real/c40_s28/encode_0/test.npz',
#     '/home/murtaza/Documents/GrapHiC/data/predicted/GM12878_hicnn2-16_encode0/'
# )

# graph_predict(graphic_model, 
#              '/home/murtaza/Documents/GrapHiC/data/datasets/real/graphic40/valid.npz', 
#              '/home/murtaza/Documents/GrapHiC/data/predicted/GM12878_graphic40_encode0/'
# )

# visualize_matrix(
#     '/home/murtaza/Documents/GrapHiC/data/predicted/GM12878_hicplus-16_encode0/chr19.npz',
#     '/home/murtaza/Documents/GrapHiC/outputs/visualizations/graphic40_chr8_1000-1500.png',
#     200,
#     200
# )

# visualize_matrix(
#     '/home/murtaza/Documents/GrapHiC/data/predicted/GM12878_hicnn-16_encode0/chr19.npz',
#     '/home/murtaza/Documents/GrapHiC/outputs/visualizations/graphic40_chr8_1000-1500.png',
#     200,
#     200
# )
# visualize_matrix(
#     '/home/murtaza/Documents/GrapHiC/data/predicted/GM12878_hicnn2-16_encode0/chr19.npz',
#     '/home/murtaza/Documents/GrapHiC/outputs/visualizations/graphic40_chr8_1000-1500.png',
#     200,
#     200
# )


# import numpy as np




# # matrix = np.load('/media/murtaza/ubuntu/updated_hic_data/data/chromosome_files/GM12878_rao_et_al/chr22.npz', allow_pickle=True)['hic']
# # print(matrix.shape[0])

# # with open('test.smat', 'w') as f:
# #     for i in range(matrix.shape[0]):
# #         for j in range(matrix.shape[1]):
# #             if matrix[i][j] != 0:
# #                 f.write('{}\t{}\t{}\n'.format(i, j, matrix[i][j]))


# lrc_dataset_to_readcount = {
#     'encode0': 50,
#     'encode1': 50,
#     'encode2': 25,
#     'hic023': 16,
#     'hic033': 100,
#     'hic057': 16,
#     'hic073': 16
# }

# lrc_datasets = {
#     'GM12878': ['encode0', 'encode1', 'encode2', 'hic026', 'hic033'], 
#     'IMR90': ['hic057'],
#     'K562': ['hic073']
# }

# inputs_base_directory = 'data/datasets/'
# output_base_directory = '/media/murtaza/ubuntu/hic_data/chromosome_files/'
# weights_path = 'weights/'


# def upscale(model_name, input_path, output_path):
#     model = model_name.split('-')[0]
    
#     #Check if the model exists
#     if not os.path.exists(os.path.join(weights_path, model_name)):
#         print('Provided model configuration {} doesnt exist! Exiting...'.format(model_name))
#         exit(1)

#     if model == 'hicplus':
#         model = HiCPlus(HYPERPARAMETERS, device, os.path.join(weights_path, model_name))
    
#     elif model == 'hicnn':
#         model = HiCNN(HYPERPARAMETERS, device, os.path.join(weights_path, model_name))
        
#     elif model == 'hicnn2':
#         model = HiCNN2(HYPERPARAMETERS, device, os.path.join(weights_path, model_name))

#     elif model == 'deephic':
#         model = DeepHiC(HYPERPARAMETERS, device, os.path.join(weights_path, model_name))

#     elif model == 'vehicle':
#         return

#     elif model == 'smoothing':
#         model = Smoothing(HYPERPARAMETERS, device)

#     else:
#         print('Invalid Model {} requested! Exiting...'.format(model))
#         exit(1)
        
#     image_predict(
#         model, 
#         input_path, 
#         output_path
#     )


########################################################################### Upscaling Synthetic Datasets ########################################################################################################################

# # Upscale synthetic datasets with c40s40 models
# dataset_configuration = 'synthetic'
# chunk_stride = 'c40_s40'
# model_names = ['smoothing-gaussian', 'deephic-16', 'deephic-25', 'deephic-50', 'deephic-100'] 


# for cell_line in ['GM12878', 'IMR90', 'K562']:
#     for dataset in ['synthetic16', 'synthetic25', 'synthetic50', 'synthetic100']:
#         for model_name in model_names:
#             upscale(
#                 model_name, 
#                 os.path.join(inputs_base_directory, dataset_configuration, chunk_stride, cell_line, dataset, 'test.npz'),
#                 os.path.join(output_base_directory, dataset_configuration, '{}_{}_{}'.format(cell_line, model_name, dataset))
#             )


# Upscale synthetic datasets with c40s28 models
# dataset_configuration = 'synthetic'
# chunk_stride = 'c40_s28'
# model_names = [
#         'hicplus-16', 'hicplus-25', 'hicplus-50', 'hicplus-100',
#         'hicnn-16', 'hicnn-25', 'hicnn-50', 'hicnn-100',
#         'hicnn2-16', 'hicnn2-25', 'hicnn2-50', 'hicnn2-100',
# ] 

# for cell_line in ['GM12878', 'IMR90', 'K562']:
#     for dataset in ['synthetic16' 'synthetic25', 'synthetic50', 'synthetic100']:
#         for model_name in model_names:
#             upscale(
#                 model_name, 
#                 os.path.join(inputs_base_directory, dataset_configuration, chunk_stride, cell_line, dataset, 'test.npz'),
#                 os.path.join(output_base_directory, dataset_configuration, '{}_{}_{}'.format(cell_line, model_name, dataset))
#             )


# # Upscale synthetic datasets with c269_s257 models
# dataset_configuration = 'synthetic'
# chunk_stride = 'c269_s257'
# model_names = ['vehicle'] 
# for cell_line in ['GM12878', 'IMR90', 'K562']:
#     for dataset in ['synthetic16', 'synthetic25', 'synthetic50', 'synthetic100']:
#         for model_name in model_names:
#             upscale(
#                 model_name, 
#                 os.path.join(inputs_base_directory, dataset_configuration, chunk_stride, cell_line, dataset),
#                 os.path.join(output_base_directory, dataset_configuration, '{}_{}_{}'.format(cell_line, model_name, dataset))
#             )

#################################################################################################################################################################################################################################


########################################################################### Upscaling LRC Datasets ##############################################################################################################################

# dataset_configuration = 'real'
# chunk_stride = 'c40_s40'
# model_names = ['smoothing-gaussian', 'deephic-16', 'deephic-25', 'deephic-50', 'deephic-100'] 

# for cell_line in lrc_datasets.keys():
#     for dataset in lrc_datasets[cell_line]:
#         for model_name in model_names:
#             upscale(
#                 model_name, 
#                 os.path.join(inputs_base_directory, dataset_configuration, chunk_stride, cell_line, dataset, 'test.npz'),
#                 os.path.join(output_base_directory, dataset_configuration, '{}_{}_{}'.format(cell_line, model_name, dataset))
#             )


# dataset_configuration = 'real'
# chunk_stride = 'c40_s28'
# model_names = [
#         # 'hicplus-16', 'hicplus-25', 'hicplus-50', 'hicplus-100',
#         # 'hicnn-16', 'hicnn-25', 'hicnn-50', 'hicnn-100',
#         # 'hicnn2-16', 'hicnn2-25', 'hicnn2-50', 'hicnn2-100',
#         # 'hicnn-encode0', 'hicnn-encode2', 'hicnn-hic026', 'hicnn-synthetic-ensemble',
#         # 'hicnn-real-ensemble', 
#         'hicnn-random-noise', 'hicnn-gaussian-noise', 
#         'hicnn-uniform-noise'
# ]

# for cell_line in lrc_datasets.keys():
#     for dataset in lrc_datasets[cell_line]:
#         for model_name in model_names:
#             upscale(
#                 model_name, 
#                 os.path.join(inputs_base_directory, dataset_configuration, chunk_stride, cell_line, dataset, 'test.npz'),
#                 os.path.join(output_base_directory, dataset_configuration, '{}_{}_{}'.format(cell_line, model_name, dataset))
#             )


# dataset_configuration = 'real'
# chunk_stride = 'c269_s257'
# model_names = ['vehicle'] 

# for cell_line in lrc_datasets.keys():
#     for dataset in lrc_datasets[cell_line]:
#         for model_name in model_names:
#             upscale(
#                 model_name, 
#                 os.path.join(inputs_base_directory, dataset_configuration, chunk_stride, cell_line, dataset),
#                 os.path.join(output_base_directory, dataset_configuration, '{}_{}_{}'.format(cell_line, model_name, dataset))
#             )

#################################################################################################################################################################################################################################


