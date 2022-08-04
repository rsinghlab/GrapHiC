import os
import torch
import numpy as np

from src.parse_hic_files import parse_hic_file
from src.dataset_creator import create_dataset_from_hic_files

# import matplotlib.pyplot as plt

# from src.models.DeepHiC import DeepHiC
# from src.models.HiCNN import HiCNN
# from src.models.HiCNN2 import HiCNN2
# from src.models.HiCPlus import HiCPlus
# from src.models.Smoothing import Smoothing
# from src.models.GrapHiC import GraphConvGrapHiC, FullyConnected, ContactCNN
# from src.predict_imagebased import predict as image_predict
# from src.train_imagebased import train as image_train
# from src.train_graphbased import train as graph_train
# from src.predict_graphbased import predict as graph_predict
#from torchsummary import summary


# from src.utils import graph_positional_encoding, visualize_matrix, normalize_hic_matrix, create_dataset_from_hic_files
# from src.difficult_to_map_region_analysis import compare_regions
# print(compare_regions(
#     '/media/murtaza/ubuntu2/hic_data/chromosome_files/real/H1_rao_et_al/', 
#     '/media/murtaza/ubuntu2/updated_hic_data/data/difficult_to_map_regions/regions.bed'
# ), ' overlap precentage for chromosome 1')


# parse_hic_file(
#     '/users/gmurtaza/data/gmurtaza/hic_datasets/H1/4DN/4dn-0.hic',
#     '/users/gmurtaza/data/gmurtaza/parsed_hic_datasets/H1/',
#     10000 
# )


# HYPERPARAMETERS = {
#     'batch_size': 128,
#     'learning_rate': 0.001,
#     'momentum': 0.9,
#     'epochs': 100,
#     'optimizer_type': 'ADAM',
#     'input_shape': -1
# }

# cropping_params={
#     'chunk_size':200,
#     'stride'    :200,
#     'bounds'    :190,
#     'padding'   :True
# }

normalization_params={
    'norm'              : True,
    'remove_zeros'      : True,
    'set_diagonal_zero' : False,
    'cutoff'            : 95.0,
    'rescale'           : True,
    'chrom_wide'        : True, 
    'draw_dist_graphs'  : False
}
# # pos_encoding_idx = 3
# pos_encoding = ['graph']

# input_shape = {
#     'constant': 1,
#     'monotonic': 1,
#     'transformer': 4,
#     'graph': 4
# }



# cell_line_idx = 3
# cell_lines = ['GM12878', 'IMR90', 'K562', 'H1', 'HG002']



# dataset_path = '/users/gmurtaza/GrapHiC/data/datasets/real/{}/c:{}_s:{}_b:{}_n:{}_rz:{}_sdz:{}_p:{}_r:{}_cw:{}_enc:{}/'.format(
#     cell_lines[cell_line_idx],
#     cropping_params['chunk_size'],
#     cropping_params['stride'],    
#     cropping_params['bounds'],
#     normalization_params['norm'],
#     normalization_params['remove_zeros'],
#     normalization_params['set_diagonal_zero'],
#     normalization_params['cutoff'],
#     normalization_params['rescale'],
#     normalization_params['chrom_wide'],
#     pos_encoding[0]
# )
# print(dataset_path)

# if not os.path.exists(os.path.join(dataset_path, 'train')):
#     create_dataset_from_hic_files(
#         '/users/gmurtaza/data/gmurtaza/parsed_hic_datasets/H1/resolution_10000'.format(cell_lines[cell_line_idx]),
#         '/users/gmurtaza/data/gmurtaza/parsed_hic_datasets/H1/resolution_10000'.format(cell_lines[cell_line_idx]),
#         dataset_path,
#         pos_encoding[0],
#         [],
#         cropping_params,
#         normalization_params,
#         None,
#         'intersection',
#         ['train', 'valid', 'test'],
#         True
#     )
# else:
#     print('Dataset already exists')

# model_name = 'graphic_l1loss_cell:{}_target:{}_c:{}_s:{}_b:{}_n:{}_rz:{}_sdz:{}_p:{}_r:{}_cw:{}_enc:{}'.format(
#         'H1',
#         'H1',
#         cropping_params['chunk_size'],
#         cropping_params['stride'],    
#         cropping_params['bounds'],
#         normalization_params['norm'],
#         normalization_params['remove_zeros'],
#         normalization_params['set_diagonal_zero'],
#         normalization_params['cutoff'],
#         normalization_params['rescale'],
#         normalization_params['chrom_wide'],
#         'graph'
# )
from src.utils import load_hic_file
from src.matrix_operations import graph_rw_smoothing
from matplotlib.colors import LinearSegmentedColormap
from src.normalizations import normalize_hic_matrix
import matplotlib.pyplot as plt
from scipy.special import softmax

# REDMAP = LinearSegmentedColormap.from_list("bright_red", [(1,1,1),(1,0,0)])


# # print(model_name)
# hic_matrix = load_hic_file('/users/gmurtaza/data/gmurtaza/parsed_hic_datasets/H1/resolution_10000/chr10.npz')['hic']
# hic_matrix = normalize_hic_matrix(hic_matrix, normalization_params, 'H1', 'chr10')

# hic_matrix = hic_matrix[1000:1200, 1000:1200]
# hic_matrix = softmax(hic_matrix, axis=0)
# print(np.sum(hic_matrix[0, : ]))


# t1 = graph_rw_smoothing(hic_matrix, 1)
# t2 = graph_rw_smoothing(hic_matrix, 2)
# t3 = graph_rw_smoothing(hic_matrix, 3)
# t4 = graph_rw_smoothing(hic_matrix, 4)


# plt.matshow(hic_matrix, cmap=REDMAP)
# plt.savefig('t0.png')
# plt.matshow(t1, cmap=REDMAP)
# plt.savefig('t1.png')
# plt.matshow(t2, cmap=REDMAP)
# plt.savefig('t2.png')
# plt.matshow(t3, cmap=REDMAP)
# plt.savefig('t3.png')
# plt.matshow(t4, cmap=REDMAP)
# plt.savefig('t4.png')


# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda:0" if use_cuda else "cpu")


# graphic_model = GraphConvGrapHiC(HYPERPARAMETERS, device, model_name)


# graph_train(graphic_model, 
#             os.path.join(dataset_path, 'train.npz'),
#             os.path.join(dataset_path, 'valid.npz'),
#             model_name,
#             clean_existing_weights=True, debug=True
# )









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


