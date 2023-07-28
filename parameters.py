'''
    This file contains all the parameters we have used for GrapHiC
'''


### These are default parameters, to run interesting experiments change these parameters ###
hic_data_resolution = 10000

# Paths
BASE_DIRECTORY = '/users/gmurtaza/GrapHiC/'
DATA_DIRECTORY = '/users/gmurtaza/data/gmurtaza/'


# These hyperparameters go to GrapHiC model controlling the training batch size, optimizer parameters 
PARAMETERS = {
    'epochs'                                : 250,              # We used 250 for most of the architecture search
    'batch_size'                            : 8,                # Change to control per batch GPU memory requirement
    'optimizer_type'                        : 'ADAM',           # We used the same optimizer ADAM
    'learning_rate'                         : 0.0001,           # Optimzer parameters
    'momentum'                              : 0.9,              # Optimizer parameters
    'num_heads'                             : 4,                # Number of heads a parameter required for TransformerConv layer and GAT layers (if applicable)
    'embedding_size'                        : 32,               # Size of the embeddings, latent dimensions and number of filters
    'resblocks'                             : 5,                # Number of resblocks for Decoder 
    'graphconvblocks'                       : 1,                # Number of Graph convolution layers for Encoder
    'graphconvalgo'                         : 'Transformer',    # Graph convolution function
    'loss_func'                             : 'MSE',
    'decoderstyle'                          : 'Unet',           # Graph decoding algoritm, defaults to ContactCNN
    'laplacian_norm'                        : False,
}


# These parameters are also used by the dataset creator function but this set contains the normalization parameters
dataset_creation_parameters = {
    'norm'                                  : True,             # To normalize or not
    'remove_zeros'                          : True,             # Remove zero before percentile calculation
    'set_diagonal_zero'                     : False,            # Remove the diagonal before percentile calculation
    'percentile'                            : 99.90,            # Percentile
    'edge_culling'                          : -1,               # Set all the hic contacts to a particular value 
    'rescale'                               : True,             # After applying cutoff, rescale between 0 and 1
    'chrom_wide'                            : True,             # Apply it on chromosome scale #TODO: Sample wise normalization isn't implemented
    'draw_dist_graphs'                      : False,            # Visualize the distribution of the chromosome
    'positional_encoding_method'            : 'graph',          # Required for GrapHiC, can take values between ['constant', 'monotonic', 'transformer', 'graph']
    'positional_encoding_dim'               : 8,                # Dimension of positional encodings
    'lap_norm'                              : 'sym',            # If the positional encoding method is graph_lap_pe, we use this method to create the laplacian
    'eig_norm'                              : 'L2',             # If the positional encoding method is graph_lap_pe, we use this method to normalize our eigen vectors
    'non_informative_row_resolution_method' : 'target',         # This finds common non-informative rows in both dataset and then removes them. Can take ['ignore', 'target', 'intersection']
    'noise'                                 : 'none',           # This function adds noise to all the input samples should improve training in certain cases. Can take ['none', 'random', 'uniform', 'gaussian']
    'node_embedding_concat_method'          : 'concat',         # Method to merge the positional and node encodings
    'sub_mat'                               : 256,              # Size of the submatrices  
    'stride'                                : 30,               # Stride after sampling each submatrix
    'bounds'                                : 20,               # Bounds to stay within, 20 ensures that we never sample anything outside 2MB
    'padding'                               : True,             # To pad the matrices with 0 for the last sample
    'add_expected_hic'                      : False,            # If the input matrix structure is weak, we add an expected Hi-C matrix to provide support for our graph model
    'add_expected_hic_merge'                : False,
    'replace_with_expected_hic'             : False
} 


epigenetic_features = {
    # Combined features
    'All': ['RAD-21', 'RNA-Pol2','CTCF', 'DNASE-Seq', 'H3K27ME3', 'H3K27AC', 'H3K36ME3', 'H3K4ME1', 'H3K4ME2', 'H3K4ME3', 'H3K79ME2', 'H3K9AC', 'H4K20ME1', 'H3K9ME3'],
    'DNA-Acessibility': ['RAD-21', 'RNA-Pol2','CTCF', 'DNASE-Seq'],
    'Repression-Marker': ['H3K27ME3', 'H3K4ME2', 'H4K20ME1'],
    'Activating-Marker': ['H3K4ME3', 'H3K9AC', 'H3K9ME3'],
    'Enchancer-Interaction-Marker': ['H3K36ME3', 'H3K79ME2'],
    'Gene-Related': ['H3K36ME3', 'H3K79ME2'],
    'HiC-Reg-Reduced': ['CTCF', 'DNASE-Seq', 'H4K20ME1', 'H3K27ME3', 'H3K9ME3', 'H3K9AC', 'H3K4ME1', 'H3K27AC'],    
    'CAESAR': ['CTCF', 'DNASE-Seq', 'H3K4ME1', 'H3K4ME3', 'H3K27AC', 'H3K27ME3'], 
    'Origami': ['CTCF', 'DNASE-Seq'],
    # Individual features
    'RAD-21': ['RAD-21'],
    'RNA-Pol2': ['RNA-Pol2'],
    'CTCF': ['CTCF'],
    'DNASE-Seq': ['DNASE-Seq'],
    'H3K27ME3': ['H3K27ME3'],
    'H3K27AC': ['H3K27AC'],
    'H3K36ME3': ['H3K36ME3'],
    'H3K4ME1': ['H3K4ME1'],
    'H3K4ME2': ['H3K4ME2'],
    'H3K4ME3': ['H3K4ME3'],
    'H3K79ME2': ['H3K79ME2'],
    'H3K9AC': ['H3K9AC'],
    'H4K20ME1': ['H4K20ME1'],
    'H3K9ME3': ['H3K9ME3'],
    'GrapHiC-Med': ['CTCF', 'RAD-21', 'DNASE-Seq'],
    'GrapHiC-Trimmed': ['CTCF', 'DNASE-Seq', 'H3K4ME3', 'H3K27AC', 'H3K27ME3'],
    'RNA-Seq': ['RNA-Seq+', 'RNA-Seq-'],
    'scGrapHiC': ['RNA-Seq+', 'RNA-Seq-', 'DNASE-Seq'],
} 
