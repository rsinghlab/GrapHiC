'''
    This file contains all the parameters we have used for GrapHiC
'''


### These are default parameters, to run interesting experiments change these parameters ###
hic_data_resolution = 10000



# These hyperparameters go to GrapHiC model controlling the training batch size, optimizer parameters 
PARAMETERS = {
    'epochs'                                : 250,             # We used 250 for most of the architecture search
    'batch_size'                            : 16,               # Change to control per batch GPU memory requirement
    'optimizer_type'                        : 'ADAM',           # We used the same optimizer ADAM
    'learning_rate'                         : 0.0001,               # Optimzer parameters
    'momentum'                              : 0.9,                  # Optimizer parameters
    'num_heads'                             : 4,                # Number of heads a parameter required for TransformerConv layer and GAT layers (if applicable)
    'embedding_size'                        : 32,               # Size of the embeddings, latent dimensions and number of filters
    'resblocks'                             : 5,                # Number of resblocks for Decoder 
    'transformblocks'                       : 1,                # Number of Transformer layers for Encoder
}


# These parameters are also used by the dataset creator function but this set contains the normalization parameters
dataset_creation_parameters = {
    'norm'                                  : True,             # To normalize or not
    'remove_zeros'                          : True,             # Remove zero before percentile calculation
    'set_diagonal_zero'                     : False,            # Remove the diagonal before percentile calculation
    'percentile'                            : 99.75,            # Percentile
    'edge_culling'                          : -1,               # Set all the hic contacts to a particular value 
    'rescale'                               : True,             # After applying cutoff, rescale between 0 and 1
    'chrom_wide'                            : True,             # Apply it on chromosome scale #TODO: Sample wise normalization isn't implemented
    'draw_dist_graphs'                      : False,            # Visualize the distribution of the chromosome
    'positional_encoding_method'            : 'graph',          # Required for GrapHiC, can take values between ['constant', 'monotonic', 'transformer', 'graph']
    'positional_encoding_dim'               : 13,               # Dimension of positional encodings
    'non_informative_row_resolution_method' : 'intersection',   # This finds common non-informative rows in both dataset and then removes them. Can take ['ignore', 'target', 'intersection']
    'noise'                                 : 'none',           # This function adds noise to all the input samples should improve training in certain cases. Can take ['none', 'random', 'uniform', 'gaussian']
    'node_embedding_concat_method'          : 'concat',         # Method to merge the positional and node encodings
    'sub_mat'                               : 200,              # Size of the submatrices  
    'stride'                                : 50,               # Stride after sampling each submatrix
    'bounds'                                : 40,               # Bounds to stay within, 40 ensures that we never sample anything outside 2MB
    'padding'                               : True              # To pad the matrices with 0 for the last sample
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
    'None': []
} 
