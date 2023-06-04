from operator import pos
import os
import numpy as np
import time
import multiprocessing


from itertools import repeat
from multiprocessing import Pool

from src.matrix_operations import divide
from src.matrix_operations import compactM

from src.utils import load_hic_file
from src.utils import create_entire_path_directory

from src.epigentic_encodings import read_node_encoding_files
from src.generate_expected_hic import generate_and_add_expected_contact_matrix, generate_expected_contact_matrix
from src.normalizations import normalize_hic_matrix
from src.positional_encodings import encoding_methods
from src.noise import noise_types

MULTIPROCESSING = True


dataset_partitions = {
    'train': [1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15, 16, 17, 18],
    'valid': [8, 19, 10, 22],
    'test': [20, 21, 11],
    'debug_train': [17],
    'debug_test': [20], 
    'all': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
}




def process_compact_idxs(base_compact_idx, target_compact_idx, compact_method='intersection'):
    '''
        Processes the compact indexes based on the method type provided.
        @params: base_compact_idx <List>, Compaction indices of the base matrix
        @params: target_compact_idx <List>, Compaction indices of the target matrix
        @params: compact_method <string>, Compaction strategy, intersection is the default strategy  
        @returns: <List> Compact indices
    '''
    if compact_method == 'ignore':
        compact_indexes = []
            
    elif compact_method == 'intersection':
        compact_indexes = list(set.intersection(set(base_compact_idx), set(target_compact_idx)))
       
    elif compact_method == 'target':
        compact_indexes = target_compact_idx
        
    else:
        print("Invalid value for variable compact_type, please choose one of [ignore, intersection, target].")
        exit(1)

    return compact_indexes

def merge_encodings(node_features, pos_encoding, merging_type):
    # Currently just returns the positional encodings
    print('NODE FEATURES:',np.min(node_features), np.max(node_features), node_features.shape)
    print('POS FEATURES:',np.min(pos_encoding), np.max(pos_encoding), pos_encoding.shape)
    
    if merging_type == 'sum':
        return node_features + pos_encoding
    elif merging_type == 'mean':
        return (node_features + pos_encoding)/2
    elif merging_type == 'concat':
        return np.concatenate((node_features, pos_encoding), axis=2)
    elif merging_type == 'positional':
        return pos_encoding
    elif merging_type == 'epigenetic':
        return node_features  
    else:
        print('Invalid merging operator defined, Aborting')
        exit(1)

def process_chromosome_files(args):
    return _process_chromosome_files(*args)

def _process_chromosome_files(
        path_to_base_chrom, 
        path_to_target_chrom,
        node_encoding_files,
        PARAMETERS,
        verbose
    ):
    chromosome = int(path_to_base_chrom.split('/')[-1].split('.')[0].split('chr')[1])

    if verbose: print('Processing chromosome {}'.format(chromosome))
    # Read chromosome files
    try:
        base_data = load_hic_file(path_to_base_chrom)
        target_data = load_hic_file(path_to_target_chrom)
    except FileNotFoundError:
        print(path_to_base_chrom, path_to_target_chrom)
        
        print('Skipping chromosome {} because file is missing'.format(chromosome))
        return ()

    base_chrom = base_data['hic']
    target_chrom = target_data['hic']
    
    #print(base_chrom.shape, target_chrom.shape)
    
    # Ensure that that shapes match, otherwise we can not proceede
    assert(base_chrom.shape[0] == target_chrom.shape[0])
    full_size = base_chrom.shape[0]
    
    # Find the informative rows and remove the rest from the data
    if verbose: print('Removing non-infomative rows and columns')
    base_compact_idx = base_data['compact']
    target_compact_idx = target_data['compact'] 
    
    if len(base_compact_idx) == 0 or len(target_compact_idx) == 0:
        print('Chromosome file {} does not contain valid data'.format(base_compact_idx))
        return ()
        

    compact_indexes = process_compact_idxs(
        base_compact_idx, 
        target_compact_idx, 
        compact_method=PARAMETERS['non_informative_row_resolution_method']
    )

    base_chrom = compactM(base_chrom, compact_indexes)
    target_chrom = compactM(target_chrom, compact_indexes)
    
    #print(base_chrom.shape, target_chrom.shape)
          
    if verbose: print('Processing Node Features')
    # Read Node feature files and compile them in a single array
    node_features, _, order = read_node_encoding_files(
        node_encoding_files, 
        chromosome, 
        PARAMETERS, 
        compact_indexes
    )
    print(node_features.shape)
    
    
    # Normalize the HiC Matrices
    if PARAMETERS['chrom_wide']:
        if verbose: print('Performing Chromosome wide normalization...')
        base_chrom = normalize_hic_matrix(base_chrom, 
            PARAMETERS, 
            chromosome=chromosome, 
            target=False
        )
        target_chrom = normalize_hic_matrix(
            target_chrom, 
            PARAMETERS, 
            chromosome=chromosome, 
            target=True
        )

    if PARAMETERS['noise']:
        if verbose: print('Adding {} noise to the data'.format(PARAMETERS['noise']))
        base_chrom = noise_types[PARAMETERS['noise']](base_chrom)
        
    # Divide the HiC Matrices
    if verbose: print('Dividing the Chromosomes...')
    bases, _ = divide(base_chrom, chromosome, PARAMETERS)        
    targets, inds = divide(target_chrom, chromosome, PARAMETERS)
    
    if PARAMETERS['add_expected_hic']:
        bases = np.array(list(map(lambda x: generate_and_add_expected_contact_matrix(x), bases)))
    
    if PARAMETERS['replace_with_expected_hic']:
        bases = np.array(list(map(lambda x: generate_expected_contact_matrix(x), bases)))
    
    if PARAMETERS['node_embedding_concat_method'] in ['concat', 'positional']:
        encoding_dim = PARAMETERS['positional_encoding_dim']
    else:
        encoding_dim = node_features.shape[2]

    
    if verbose: print('Generating Positional Encodings')
    # 3 out of 4 positional encodings are independent of the graph except the 'graph' encodings
    pos_encodings = []
    for base in bases:
        if PARAMETERS['positional_encoding_method'] in ['constant', 'monotonic', 'transformer', 'graph']:
            pos_encoding = encoding_methods[PARAMETERS['positional_encoding_method']](
                base, encoding_dim=encoding_dim
            )
            pos_encodings.append(pos_encoding)
        elif PARAMETERS['positional_encoding_method'] == 'graph_lap_pe':
            pos_encoding = encoding_methods[PARAMETERS['positional_encoding_method']](
                base, encoding_dim=encoding_dim, 
                lap_norm=PARAMETERS['lap_norm'],
                eig_norm=PARAMETERS['eig_norm']
            )
            pos_encodings.append(pos_encoding)
       
        elif PARAMETERS['positional_encoding_method'] == 'rw_se':
            pos_encoding = encoding_methods[PARAMETERS['positional_encoding_method']](
                base, encoding_dim=encoding_dim, 
            )
            pos_encodings.append(pos_encoding)
        
        elif PARAMETERS['positional_encoding_method'] == 'heat_kernel_se':
            pos_encoding = encoding_methods[PARAMETERS['positional_encoding_method']](
                base, encoding_dim=encoding_dim, 
            )
            pos_encodings.append(pos_encoding)

        else:
            print('Invalid PE operator')
            exit(1)
    
    
    
    pos_encodings = np.array(pos_encodings)
    order = ['pe']*pos_encodings.shape[-1] + order
    
    encodings = merge_encodings(
        node_features, 
        pos_encodings, 
        merging_type=PARAMETERS['node_embedding_concat_method']
    )
    
    return (
            chromosome, 
            bases, 
            targets,
            inds,
            encodings,
            compact_indexes,
            full_size,
            order
        )


def create_dataset_from_hic_files(
                                path_to_base_files,
                                path_to_target_files, 
                                path_to_output_folder,
                                node_encoding_files,
                                PARAMETERS,
                                datasets=[
                                'train', 
                                'valid', 
                                'test'
                                ],
                                verbose=True
    ):
    '''
        This function takes in path to the base chromospositional_encoding_methodte path where to store the generated dataset file
        @params: positional_encoding_method <string>, what positional encoding method to use
        @params: nodeargs_encoding_files <string> path to the node encoding files, these files contain the epigenetic signals
        @params: cropping_params <dict> contains the cropping parameters necessary for cropping huge chromosome files
        @params: normalization_params <dict> normalization parameters defining how to normalize the Hi-C matrix
        @params: compact <string>, what matrix compaction method to use to remove rows (and columns) that are non informative. Defaults 'intersection'. 
        @params: dataset <list>, what dataset to construct 
        @returns: None
    '''

    create_entire_path_directory(path_to_output_folder)

    for dataset in datasets:
        start_time = time.time()
        if verbose: print('Creating {} dataset'.format(dataset))

        chromosomes = dataset_partitions[dataset]
        output_file = os.path.join(path_to_output_folder, '{}.npz'.format(dataset))
        
        base_chrom_paths = list(
            map(
                lambda x: os.path.join(path_to_base_files, 'chr{}.npz'.format(x)),
                chromosomes
        ))
        target_chrom_paths = list(
            map(
                lambda x: os.path.join(path_to_target_files, 'chr{}.npz'.format(x)),
                chromosomes
        ))
        
        args = zip(
            base_chrom_paths, 
            target_chrom_paths,
            repeat(node_encoding_files),
            repeat(PARAMETERS),
            repeat(verbose),
        )
        num_cpus = multiprocessing.cpu_count() if MULTIPROCESSING else 1
        
        with Pool(num_cpus) as pool:
            results = pool.map(process_chromosome_files, args)
        
        
        results = list(filter(lambda x: len(x) != 0, results))
            

        data = np.concatenate([r[1] for r in results])
        target = np.concatenate([r[2] for r in results])
        inds = np.concatenate([r[3] for r in results])
        encodings = np.concatenate([r[4] for r in results])
        
        compacts = {r[0]: r[5] for r in results}
        sizes = {r[0]: r[6] for r in results}
        orders = {r[0]: r[7] for r in results}
         
        print('Saving file:', output_file)
        np.savez_compressed(output_file, data=data, target=target, inds=inds, encodings=encodings, compacts=compacts, sizes=sizes, enc_order=orders)
        end_time = time.time()

        print('Creating {} dataset took {} seconds!'.format(dataset, end_time-start_time))

