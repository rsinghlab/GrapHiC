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
from src.normalizations import normalize_hic_matrix
from src.positional_encodings import encoding_methods
from src.noise import noise_types

MULTIPROCESSING = True


dataset_partitions = {
    'train': [1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15, 16, 17, 18],
    'valid': [8, 9, 10, 11],
    'test': [19, 20, 21, 22],
    'debug': [19],
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

def merge_encodings(node_features, pos_encoding):
    # Currently just returns the positional encodings
    return pos_encoding


def process_chromosome_files(args):
    return _process_chromosome_files(*args)

def _process_chromosome_files(
        path_to_base_chrom, 
        path_to_target_chrom,
        positional_encoding_method,
        node_encoding_files,
        normalization_params,
        cropping_params,
        compact,
        noise,
        verbose
    ):
    chromosome = path_to_base_chrom.split('/')[-1].split('.')[0]

    # Read chromosome files
    try:
        base_data = load_hic_file(path_to_base_chrom)
        target_data = load_hic_file(path_to_target_chrom)
    except FileNotFoundError:
        print('Skipping chromosome {} because file is missing'.format(chromosome))
        return ()

    base_chrom = base_data['hic']
    target_chrom = target_data['hic']
    
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
        

    compact_indexes = process_compact_idxs(base_compact_idx, target_compact_idx, compact_method=compact)
    base_chrom = compactM(base_chrom, compact_indexes)
    target_chrom = compactM(target_chrom, compact_indexes)
                 
    if verbose: print('Processing Node Features')
    # Read Node feature files and compile them in a single array
    node_features = read_node_encoding_files(node_encoding_files, compact_indexes)

    # Normalize the HiC Matrices
    if normalization_params['chrom_wide']:
        if verbose: print('Performing Chromosome wide normalization...')
        base_chrom = normalize_hic_matrix(base_chrom, normalization_params, chromosome=chromosome)
        target_chrom = normalize_hic_matrix(target_chrom, normalization_params, chromosome=chromosome)

    if noise:
        if verbose: print('Adding {} noise to the data')
        base_chrom = noise_types[noise](base_chrom)
        


    # Divide the HiC Matrices
    if verbose: print('Dividing the Chromosomes...')
    bases, _ = divide(base_chrom, chromosome, cropping_params)        
    targets, inds = divide(target_chrom, chromosome, cropping_params)
    
    if verbose: print('Generating Positional Encodings')
    # 3 out of 4 positional encodings are independent of the graph except the 'graph' encodings
    pos_encodings = []
    for base in bases:
        pos_encoding = encoding_methods[positional_encoding_method](base)
        pos_encodings.append(pos_encoding)

    encodings = merge_encodings(node_features, pos_encoding)

    return (
            chromosome, 
            bases, 
            targets,
            inds,
            encodings,
            compact_indexes,
            full_size,
        )


def create_dataset_from_hic_files(
                                path_to_base_files,
                                path_to_target_files, 
                                path_to_output_folder,
                                positional_encoding_method, 
                                node_encoding_files,
                                cropping_params={
                                'chunk_size':200,
                                'stride'    :200,
                                'bounds'    :190,
                                'padding'   :True
                                },
                                normalization_params={
                                'norm'              : True,
                                'remove_zeros'      : True,
                                'set_diagonal_zero' : False,
                                'cutoff'            : 95.0,
                                'rescale'           : True,
                                'draw_dist_graphs'  : False
                                },
                                noise='none',
                                compact='intersection',
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
            repeat(positional_encoding_method),
            repeat(node_encoding_files),
            repeat(normalization_params),
            repeat(cropping_params),
            repeat(compact),
            repeat(noise),
            repeat(verbose),
        )
        num_cpus = multiprocessing.cpu_count() if MULTIPROCESSING else 1
        
        with Pool(num_cpus) as pool:
            results = pool.map(process_chromosome_files, args)
        
        
        results = list(filter(lambda x: len(x) != 0, results))
        
        # else:
        #     for chromosome in chromosomes:
        #         if verbose: print('Working with chromosome {} files...'.format(chromosome))
        #         base_chrom_path = os.path.join(path_to_base_files, 'chr{}.npz'.format(chromosome))
        #         target_chrom_path = os.path.join(path_to_target_files, 'chr{}.npz'.format(chromosome))
        #         results.append(process_chromosome_files(
        #             base_chrom_path, 
        #             target_chrom_path,
        #             positional_encoding_method, 
        #             node_encoding_files, 
        #             normalization_params, 
        #             cropping_params, 
        #             compact, 
        #             noise, 
        #             verbose)
        #         )
                
            
        #     # Filter out empty tuples
        #     results = list(filter(lambda x: len(x) != 0, results))            

        data = np.concatenate([r[1] for r in results])
        target = np.concatenate([r[2] for r in results])
        inds = np.concatenate([r[3] for r in results])
        encodings = np.concatenate([r[4] for r in results])
        
        compacts = {r[0].split('chr')[0]: r[5] for r in results}
        sizes = {r[0].split('chr')[0]: r[6] for r in results}

        
        print('Saving file:', output_file)
        np.savez_compressed(output_file, data=data, target=target, inds=inds, encodings=encodings, compacts=compacts, sizes=sizes)
        end_time = time.time()

        print('Creating {} dataset took {} seconds!'.format(dataset, end_time-start_time))


