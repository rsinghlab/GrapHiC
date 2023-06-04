'''
    This file contains all the scripts used to read the node encoding files
    Implementation is under progress: 
    Will add support to read:
        1) ATAC signal,
        2) DNAse signal 
        4) H3K4Me1
        5) H3K4Me3
        6) H3K27ac
        7) H3K27Me3
        
    In its fully implemented form we should have a wrapper function that reads all 
    the provided epigentic signal files and read them and return them in a form that is agreeable 
    with the node encoding representation
'''
import os
import numpy as np
import pyBigWig
import time
from src.parse_hic_files import download_file
from src.utils import create_entire_path_directory, epigenetic_factor_paths, hic_data_resolution, PARSED_EPIGENETIC_FILES_DIRECTORY, download_file


def parse_node_encoding_file(file_path, output_path, resolution=10000, debug=False):
    # Currently we are assuming all the files are bigwig files
    print(file_path)
    
    bw = pyBigWig.open(file_path)
    if not bw.isBigWig():
        print('Currently only supporting BigWig file formats...')
        return
    start_time = time.time()

    create_entire_path_directory(output_path)

    for chrom in bw.chroms().keys():
        output_file_path = os.path.join(output_path, '{}.npz'.format(chrom))
        if os.path.exists(output_file_path):
            if debug: print('Already parsed')
            continue

        chrom_length = bw.chroms()[chrom]
        nBins = chrom_length // resolution # Lower bound?
        
        chrom_bins = np.array(bw.stats(
            chrom, 
            0, chrom_length,
            nBins = nBins, type='mean'
        ))
        chrom_bins[chrom_bins == None] = 0
        chrom_bins.astype(float)

        np.savez_compressed(output_file_path, epi=chrom_bins, resolution=resolution, size=chrom_length)
        if debug: print('Saving parsed node encoding file at: {}'.format(output_file_path))
    end_time = time.time()
    print('Parsing all files took {} seconds!'.format(end_time - start_time))


def read_node_encoding_files(node_encoding_files, chromosome, cropping_params, compact_idx=[], divided=True):
    node_encodings = []
    node_encodings_order = []
    
    for node_encoding_file in node_encoding_files:
        print(node_encoding_file)

        # We get the top level directory path, and each chromosome is stored as a 
        # separate file
        node_encoding_file_path = os.path.join(
            node_encoding_file,
            'chr{}.npz'.format(chromosome)
        )

        # Read the npz file 
        node_encoding_data = np.load(node_encoding_file_path, allow_pickle=True)
        node_encoding_data = node_encoding_data['epi']
        
        node_encodings_order.append(node_encoding_file.split('/')[-1])

        print(node_encoding_data.shape, len(compact_idx))
        
        # Only take the node encodings that are informative in HiC data as well
        if len(compact_idx) != 0:
            node_encoding_data = node_encoding_data.take(compact_idx)
        
        
        # Append the encodings
        node_encodings.append(node_encoding_data)
    
    
    node_encodings = np.array(node_encodings)
    node_encodings = node_encodings.T
    
    node_encodings = normalize_epigenetic_encodings(node_encodings)


    # Control flow for the hicreg parser
    if not divided:
        return node_encodings, []


    divided_signal, idxs = divide_signal(node_encodings, chromosome, cropping_params)
    
    divided_signal = divided_signal[:, 0, :, :]

    # Return in numpy.array format
    return divided_signal, idxs, node_encodings_order








def divide_signal(encodings, chr, cropping_params):
    result = []
    index = []

    stride = cropping_params['stride']
    chunk_size = cropping_params['sub_mat']
    padding = cropping_params['padding']

    if (stride < chunk_size and padding):
        pad_len = (chunk_size - stride) // 2
        encodings = np.pad(encodings, ((pad_len,pad_len), (0, 0)), 'constant')
    
    size = encodings.shape[0]

    # mat's shape changed, update!
    for i in range(0, size, stride):
        if (i+chunk_size)<size:
            subImage = encodings[i:i+chunk_size, :]
            result.append([subImage])
            index.append((int(chr), int(size), int(i)))
    
    result = np.array(result, dtype=float)
    index = np.array(index, dtype=int)

    return result, index


def normalize_epigenetic_encodings(encodings):
    percentile = np.percentile(encodings, 99)
    encodings = np.minimum(percentile, encodings)
    encodings = np.maximum(encodings, 0)
    encodings = encodings / (np.max(encodings) + 1)
    
    
    return encodings



def download_all_epigenetic_datasets():
    for cell_line in epigenetic_factor_paths.keys():
        for histone_mark in epigenetic_factor_paths[cell_line].keys():
            if not os.path.exists(epigenetic_factor_paths[cell_line][histone_mark]['local_path']):
                download_file(epigenetic_factor_paths[cell_line][histone_mark])
            parse_node_encoding_file(
                epigenetic_factor_paths[cell_line][histone_mark]['local_path'],
                os.path.join(PARSED_EPIGENETIC_FILES_DIRECTORY, cell_line, histone_mark),
                hic_data_resolution
            )
            