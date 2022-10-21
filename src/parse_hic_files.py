'''
    This file contains the scripts that handle parsing the HiC files
    Currently I am only supporting, '.hic format'
'''

from concurrent.futures import process
import hicstraw
import os
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, vstack, hstack
from scipy import sparse


import time
from functools import partial
import multiprocessing
from multiprocessing import Process
from src.utils import create_entire_path_directory, hic_file_paths, PARSED_HIC_FILES_DIRECTORY, hic_data_resolution, download_file



# Main Multiprocessing switch for the HiC parser
MULTIPROCESSING=True

def process_chromosome(hic, output, resolution, chromosome, debug):
    '''
        This function handles the bulk of the work of extraction of Chromosome from
        the HiC file and storing it in its dense 2D contact map form
        @params: hic <hicstraw.HiCFile>, HiC file object as returned by the hicstraw utility
        @params: output <os.path>, path where to store the output files
        @params: resolution <int>, resolution to sample the HiC data at
        @params: chromosome <hicstraw.chromosome>, hicstraw chromosome objects that contains its name and misc properties
        @returns: None
    '''
    index = chromosome.index
    length = chromosome.length
    name = chromosome.name
    output_path = os.path.join(
        output, 
        'resolution_{}'.format(resolution)
    )
    create_entire_path_directory(output_path)
    
    output_path = os.path.join(
        output_path, 
        'chr{}.npz'.format(name)
    )

    if name in ['Y','MT']:
        return 

    if os.path.exists(output_path):
        if debug: print('Already parsed!')
        return

    if debug: print('Starting parsing Chromosome {}'.format(name))

    
    chromosome_matrix = hic.getMatrixZoomData(
        chromosome.name, chromosome.name, 
        'observed', 'KR', 'BP', resolution                                          
    )
    
    informative_indexes = np.array(chromosome_matrix.getNormVector(index))
    informative_indexes = np.where(np.isnan(informative_indexes)^True)[0]
    print(informative_indexes.shape)


    if len(informative_indexes) == 0:
        print('Chromosome {} doesnt contain any informative rows'.format(name))
        return

    results = chromosome_matrix.getRecords(0, length, 0, length)
    
    # Bottleneck step
    results = np.array([[(r.binX//resolution), (r.binY//resolution), r.counts] for r in results])

    N = length//resolution
    mat = csr_matrix((results[:, 2], (results[:, 0], results[:, 1])), shape=(N,N))
    mat = csr_matrix.todense(mat)
    mat = mat.T
    mat = mat + np.tril(mat, -1).T

    np.savez_compressed(output_path, hic=mat, compact=informative_indexes, size=length)
    if debug: print('Saving Chromosome at path {}'.format(output_path))
    return True

    
    
def parse_hic_file(path_to_hic_file, output, resolution=10000, debug=False):
    '''
        This function provides a wrapper on all the methods that 
        reads the .hic file and stores individual chromosomes in a 
        dense matrix format at provided location
        @params: path_to_hic_file <os.path>, path to where hic file is stored
        @params: output_directory <os.path>, path where to store the generated chromsomes
        @params: resolution <int>, resolution at which we sample the HiC contacts, defaults to 10000
        @returns: None
    '''
    print('Parsing out intra-chromosomal contact matrices from {} file.'.format(path_to_hic_file))
    # Read the hic file into memory
    hic = hicstraw.HiCFile(path_to_hic_file)
    
    if resolution not in hic.getResolutions():
        print('Resolution not supported by the provided .hic file, try a resolution from list {}'.format(
            hic.getResolutions()
        ))
        exit(1)
    
    # Get all the available chromosomes
    chromosomes = hic.getChromosomes()[1:]
    start_time = time.time()

    if MULTIPROCESSING:
        process_pool = []

        for idx in range(len(chromosomes)):
            p = Process(target=process_chromosome, args=(hic, output, resolution, chromosomes[idx], debug ))
            process_pool.append(p)
            p.start()
        
        for process in process_pool:
            process.join()
    else:
        for idx in range(len(chromosomes)):
            process_chromosome(hic, output, resolution, chromosomes[idx])


    end_time = time.time()
    
    print('Parsing took {} seconds!'.format(end_time - start_time))











def download_all_hic_datasets():
    for hic_file_key in hic_file_paths.keys():
        if not os.path.exists(hic_file_paths[hic_file_key]['local_path']):
            download_file(hic_file_paths[hic_file_key])
        
        parse_hic_file(
            hic_file_paths[hic_file_key]['local_path'], 
            os.path.join(PARSED_HIC_FILES_DIRECTORY, hic_file_key),
            hic_data_resolution
        )
















