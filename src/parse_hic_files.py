'''
    This file contains the scripts that handle parsing the HiC files
    Currently I am only supporting, '.hic format'
'''

import hicstraw
import os
import numpy as np
import time
from multiprocessing import Process

MULTIPROCESSING=False

def process_chromosome(hic, output, resolution, chromosome):
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
    print('Starting parsing Chromosome {}'.format(name))
    
    
    chromosome_matrix = hic.getMatrixZoomData(
        chromosome.name, chromosome.name, 
        'observed', 'KR', 'BP', resolution                                          
    )
    informative_indexes = np.array(chromosome_matrix.getNormVector(index))
    informative_indexes = informative_indexes[~np.isnan(informative_indexes)]
    
    matrix = np.array(chromosome_matrix.getRecordsAsMatrix(0, length, 0, length))
    
    output_path = os.path.join(
        output, 
        'resolution_{}'.format(resolution),
        'chr{}.npz'.format(name)
    )
    
    print('Saving Chromosome at path {}'.format(output_path))
    np.savez_compressed(output_path, hic=matrix, compact=informative_indexes, size=length)
    

    
    
def parse_hic_file(path_to_hic_file, output, resolution=10000):
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
    chromosomes = hic.getChromosomes()[20:]
    procs = [] 
    
    start_time = time.time()
    
    if MULTIPROCESSING:    
        for idx in range(len(chromosomes)):
            proc = Process(target=process_chromosome, args=(hic, output, resolution, chromosomes[idx], ))
            procs.append(proc)
            proc.start()
        
        for proc in procs: 
            proc.join()
    
    else:
        for idx in range(len(chromosomes)):
            process_chromosome(hic, output, resolution, chromosomes[idx])
    
    end_time = time.time()
    
    print('Parsing took {} seconds!'.format(end_time - start_time))





























