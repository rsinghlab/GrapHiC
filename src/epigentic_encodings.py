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

from src.utils import create_entire_path_directory

# TODO: Implement the required functionality
def read_node_encoding_files(node_encoding_files, compact_idx=[]):
    pass



def parse_node_encoding_file(file_path, output_path, resolution=10000):
    # Currently we are assuming all the files are bigwig files
    bw = pyBigWig.open(file_path)
    if not bw.isBigWig():
        print('Currently only supporting BigWig file formats...')
        return

    create_entire_path_directory(output_path)

    for chrom in bw.chroms().keys():
        output_file_path = os.path.join(output_path, '{}.npz'.format(chrom))
        if os.path.exists(output_file_path):
            print('Already parsed')
            return
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

        
