import os
import cooler
import numpy as np
import pandas as pd

from src.utils import load_hic_file


def create_genomic_bins(
        chromosome_name,
        resolution,
        size
    ):
    
    """
    The only currently supported type is 'bed' format which is chromosome_id, start, end
    So the function requires input of 'chromosome_name' chromosome name and 'resolution' resolution of of the file. 
    This function also requires size of the chromosome to estimate the maximum number of bins
    """
    
    chr_names = np.array([chromosome_name]*size)
    starts = (np.arange(0, size, 1, dtype=int))*resolution
    ends = (np.arange(1, size+1, 1, dtype=int))*resolution
    
    bins = {
        'chrom': chr_names,
        'start': starts,
        'end': ends
    }

    bins = pd.DataFrame(data=bins)

    return bins



def create_genomic_pixels(dense_matrix):
    """
        Converts a dense matrix into a .bed style sparse matrix file
        @params: dense_matrix <np.array>, input dense matrix
        @params: output_type <string>, output type, currently only supported style is bed style
    """
    
    lower_triangular_matrix_coordinates = np.tril_indices(dense_matrix.shape[0], k=-1)
    dense_matrix[lower_triangular_matrix_coordinates] = 0
    
    non_zero_indexes = np.nonzero(dense_matrix)
    bin_ones = non_zero_indexes[0]
    bin_twos = non_zero_indexes[1]
    counts = dense_matrix[np.nonzero(dense_matrix)]
    
    pixels = {
        'bin1_id': bin_ones,
        'bin2_id': bin_twos,
        'count': counts
    }

    pixels = pd.DataFrame(data=pixels)

    return pixels

def balance_cooler_file(cooler_file_path):
    """
        This function uses cooler utility to balance a .cool matrix
        
        @params: cooler_file_path <string>, path to the cooler file to balance
    """
    cmd = 'cooler balance {}'.format(cooler_file_path)
    
    os.system(cmd)



def create_cool_file_from_numpy(
    numpy_file_path,
    output_file_path,
    chromosome_name,
    resolution = 10000,
    upscale=255,
    ):

    '''
        @params: <string> numpy_file_path file path of the numpy array that needs to be converted to cooler file
        @params: <string> output_file_path full path to the output folder that contains the output file
        @chromosome_name: <string> name of the chromosome for example: chr21 
        @resolution: <int> resolution at which we have sampled the input dense array
    '''
    print('Generating Cooler file')
    
    # loading the file, we always assume correct full input file path
    
    dense_data = load_hic_file(numpy_file_path)['hic']*upscale

    h, w = dense_data.shape

    dense_hic_file_genomic_bins = create_genomic_bins(chromosome_name, resolution, h)
    dense_hic_file_pixels_in_bins = create_genomic_pixels(dense_data)
    
    # This generates a cooler file in the provided output file path
    cooler.create_cooler(output_file_path, dense_hic_file_genomic_bins, dense_hic_file_pixels_in_bins,
                        dtypes={"count":"int"}, 
                        assembly="hg19")
    
    