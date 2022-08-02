'''
    This file contains the scripts to normalize various data types we have used
    Currently this file pre-dominantly contains HiC normalization scripts.
'''
import numpy as np
from src import visualizations


SAMPLE_HIC_NORMALIZATION_PARAMS = {
    'norm'              : True,  # This controls if we want to apply normalization or not
    'remove_zeros'      : True,  # Remove zeros before the percentile computation
    'set_diagonal_zero' : False, # Set the diagonals to zero before percentile computation
    'cutoff'            : 95.0,  # What percentile to use for cutoff
    'rescale'           : True,  # Rescale the clipped matrix between 0-1
    'chrom_wide'        : True,  # Apply the normalization chromosome-wide or sample wide, where sample is the submatrix from the chromosomes
    'draw_dist_graphs'  : True   # A visualiation handle, to visualize the distribution of the HiC matrix
}


def normalize_hic_matrix(hic_matrix, params, cell_line='H1', chromosome='chr1'):
    '''
        This fuction performs chromosome wide normalization of the HiC matrices 
        @params: hic_matrix <np.array>, 2D array that contains all the intra-chromosomal contacts
        @params: params <dict>, A dictionary that contains all the required parameters to perform the normalization
        @returns: <np.array> A normalized HiC matrix
    '''
    # Do not perform any normalization (Not Recommended)
    if not params['norm']:
        return hic_matrix

    # Set diagonal zero 
    if params['set_diagonal_zero']:
        np.fill_diagonal(hic_matrix, 0)
    

    if params['cutoff'] == -1:
        return hic_matrix


    # Get the value distribution in a flattened matrix
    all_values = hic_matrix.flatten()
    
    # Remove zeros
    if params['remove_zeros']:
        all_values = all_values[all_values>0]
        


    # Draw distribution graphs for visualizations
    if params['draw_dist_graphs']:
        name_of_graph = 'c-{}:{}_sdz-{}_rz-{}_precentiles-vs-contacts.png'.format(
            cell_line, chromosome, params['set_diagonal_zero'], params['remove_zeros']
        )
        visualizations.plot_distribution_with_precentiles(all_values, name_of_graph)
        

    # Compute and apply cutoff
    cutoff_value = np.percentile(all_values, params['cutoff'])

    hic_matrix = np.minimum(cutoff_value, hic_matrix)
    hic_matrix = np.maximum(hic_matrix, 0)

    # Rescale
    if params['rescale']:
        hic_matrix = hic_matrix / (np.max(cutoff_value) + 1)

    return hic_matrix