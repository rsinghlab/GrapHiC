'''
    This file contains the scripts to normalize various data types we have used
    Currently this file pre-dominantly contains HiC normalization scripts.
'''
import numpy as np
from src import visualizations


SAMPLE_HIC_NORMALIZATION_PARAMS = {
    'norm'              : True,  # To normalize or not
    'remove_zeros'      : True,  # Remove zero before percentile calculation
    'set_diagonal_zero' : False, # Remove the diagonal before percentile calculation
    'percentile'        : 95.0,  # Percentile 
    'rescale'           : True,  # After applying cutoff, rescale between 0 and 1
    'chrom_wide'        : True,  # Apply it on chromosome scale #TODO: Sample wise normalization isn't implemented
    'draw_dist_graphs'  : False  # Visualize the distribution of the chromosome
}

def normalize_hic_matrix(hic_matrix, params, cell_line='H1', chromosome='chr1', target=True):
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
    

    if params['percentile'] == -1:
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
    cutoff_value = np.percentile(all_values, params['percentile'])

    hic_matrix = np.minimum(cutoff_value, hic_matrix)
    hic_matrix = np.maximum(hic_matrix, 0)

    

    # Rescale
    if params['rescale']:
        hic_matrix = hic_matrix / (np.max(cutoff_value) + 1)

    hic_matrix[hic_matrix < 0.001] = 0 # Nice magic number here mate

    # Set edges to a particular value
    if params['edge_culling'] != -1 and target == False:
        print('Setting all edges to {}'.format(params['edge_culling']))
        hic_matrix[:] = params['edge_culling']

    return hic_matrix