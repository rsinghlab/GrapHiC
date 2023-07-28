import sys
sys.path.append('../GrapHiC/')


import os
import json
import numpy as np
from parameters import *
from src.dataset_creator import dataset_partitions
from src.matrix_operations import divide, compactM
from src.utils import create_entire_path_directory
from src.utils import PREDICTED_FILES_DIRECTORY, GENERATED_RESULTS_DIRECTORY
from evaluation.visualize_samples import visualize_samples
from evaluation.correlation_analysis import compute_correlation_metrics
from evaluation.recon_analysis import generate_models, reconstruction_score
from evaluation.bio_feature_analysis import extract_features, compare_features
from evaluation.hic_similarity_analysis import create_required_analysis_files, create_metadata_files_for_hic_similarity_analysis


rp_values = {
    'loops': 3,
    'borders': 3,
    'hairpins': 3,
    
}

#relevant_directories = [sys.argv[1]]


# relevant_directories = list(filter(lambda x: '@' in x, os.listdir(PREDICTED_FILES_DIRECTORY)))
relevant_directories = ['GM12878-encode-0@GM12878-geo-raoetal']
relevant_directories = list(map(lambda x: os.path.join(PREDICTED_FILES_DIRECTORY, x) , relevant_directories))

print(relevant_directories)



test_chroms = dataset_partitions['test']

def read_and_crop_chrom_files(path, cpt_idx_dict):
    predicted = []
    inds = []
    
    for chrom in test_chroms:
        chrom_path = os.path.join(path, 'chr{}.npz'.format(chrom))
        predicted_chrom_data = np.load(chrom_path)
        try:
            compact_idxs = cpt_idx_dict[chrom] 
            
        except:
            compact_idxs = predicted_chrom_data['compact']
            cpt_idx_dict[chrom] = compact_idxs
        
        predicted_chrom_data = compactM(predicted_chrom_data['hic'], compact_idxs)
        divided_chrom, ind = divide(predicted_chrom_data, chrom, dataset_creation_parameters)
        predicted.append(divided_chrom)
        inds.append(ind)


    predicted = np.concatenate(predicted, axis=0)    
    inds = np.concatenate(inds, axis=0)

    return predicted, inds, cpt_idx_dict



# Testing function to reduce the number of samples and check end-to-end performance
def get_every_nth_element(data, step=10):
    num = data.shape[0]//step
    array = data[np.round(np.linspace(1, data.shape[0]-1, num=num)).astype(int)]
    
    return array



def save_results(results, path):
    with open(path, 'w') as f:
        f.write(json.dumps(results))


for relevant_directory in relevant_directories:
    methods = os.listdir(relevant_directory)
    inputs = np.array([])
    targets = np.array([])
    indexes = np.array([])
    enc_order = None
    input_encodings = np.array([])
    compact_idxs = {}
    
    
    samples_dictionary = {}
    recon_analysis_files = {}
    bio_feature_analysis_files = {}
    hic_similarity_analysis_files = {}
    
    
    base_cell_line = relevant_directory.split('/')[-1].split('@')[0]
    target_cell_line = relevant_directory.split('/')[-1].split('@')[1]
    
    # This stupid trick is necessary to make sure the data is processed in correct order
    methods.sort(reverse=False)
    methods = methods[1:] + [methods[0]]
    
    # Crawl out all the samples from the generated data
    for method in methods:
        if method == 'inputs' or method == 'target' or method=='hic_similarity_analysis':
            continue

        predicted_data_directory = os.path.join(relevant_directory, method)
        print('Reading samples from {}'.format(predicted_data_directory))       
        if 'graphic' in predicted_data_directory:
            predicted_samples = np.load(os.path.join(predicted_data_directory, 'predicted_samples.npz'), allow_pickle=True)
            if inputs.shape[0] == 0:
                inputs = predicted_samples['input']
            if targets.shape[0] == 0:
                targets = predicted_samples['target']
            if not enc_order:
                enc_order = predicted_samples['enc_order']
            if input_encodings.shape[0] == 0:
                input_encodings = predicted_samples['input_encodings']
            if indexes.shape[0] == 0:
                indexes = predicted_samples['index']
            if method not in samples_dictionary.keys():
                samples_dictionary[method] = predicted_samples['graphic']
                
        else:
            predicted, idxes, cmpt_idx = read_and_crop_chrom_files(
                os.path.join(
                    predicted_data_directory,
                    base_cell_line
                ),
                compact_idxs
            )
            compact_idxs = cmpt_idx
            # Double check if the idxes are the same otherwise we made a mistake with this
            if not np.array_equal(idxes, indexes):
                print(predicted.shape)
                print(method, ' doesnt have valid results')
                exit(1)
            
            if method not in samples_dictionary.keys():
                samples_dictionary[method] = predicted
    
    
    samples_dictionary['inputs'] = inputs
    samples_dictionary['target'] = targets
    
    for key in samples_dictionary.keys():
        samples_dictionary[key] = get_every_nth_element(samples_dictionary[key])
    input_encodings = get_every_nth_element(input_encodings)
    indexes = get_every_nth_element(indexes)
    
    
              
    # Now we start setting up the evaluation files and visualizations
    for key in samples_dictionary.keys():
        output_path = os.path.join(relevant_directory, key)
        print(output_path)

        # Step 1: Do visualizations with the epigenetic features
        visualization_path = os.path.join(output_path, 'visualizations')
        create_entire_path_directory(visualization_path)
        print('Creating visualization files for {}'.format(key))
        visualize_samples(
            samples_dictionary[key], 
            indexes, 
            input_encodings, 
            enc_order,
            visualization_path
        )
        
        # # Step 2: Setup the 3D recon analysis files
        # recon_analysis_path = os.path.join(output_path, 'recon_analysis')
        # create_entire_path_directory(recon_analysis_path)
        # print('Creating Recon analysis files for {}'.format(key))
        # recon_analysis_files[key] = generate_models(samples_dictionary[key], indexes, recon_analysis_path)
        
        # Step 3: Setup the bio feature analysis files
        # bio_feature_analysis_path = os.path.join(output_path, 'bio_feature_analysis')
        # create_entire_path_directory(bio_feature_analysis_path)
        # print('Creating biological feature analysis files for {}'.format(key))
        # if key != 'inputs':
        #     loops, borders, hairpins = extract_features(samples_dictionary[key], indexes, bio_feature_analysis_path)
        #     bio_feature_analysis_files[key] = {
        #         'loops': loops,
        #         'borders': borders, 
        #         'hairpins': hairpins
        #     }
            
        # # # Step 4: Setup Hi-C similarity analysis files
        # hic_similarity_analysis_path = os.path.join(output_path, 'hic_similarity_analysis')
        # create_entire_path_directory(hic_similarity_analysis_path)
        # print('Creating Hi-C similarity files for {}'.format(key))
        # hic_similarity_analysis_files[key] = create_required_analysis_files(samples_dictionary[key], indexes, hic_similarity_analysis_path, key)
    

    correlation_results = {}
    reconstruction_results = {}
    loops_feature_results = {}
    borders_feature_results = {}
    hairpins_feature_results = {}
    
    
    results_directory = os.path.join(
        GENERATED_RESULTS_DIRECTORY,
        relevant_directory.split('/')[-1]
    )
    create_entire_path_directory(results_directory)
    
    
    
    # Now we run actual evaluations
    for key in samples_dictionary.keys():
        if key == 'target' or key == 'inputs':
            continue
        
        # Step 1: Compute the correlation metrics
        print('Running correlation evaluations for {}'.format(key))
        results = compute_correlation_metrics(samples_dictionary[key], samples_dictionary['target'])
        correlation_results[key] = {}
        for result in results:
            correlation_results[key][result] = float(np.mean(results[result]))
        
           
        # # Step 2: Compute 3D Recon results
        # print('Running reconstruction evaluations for {}'.format(key))
        # results = reconstruction_score(recon_analysis_files[key], recon_analysis_files['target'])
        # reconstruction_results[key] = {}
        # for result in results:
        #     reconstruction_results[key][result] = float(np.mean(results[result]))
        
        # Step 3: Compute Bio feature results
        # print('Running loop feature analysis evaluations for {}'.format(key))
        # loop_results = compare_features(bio_feature_analysis_files[key]['loops'], bio_feature_analysis_files['target']['loops'], rp_values['loops'])
        # loops_feature_results[key] = {}
        # for result in loop_results:
        #     loops_feature_results[key][result] = float(np.mean(loop_results[result]))

        
        # print('Running borders feature analysis evaluations for {}'.format(key))
        # borders_results = compare_features(bio_feature_analysis_files[key]['borders'], bio_feature_analysis_files['target']['borders'], rp_values['borders'])
        # borders_feature_results[key] = {}
        # for result in borders_results:
        #     borders_feature_results[key][result] = float(np.mean(borders_results[result]))
        
        
        # print('Running hairpin feature analysis evaluations for {}'.format(key))
        # hairpin_results = compare_features(bio_feature_analysis_files[key]['hairpins'], bio_feature_analysis_files['target']['hairpins'], rp_values['hairpins'])
        # hairpins_feature_results[key] = {}
        # for result in hairpin_results:
        #     hairpins_feature_results[key][result] = float(np.mean(hairpin_results[result]))
        
    
    
        
    save_results(
        correlation_results, 
        os.path.join(
            results_directory, 
            'correlation_results.json'
        )
    )
    # save_results(
    #     reconstruction_results, 
    #     os.path.join(
    #         results_directory, 
    #         'recon_results.json'
    #     )
    # )
    # save_results(
    #     borders_feature_results, 
    #     os.path.join(
    #         results_directory, 
    #         'borders_results.json'
    #     )
    # )
    # save_results(
    #     hairpins_feature_results, 
    #     os.path.join(
    #         results_directory, 
    #         'hairpin_results.json'
    #     )
    # )
    # save_results(
    #     loops_feature_results, 
    #     os.path.join(
    #         results_directory, 
    #         'loops_results.json'
    #     )
    # )
    
    
    # # For Hi-C Similarity we unfortunately have to setup new set of scripts to run them separately 
    # first = True
    # for key in samples_dictionary.keys():
    #     print('Setting up Hi-C similarity analysis metadata files {}'.format(key))

    #     output_path = os.path.join(relevant_directory)
    #     if key == 'target':
    #         continue
        
    #     input_sparse_matrix_files = hic_similarity_analysis_files[key]
    #     target_sparse_matrix_files = hic_similarity_analysis_files['target']

    #     for chrom in test_chroms:
    #         i_files = list(filter(
    #             lambda x: 'chr_{}'.format(chrom) in x, 
    #             input_sparse_matrix_files
    #         ))
    #         t_files = list(filter(
    #             lambda x: 'chr_{}'.format(chrom) in x, 
    #             target_sparse_matrix_files
    #         ))
    #         output_directory = os.path.join(output_path, 'hic_similarity_analysis', 'chr{}'.format(chrom))
            
    #         create_entire_path_directory(output_directory)
    #         create_metadata_files_for_hic_similarity_analysis(i_files, t_files, output_directory, chrom, first)

    #     first= False
        
    
    
    





