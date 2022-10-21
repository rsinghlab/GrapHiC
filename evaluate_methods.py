'''
    This file contains a super wrapper for the evaluation scripts, we work at the per-sample setup here to facilitate
    the ease of evaluation
'''
import os
import json
import numpy as np
import pandas as pd

from related_work.hicnn import test
from src.utils import PREDICTED_FILES_DIRECTORY, GENERATED_RESULTS_DIRECTORY, BASE_DIRECTORY, create_entire_path_directory
from src.dataset_creator import dataset_partitions 
from src.matrix_operations import divide, compactM
from src.visualizations import visualize_samples
from evaluation.correlation_analysis import compute_correlation_metrics
from evaluation.recon_analysis import generate_models, compute_reconstruction_scores
from evaluation.hic_similarity_analysis import setup_hic_similarity_files


cropping_params = {
    'sub_mat'   :200,
    'stride'    :50,
    'bounds'    :40,
    'padding'   :True
}

resolution = 10000

graphic_best_predicted_samples = os.path.join(PREDICTED_FILES_DIRECTORY, 'graphic-best')
graphic_positional_predicted_samples = os.path.join(PREDICTED_FILES_DIRECTORY, 'graphic-positional')
graphic_rad21_predicted_samples = os.path.join(PREDICTED_FILES_DIRECTORY, 'graphic-rad21')
graphic_baseline_predicted_samples = os.path.join(PREDICTED_FILES_DIRECTORY, 'graphic-baseline')



hicnn_predicted_samples = os.path.join(PREDICTED_FILES_DIRECTORY, 'HiCNN')
hicreg_predicted_samples = os.path.join(PREDICTED_FILES_DIRECTORY, 'HiCReg')

predicted_cell_lines = [
    'cross-celltype-k562'
]

test_chroms = dataset_partitions['test']





def collate_all_predictions(cell_line):
    output_file = os.path.join(
        GENERATED_RESULTS_DIRECTORY,
        cell_line
    )
    create_entire_path_directory(output_file)
    output_file = os.path.join(
        output_file,
        'samples.npz'
    )

    # if os.path.exists(output_file):
    #     print('File already Generated')
    #     return output_file


    # Reading Graphic samples is easy because already saved as samples
    graphic_predicted_samples_data = np.load(
        os.path.join(
            graphic_best_predicted_samples,
            cell_line,
            'predicted_samples.npz'
        )
    )


    inputs = graphic_predicted_samples_data['input']
    targets = graphic_predicted_samples_data['target']
    graphic_best_predicted = graphic_predicted_samples_data['graphic']
    indexes = graphic_predicted_samples_data['index']
    
    # graphic_predicted_samples_data = np.load(
    #     os.path.join(
    #         graphic_positional_predicted_samples,
    #         cell_line,
    #         'predicted_samples.npz'
    #     )
    # )
    # graphic_positional_predicted = graphic_predicted_samples_data['graphic']

    # graphic_predicted_samples_data = np.load(
    #     os.path.join(
    #         graphic_rad21_predicted_samples,
    #         cell_line,
    #         'predicted_samples.npz'
    #     )
    # )
    # graphic_rad21_predicted = graphic_predicted_samples_data['graphic']


    # graphic_predicted_samples_data = np.load(
    #     os.path.join(
    #         graphic_baseline_predicted_samples,
    #         cell_line,
    #         'predicted_samples.npz'
    #     )
    # )
    # graphic_baseline_predicted = graphic_predicted_samples_data['graphic']



    print(inputs.shape, targets.shape, graphic_best_predicted.shape)
    hic_reg_compact = {}

    # Read HiCNN chroms and crop them
    hicnn_predicted = []
    hicnn_predicted_inds = []

    for test_chrom in test_chroms:
        predicted_chrom_data = np.load(os.path.join(
            hicnn_predicted_samples,
            cell_line,
            'chr{}.npz'.format(test_chrom)
        ))
        compact_idxs = predicted_chrom_data['compact']
        hic_reg_compact[test_chrom] = compact_idxs

        predicted_chrom_data = compactM(predicted_chrom_data['hic'], compact_idxs)

        divided_chrom, inds = divide(predicted_chrom_data, test_chrom, cropping_params)
        hicnn_predicted.append(divided_chrom)
        hicnn_predicted_inds.append(inds)


    hicnn_predicted = np.concatenate(hicnn_predicted, axis=0)    
    hicnn_predicted_inds = np.concatenate(hicnn_predicted_inds, axis=0)

    print(hicnn_predicted.shape, hicnn_predicted_inds.shape)
    
    if not np.array_equal(hicnn_predicted_inds, indexes):
        print('Unequal arrays')
        exit(1)




    # Read HiCReg chroms and crop them
    # hicreg_predicted = []
    
    # for test_chrom in test_chroms:
    #     predicted_chrom_data = np.load(os.path.join(
    #         hicreg_predicted_samples,
    #         cell_line.split('-')[0],
    #         'chr{}.npz'.format(test_chrom)
    #     ))

    #     predicted_chrom_data = compactM(predicted_chrom_data['hic'], hic_reg_compact[test_chrom])

    #     divided_chrom, _ = divide(predicted_chrom_data, test_chrom, cropping_params)
    #     hicreg_predicted.append(divided_chrom)
        

    # hicreg_predicted = np.concatenate(hicreg_predicted, axis=0)
    
    print('Saving combined results file at {}'.format(output_file))
    np.savez_compressed(
        output_file, 
        input=inputs, 
        target=targets,
        graphic_best=graphic_best_predicted,
        # graphic_positional=graphic_positional_predicted,
        # graphic_rad21=graphic_rad21_predicted,
        # graphic_baseline=graphic_baseline_predicted,
        hicnn=hicnn_predicted,
        # hicreg=hicreg_predicted, 
        index=indexes
    )

    # Combine all these samples together and save the partially processed samples and return
    return output_file



def get_every_nth_element(data, step=3):
    return_data = {}
    for key in data.keys():
        array = data[key]
        num = len(array)//step

        req_indexes = np.round(np.linspace(1, len(array)-1, num=num)).astype(int)
        return_data[key] = array[req_indexes]
    return return_data


def evaluate():
    methods = ['input', 'target', 'hicnn', 'hicreg']
    samples_files = []
    # We do this first to save all the files to save recomputation costs
    for predicted_cell_line in predicted_cell_lines:
        samples_files.append(collate_all_predictions(predicted_cell_line))

    visualizations_path = os.path.join(
        GENERATED_RESULTS_DIRECTORY,
        'visualizations'
    )
    create_entire_path_directory(visualizations_path)
    

    # results_json['MSE' ] = np.array([])
    # results_json['MAE' ] = np.array([])
    # results_json['PSNR'] = np.array([])
    # results_json['SSIM'] = np.array([])
    # results_json['PCC' ] = np.array([])
    # results_json['SCC'] = np.array([])
    # results_json['INS'] = np.array([])
    # results_json['RECON'] = np.array([])

    for idx, predicted_cell_line in enumerate(predicted_cell_lines):
        results_json = {
            'cell_line' : np.array([]),
            'method'    : np.array([]),
            'chromosome': np.array([]),
            'i_pos'     : np.array([]),
            'j_pos'     : np.array([]),
            'RECON'     : np.array([]),
        } 
        # Results file path
        results_path = os.path.join(
            GENERATED_RESULTS_DIRECTORY,
            '{}_results.csv'.format(predicted_cell_line)
        )
        # Sample file path
        samples_file_path = samples_files[idx]
        
        # Read out the data
        data = np.load(samples_file_path)
        
        data = get_every_nth_element(data, 6)
        # indexes = data['index']
        # targets = data['target']
        
        # for method in methods:   
        #     samples = data[method]
        #     path = os.path.join(visualizations_path, predicted_cell_line, method)
        #     create_entire_path_directory(path)

        #     visualize_samples(samples, indexes, path)
            
        
        
        
        # models_path = generate_models(data, predicted_cell_line)
        command_script = setup_hic_similarity_files(data, predicted_cell_line, resolution)

        # print(command_script)

        # print('Total Samples: {}'.format(targets.shape))

        # for method in methods:
        #     curr_method = data[method]
            
            

        #     results_json['cell_line']   = np.concatenate((results_json['cell_line'], [predicted_cell_line]*targets.shape[0]), axis=0)
        #     results_json['method']      = np.concatenate((results_json['method'], [method]*targets.shape[0]), axis=0)
        #     results_json['chromosome']  = np.concatenate((results_json['chromosome'], indexes[:, 0].tolist()), axis=0)
        #     results_json['i_pos']       = np.concatenate((results_json['i_pos'], indexes[:, 2].tolist()), axis=0)
        #     results_json['j_pos']       = np.concatenate((results_json['j_pos'], indexes[:, 3].tolist()), axis=0)
        
        
        #     # results = compute_correlation_metrics(
        #     #     curr_method,
        #     #     targets
        #     # )
                
        #     recon_scores = compute_reconstruction_scores(models_path, method, predicted_cell_line, indexes)
            

        #     # for key in results.keys():
        #     #     results_json[key] = np.concatenate((results_json[key], results[key]), axis=0)
            
        #     results_json['RECON'] = np.concatenate((results_json['RECON'], recon_scores), axis=0)

        # for key in results_json.keys():
        #     print(key, len(results_json[key]))


        # results_df = pd.DataFrame.from_dict(results_json)
        # results_df.to_csv(results_path)

        



    #     if visualize_samples:
    #         visualization_output = os.path.join(
    #             GENERATED_RESULTS_DIRECTORY,
    #             predicted_cell_line,
    #             'visualizations',
    #         )
    #         create_entire_path_directory(visualization_output)

    #         for i in range(targets.shape[0]):
    #             visualize_multiple_samples(
    #                 data['input'][i, 0, :, :],
    #                 data['graphic_best'][i, 0, :, :],
    #                 data['graphic_rad21'][i, 0, :, :],
    #                 data['graphic_positional'][i, 0, :, :],
    #                 data['hicnn'][i, 0, :, :],
    #                 data['hicreg'][i, 0, :, :],
    #                 data['target'][i, 0, :, :],
    #                 data['index'][i, :],
    #                 visualization_output
    #             )

        



    



    


def summarize_results():
    results_file = os.path.join(
        GENERATED_RESULTS_DIRECTORY,
        'GM12878-geo-026_results.csv'
    )
    
    results_df = pd.read_csv(results_file)
    results_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    results_df = results_df.dropna()
    results_df.reset_index(drop=True)

    
    metrics = [
      'RECON'
    ]
    methods = ['input', 'graphic_best', 'graphic_positional', 'graphic_rad21', 'hicreg', 'hicnn']
    
    for method in methods:    
        method_df = results_df.loc[results_df['method'] == method]

        for metric in metrics:
            scores = method_df[metric].dropna() 
            print('{} {}: {:.4f}'.format(method, metric, np.mean(scores)))
            

        

    




evaluate()
#summarize_results()

































































