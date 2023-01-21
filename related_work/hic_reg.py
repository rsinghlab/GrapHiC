'''
    This file contains the implementation of HiCReg 
'''
import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor


from src.epigentic_encodings import read_node_encoding_files
from src.utils import PARSED_HIC_FILES_DIRECTORY, PARSED_EPIGENETIC_FILES_DIRECTORY, WEIGHTS_DIRECTORY, PREDICTED_FILES_DIRECTORY
from src.utils import load_hic_file, create_entire_path_directory
from src.dataset_creator import dataset_partitions
from src.normalizations import normalize_hic_matrix

HIC_REG_GENERATED_DATASETS = '/users/gmurtaza/data/gmurtaza/hic-reg_datasets/'
create_entire_path_directory(HIC_REG_GENERATED_DATASETS)

# These parameters are also used by the dataset creator function but this set contains the normalization parameters
normalization_params = {
    'norm'              : True,  # To normalize or not
    'remove_zeros'      : True,  # Remove zero before percentile calculation
    'set_diagonal_zero' : False, # Remove the diagonal before percentile calculation
    'percentile'        : 99.5,  # Percentile 
    'rescale'           : True,  # After applying cutoff, rescale between 0 and 1
    'chrom_wide'        : True,  # Apply it on chromosome scale #TODO: Sample wise normalization isn't implemented
    'draw_dist_graphs'  : False,  # Visualize the distribution of the chromosome
    'edge_culling'      : -1,
}

epi_features_list = [
    'RAD-21', 'RNA-Pol2','CTCF', 'DNASE-Seq', 'H3K27ME3', 'H3K27AC', 
    'H3K36ME3', 'H3K4ME1', 'H3K4ME2', 'H3K4ME3', 'H3K79ME2', 'H3K9AC', 'H4K20ME1', 
    'H3K9ME3'
]

def load_dataset(
    cell_line,
    resolution,
    max_distance = 200,
    dataset='debug'
):  
    output_path = os.path.join(
        HIC_REG_GENERATED_DATASETS, 
        cell_line,
        '{}_{}'.format(resolution, max_distance)
    )
    create_entire_path_directory(output_path)
    
    output_file = os.path.join(output_path, '{}.npz'.format(dataset))
    
    # if os.path.exists(output_file):
    #     print('File already exists, reading the existing dataset and returning')
    #     data = np.load(output_file)['data']

    #     return data
    
    # we store a generated array for each chrom 
    # and then merge them and shuffle them
    final_dataset = []




    for chrom in dataset_partitions[dataset]:
        # Load the hic dataset
        hic_data = load_hic_file(os.path.join(
            PARSED_HIC_FILES_DIRECTORY,
            '{}-geo-raoetal'.format(cell_line),
            'resolution_{}'.format(resolution),
            'chr{}.npz'.format(chrom)
        ))['hic']
        hic_data = normalize_hic_matrix(hic_data, normalization_params, chromosome=chrom)

        node_encoding_files = list(map(
            lambda x: os.path.join(
                PARSED_EPIGENETIC_FILES_DIRECTORY,
                cell_line, 
                x
            ),
            epi_features_list
        ))
        epi_features, _ = read_node_encoding_files(node_encoding_files, chrom, {}, [], False)

        idxes = []
        for row in range(0, hic_data.shape[0]):
            for col in range(row, row + max_distance):
                if col < hic_data.shape[1]:
                    idxes.append((row, col))

        # 2 times epi features for each bin, 2 for chrom id and size and 2 for hic count and distance
        chrom_dataset = np.zeros((len(idxes), (2*epi_features.shape[1] + 2 + 2)))

        for index, idx in enumerate(idxes):
            i, j = idx
            distance = abs((i*resolution) - (j*resolution))
            actual_hic_count = hic_data[i, j]
            epi_i = epi_features[i, :]
            epi_j = epi_features[j, :]
            
            chrom_dataset[index, 0] = chrom
            chrom_dataset[index, 1] = hic_data.shape[0]
            chrom_dataset[index, 2] = i
            chrom_dataset[index, 3] = j
            chrom_dataset[index, 4:(4+epi_features.shape[1])] = epi_i
            chrom_dataset[index, (4+epi_features.shape[1]):((4+epi_features.shape[1])+epi_features.shape[1])] = epi_j
            chrom_dataset[index, -2] = distance
            chrom_dataset[index, -1] = actual_hic_count

        
        final_dataset.append(chrom_dataset)
    
    final_dataset = np.concatenate(final_dataset, axis=0)

    # save the final dataset before returning 
    np.savez_compressed(output_file, data=final_dataset)

    return final_dataset






def run(
    cell_line, 
    resolution, 
    train=False
):
    
    generated_chrom_path = os.path.join(
        PREDICTED_FILES_DIRECTORY,
        cell_line,
        'HiCReg'
    )
    create_entire_path_directory(generated_chrom_path)


    hic_reg = RandomForestRegressor(
        n_estimators=20, 
        min_samples_leaf=10, 
        random_state = 42, 
        n_jobs=32,
        verbose=True
    )

    if train:
        train_dataset = load_dataset(cell_line, resolution, dataset='train')
        training_features = train_dataset[:, 4:(len(epi_features_list)*2 + 1)]
        training_labels = train_dataset[:, -1]
        hic_reg.fit(training_features, training_labels)
        joblib.dump(
            hic_reg,
            os.path.join(WEIGHTS_DIRECTORY, 'hic_reg')
        )
    else:
        hic_reg = joblib.load(os.path.join(WEIGHTS_DIRECTORY, 'hic_reg'))



    test_dataset = load_dataset(cell_line, resolution, dataset='test')

    

    testing_features = test_dataset[:, 4:(len(epi_features_list)*2 + 1)]
    testing_chrom_and_pos = test_dataset[:, :4]
    
    # Predict on test chromomsomes
    test_predictions = hic_reg.predict(testing_features)

    # Save test chromosomes
    test_chroms = np.unique(testing_chrom_and_pos[:, 0]).astype(int)    

    for chrom in test_chroms:
        # output path
        output_path = os.path.join(
            generated_chrom_path,
            'chr{}.npz'.format(
                 chrom               
            )
        )
        # get all rows that have current test chrom
        chrom_positions = np.argwhere(testing_chrom_and_pos[:, 0] == chrom)
        # Chrom shape
        chrom_shape = int(testing_chrom_and_pos[chrom_positions[0]][0, 1])
        output_chrom = np.zeros((chrom_shape, chrom_shape))

        for chrom_position in chrom_positions:
            i = int(testing_chrom_and_pos[chrom_position[0], 2])
            j = int(testing_chrom_and_pos[chrom_position[0], 3])
            output_chrom[i, j] = test_predictions[chrom_position[0]]
            output_chrom[j, i] = test_predictions[chrom_position[0]]
        
        print('Saving Chromosome at position {}'.format(output_path))
        np.savez_compressed(output_path, hic=output_chrom, compact=[])



























