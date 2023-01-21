import os
import numpy as np
from src.utils import BASE_DIRECTORY, create_entire_path_directory, compress_file, GENERATED_RESULTS_DIRECTORY
from src.dataset_creator import dataset_partitions


PARAMETERS_FILE =  os.path.join(
    BASE_DIRECTORY,
    'parameters.txt'
)

def create_bins_file(hic_similarity_files_directory, resolution, chr, sub_mat=230):
    output_path = os.path.join(
        hic_similarity_files_directory, 
        'chr{}_bins.bed'.format(chr)
    )
    
    if os.path.exists(output_path):
        return
    
    with open(output_path, 'w') as f:
        # will we support arbitrary starting locations?
        curr_starting = 0
        curr_ending = curr_starting + resolution
        chr_str = 'chr{}'.format(chr)

        for _ in range(sub_mat):
            f.write('{}\t{}\t{}\t{}\n'.format(chr_str, curr_starting, curr_ending, curr_starting))
            curr_starting = curr_ending
            curr_ending += resolution
    
    # Compress .bed file
    output_path = compress_file(output_path, clean=True)
    return output_path

def create_sparse_matrix_files(samples, indexes, output_folder, method, resolution=10000, upscale=255):
    file_paths = []
    for i in range(samples.shape[0]): 
        idx = indexes[i, :]
        hic = samples[i, 0, :, :]*upscale
        file_path = os.path.join(
            output_folder, 
            '{}_chr_{}-i_{}-j_{}-contact.matrix'.format(
                method,
                idx[0],
                idx[2], 
                idx[3]
            )
        )
        
        if not os.path.exists(file_path+'.gz'):   
            with open(file_path, 'w') as f:
                for x in range(0, hic.shape[0]):
                    for y in range(x, hic.shape[1]):
                        contact_value = int(hic[x, y])
                        ith_chr_index = x*resolution
                        jth_chr_index = y*resolution
                        if contact_value != 0:
                            f.write("{}\t{}\t{}\t{}\t{}\n".format(
                                'chr{}'.format(idx[0]), 
                                ith_chr_index,
                                'chr{}'.format(idx[0]),
                                jth_chr_index,
                                '{}'.format(contact_value)
                            ))
            file_path = compress_file(file_path, clean=True)
        
            
        file_paths.append(file_path)
    return file_paths

def create_metadata_samples(sparse_matrices_paths, file_path, append=True):
    mode = 'a+' if append else 'w'
    with open(file_path, mode) as f:
        for path in sparse_matrices_paths:
            identifier = '-'.join(path.split('/')[-1].split('-')[:-1])
            f.write(
                "{}\t{}\n".format(identifier, path)
            )

def create_metadata_pairs(inputs, targets, file_path, append=True):
    mode = 'a+' if append else 'w'
    #print(inputs, targets)
    
    with open(file_path, mode) as f:
        for input, target in zip(inputs, targets):
           
            
            input_identifier = '-'.join(input.split('/')[-1].split('-')[:-1])
            target_identifier = '-'.join(target.split('/')[-1].split('-')[:-1])   
            f.write(
                "{}\t{}\n".format(
                    input_identifier,
                    target_identifier
                )
            )



def create_required_analysis_files(samples, indexes, output_folder, method, resolution=10000, upscale=255):
    # Create all the sparse matrices
    sparse_matrix_output_folder = os.path.join(output_folder, 'sparse_matrices')
    create_entire_path_directory(sparse_matrix_output_folder)
    sparse_matrix_files = create_sparse_matrix_files(samples, indexes, sparse_matrix_output_folder, method)
    
    return sparse_matrix_files


def save_oscar_script(command_script, output_path):
    bash = '''#!/bin/bash
#SBATCH -p 3090-gcondo
#SBATCH -N 1
#SBATCH --mem=64G
#SBATCH -n 8
#SBATCH --time=24:00:00

#SBATCH -o hic-similarity-%j.out


module load python/2.7.16
module load gcc/10.2
module load R/3.4.0
module load mpi/openmpi_4.0.7_gcc_10.2_slurm22
source /users/gmurtaza/3DChromatin_ReplicateQC/hic_similiarity/bin/activate

cd /users/gmurtaza/3DChromatin_ReplicateQC/

{}
'''.format(
        command_script
    )
    with open(os.path.join(output_path, 'oscar_script'), 'w') as f:
        f.write(bash)

def create_metadata_files_for_hic_similarity_analysis(input_sparse_matrix_files, target_sparse_matrix_files, output_directory, chr_num, first=True):
    parent_directory = '/'.join(output_directory.split('/')[:-1])
    
    create_bins_file(output_directory, 10000, chr_num)
    
    metadata_samples_file = os.path.join(output_directory, 'metadata.samples')
    metadata_pairs_file = os.path.join(output_directory, 'metadata.pairs')
    
    print(metadata_pairs_file)
    
    
    if first:
        create_metadata_samples(input_sparse_matrix_files, metadata_samples_file, False)
        create_metadata_samples(target_sparse_matrix_files, metadata_samples_file, True)
    else:
        create_metadata_samples(input_sparse_matrix_files, metadata_samples_file, True)

    
    if first:
        create_metadata_pairs(input_sparse_matrix_files, target_sparse_matrix_files, metadata_pairs_file, False)
    else:
        create_metadata_pairs(input_sparse_matrix_files, target_sparse_matrix_files, metadata_pairs_file, True)
        
    
    command_script = '3DChromatin_ReplicateQC run_all --concise_analysis --parameters_file {} --metadata_samples {} --metadata_pairs {} --bins {} --outdir {}'.format(
        PARAMETERS_FILE,
        metadata_samples_file,
        metadata_pairs_file,
        os.path.join(output_directory, 'chr{}_bins.bed.gz'.format(chr_num)),
        output_directory
    )
    save_oscar_script(command_script, output_directory)








# def create_sparse_contact_matrices(samples, indexes, output_folder, method, chromosome, resolution):
#     all_file_paths = []
#     # Create a separate folder for 
#     for idx in range(len(indexes)):
#         if chromosome != indexes[idx][0]:
#             continue
        
#         file_path = os.path.join(
#             output_folder, 
#             'method_{}-i_{}-j_{}-contact.matrix'.format(
#                 method,
#                 indexes[idx][2], 
#                 indexes[idx][3]
#             )
#         )
        

#         sample = samples[idx, 0, :, :]*255

#         with open(file_path, 'w') as f:
#             for x in range(0, sample.shape[0]):
#                 for y in range(x, sample.shape[1]):
#                     contact_value = int(sample[x, y])
#                     ith_chr_index = x*resolution
#                     jth_chr_index = y*resolution
                    
#                     if contact_value != 0:
#                         f.write("{}\t{}\t{}\t{}\t{}\n".format(
#                             'chr{}'.format(indexes[idx][0]), 
#                             ith_chr_index,
#                             'chr{}'.format(indexes[idx][0]),
#                             jth_chr_index,
#                             '{}'.format(contact_value)
#                         ))
                
#         file_path = compress_file(file_path, clean=True)
#         all_file_paths.append(file_path)
    
#     return all_file_paths

# def create_metadata_samples(sparse_matrices_paths, output):
#     output = os.path.join(
#         output,
#         'metadata.samples'
#     )
#     with open(output, 'w') as f:
#         for path in sparse_matrices_paths:
#             identifier = '-'.join(path.split('/')[-1].split('-')[:-1])
#             f.write(
#                 "{}\t{}\n".format(identifier, path)
#             )
#     return output


# def create_metadata_pairs(methods, indexes, output, chromosome):
#     output = os.path.join(
#         output,
#         'metadata.pairs'
#     )
    
#     with open(output, 'w') as f:
#         for method in methods:
#             for idx in range(indexes.shape[0]):
#                 if chromosome != indexes[idx][0]:
#                     continue
#                 base_identifier = 'method_{}-i_{}-j_{}'.format(
#                     method,
#                     indexes[idx][2], 
#                     indexes[idx][3]
#                 )
#                 target_identifier = 'method_target-i_{}-j_{}'.format(
#                     indexes[idx][2], 
#                     indexes[idx][3]
#                 )
#                 f.write(
#                     "{}\t{}\n".format(
#                         base_identifier,
#                         target_identifier
#                     )
#                 )
#     return output

    


def setup_hic_similarity_files(data, predicted_cell_line, resolution):

    hic_similarity_files_base_directory = os.path.join(
        GENERATED_RESULTS_DIRECTORY,
        'hic_similarity_analysis'
    )
    
    create_entire_path_directory(hic_similarity_files_base_directory)
    
    hic_similarity_files_base_directory = os.path.join(
        hic_similarity_files_base_directory,
        predicted_cell_line
    )
    
    create_entire_path_directory(hic_similarity_files_base_directory)
    
    for chromosome in dataset_partitions['test']:
        methods = []
    
        hic_similarity_files_directory = os.path.join(
            hic_similarity_files_base_directory,
            'chr{}'.format(chromosome)
        )
        print(hic_similarity_files_directory)

        create_entire_path_directory(hic_similarity_files_directory)

        bins_file = create_bins_file(hic_similarity_files_directory, resolution, chromosome)
    
        sparse_contact_matrices_directory = os.path.join(
            hic_similarity_files_base_directory,
            'chr{}'.format(chromosome),
            'sparse_contact_matrices'
        )
        create_entire_path_directory(sparse_contact_matrices_directory)

        sparse_matrices_paths = []
        for key in data.keys():
            if key in ['index', 'graphic_best']:
                continue
            methods.append(key)
            # Create constraint files
            sparse_matrices_paths += create_sparse_contact_matrices(
                data[key],
                data['index'],
                sparse_contact_matrices_directory,
                key,
                chromosome,
                resolution
            )
        metadata_samples_file = create_metadata_samples(sparse_matrices_paths, hic_similarity_files_directory)
        metadata_pairs_file = create_metadata_pairs(
            methods, 
            data['index'], 
            hic_similarity_files_directory,
            chromosome
        )
        results_directory = os.path.join(
            GENERATED_RESULTS_DIRECTORY, 
            'hic_similarity_results',
            'chr{}'.format(chromosome),
            predicted_cell_line
        )
        command_script = '3DChromatin_ReplicateQC run_all --concise_analysis --parameters_file {} --metadata_samples {} --metadata_pairs {} --bins {} --outdir {}'.format(
            PARAMETERS_FILE,
            metadata_samples_file,
            metadata_pairs_file,
            bins_file,
            results_directory
        )
        print(command_script)
    

































