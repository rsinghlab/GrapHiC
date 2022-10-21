import os
from src.utils import BASE_DIRECTORY, create_entire_path_directory, compress_file, GENERATED_RESULTS_DIRECTORY
from src.dataset_creator import dataset_partitions


PARAMETERS_FILE =  os.path.join(
    BASE_DIRECTORY,
    'parameters.txt'
)

def create_bins_file(hic_similarity_files_directory, resolution, chr, sub_mat=200):
    output_path = os.path.join(
        hic_similarity_files_directory, 
        'bins.bed'
    )
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


def create_sparse_contact_matrices(samples, indexes, output_folder, method, chromosome, resolution):
    all_file_paths = []
    # Create a separate folder for 
    for idx in range(len(indexes)):
        if chromosome != indexes[idx][0]:
            continue
        
        file_path = os.path.join(
            output_folder, 
            'method_{}-i_{}-j_{}-contact.matrix'.format(
                method,
                indexes[idx][2], 
                indexes[idx][3]
            )
        )
        

        sample = samples[idx, 0, :, :]*255

        with open(file_path, 'w') as f:
            for x in range(0, sample.shape[0]):
                for y in range(x, sample.shape[1]):
                    contact_value = int(sample[x, y])
                    ith_chr_index = x*resolution
                    jth_chr_index = y*resolution
                    
                    if contact_value != 0:
                        f.write("{}\t{}\t{}\t{}\t{}\n".format(
                            'chr{}'.format(indexes[idx][0]), 
                            ith_chr_index,
                            'chr{}'.format(indexes[idx][0]),
                            jth_chr_index,
                            '{}'.format(contact_value)
                        ))
                
        file_path = compress_file(file_path, clean=True)
        all_file_paths.append(file_path)
    
    return all_file_paths

def create_metadata_samples(sparse_matrices_paths, output):
    output = os.path.join(
        output,
        'metadata.samples'
    )
    with open(output, 'w') as f:
        for path in sparse_matrices_paths:
            identifier = '-'.join(path.split('/')[-1].split('-')[:-1])
            f.write(
                "{}\t{}\n".format(identifier, path)
            )
    return output


def create_metadata_pairs(methods, indexes, output, chromosome):
    output = os.path.join(
        output,
        'metadata.pairs'
    )
    
    with open(output, 'w') as f:
        for method in methods:
            for idx in range(indexes.shape[0]):
                if chromosome != indexes[idx][0]:
                    continue
                base_identifier = 'method_{}-i_{}-j_{}'.format(
                    method,
                    indexes[idx][2], 
                    indexes[idx][3]
                )
                target_identifier = 'method_target-i_{}-j_{}'.format(
                    indexes[idx][2], 
                    indexes[idx][3]
                )
                f.write(
                    "{}\t{}\n".format(
                        base_identifier,
                        target_identifier
                    )
                )
    return output

    


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
    

































