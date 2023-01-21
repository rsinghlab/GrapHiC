import os
import subprocess
import tmscoring

from src.utils import JAR_LOCATION, BASE_DIRECTORY, create_entire_path_directory

# 3D reconstruction and structure comparison
def create_constrains_files(
    sample,
    index,
    constraints_folder,
):  
    file_path = os.path.join(
        constraints_folder, 
        'chrom:{}-i:{}-j:{}_constraints.txt'.format(
            index[0],
            index[2], 
            index[3]
        )
    )
    with open(file_path, 'w') as f:
        for x in range(0, sample.shape[0]):
            for y in range(x, sample.shape[1]):
                f.write('{}\t{}\t{:.5f}\n'.format(
                    x, y, sample[x, y]
                ))
    return file_path

def create_parameters(constraint, parameters_output_path, output_path):
    suffix = constraint.split('/')[-1].split('_')[0] + '_parameters.txt'
    
    stri = """NUM = 1
OUTPUT_FOLDER = {}
INPUT_FILE = {}
CONVERT_FACTOR = 0.6
VERBOSE = false
LEARNING_RATE = 1
MAX_ITERATION = 10000""".format(
            output_path,
            constraint
    )
    parameter_file = os.path.join(parameters_output_path, suffix)
    
    with open(parameter_file, 'w') as f:
        f.write(stri)
    return parameter_file


def generate_model(param):
    '''
        @params: parameters_file_path <string>, path to folder that contains all the parameters
    '''
    subprocess.run("java -Xmx5000m -jar "+JAR_LOCATION+" "+param, shell=True) 


def generate_models(samples, indexes, output_path):
    constraints_folder = os.path.join(
        output_path,
        'constraints'
    )
    parameters_folder = os.path.join(
        output_path,
        'parameters'
    )
    generated_models_folder = os.path.join(
        output_path,
        'models'
    )
    parameter_files = []
    create_entire_path_directory(constraints_folder)
    create_entire_path_directory(parameters_folder)
    create_entire_path_directory(generated_models_folder)
    for i in range(samples.shape[0]): 
        idx = indexes[i, :]
        hic = samples[i, 0, :, :]
        # Create constraint files
        constraint_file = create_constrains_files(
            hic,
            idx,
            constraints_folder
        )
        parameter_file = create_parameters(
            constraint_file, 
            parameters_folder, 
            generated_models_folder
        )
        generate_model(parameter_file)
        parameter_files.append(parameter_file)
        
    return parameter_files



def get_identifier(path):
    return path.split('/')[-1].split('_')[0]


def reconstruction_score(input_paramter_files, target_paramter_files):
    scores = []
    for input, target in zip(input_paramter_files, target_paramter_files):
        input_identifier = get_identifier(input)
        target_identifier = get_identifier(target)
        
        input_models = os.path.join('/'.join(input.split('/')[:-2]), 'models')
        target_models = os.path.join('/'.join(target.split('/')[:-2]), 'models')

        input_model_pdbs = list(filter(lambda x: x.split('.')[-1] == 'pdb', os.listdir(input_models)))
        target_model_pdbs = list(filter(lambda x: x.split('.')[-1] == 'pdb', os.listdir(target_models)))
 
        input_model = list(filter(lambda x: input_identifier in x, input_model_pdbs))[0]
        target_model = list(filter(lambda x: target_identifier in x, target_model_pdbs))[0]

        input_pdb_path = os.path.join(
            input_models,
            input_model
        )
        target_pdb_path = os.path.join(
            target_models,
            target_model
        )

        alignment = tmscoring.TMscoring(input_pdb_path, target_pdb_path)
        alignment.optimise() 

        score = alignment.tmscore(**alignment.get_current_values())
        
        scores.append(score)

    return {'tm-score': scores}







def compute_reconstruction_scores(models_path, method, predicted_cell_line, indexes):
    scores = []
    models = os.listdir(models_path)
    filter_pdb_files = list(filter(lambda x: x.split('.')[-1] == 'pdb', models))
    target_pdb_files = list(filter(lambda x: x.split('-')[0].split(':')[1] == 'target', filter_pdb_files))
    method_pdb_files = list(filter(lambda x: x.split('-')[0].split(':')[1] ==  method, filter_pdb_files))

    for i in range(indexes.shape[0]):
        target_file_prefix = 'method:{}-cell_line:{}-chrom:{}-i:{}-j:{}'.format(
            'target',
            predicted_cell_line,
            indexes[i][0],
            indexes[i][2],
            indexes[i][3],
        )
        
        method_file_prefix = 'method:{}-cell_line:{}-chrom:{}-i:{}-j:{}'.format(
            method,
            predicted_cell_line,
            indexes[i][0],
            indexes[i][2],
            indexes[i][3],
        )
        
        target_pdb_file = list(filter(lambda x: target_file_prefix in x, target_pdb_files))[0]
        method_pdb_file = list(filter(lambda x: method_file_prefix in x, method_pdb_files))[0]
       
        target_pdb_file = os.path.join(
            models_path,
            target_pdb_file
        )
        method_pdb_file = os.path.join(
            models_path,
            method_pdb_file
        )
        
        alignment = tmscoring.TMscoring(method_pdb_file, target_pdb_file)
        alignment.optimise() 

        score = alignment.tmscore(**alignment.get_current_values())
        
        scores.append(score)
    
    return scores
    