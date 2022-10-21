import os
import subprocess
import tmscoring

from src.utils import JAR_LOCATION, BASE_DIRECTORY, create_entire_path_directory

# 3D reconstruction and structure comparison
def create_constrains_files(
    samples,
    indexes,
    constraints_folder,
    method,
    cell_line,
):
    all_file_paths = []
    # Create a separate folder for 
    for idx in range(samples.shape[0]):
        file_path = os.path.join(
            constraints_folder, 
            'method:{}-cell_line:{}-chrom:{}-i:{}-j:{}-constraints.txt'.format(
                method,
                cell_line,
                indexes[idx][0],
                indexes[idx][2], 
                indexes[idx][3]
            )
        )
        all_file_paths.append(file_path)

        sample = samples[idx, 0, :, :]        
        with open(file_path, 'w') as f:
            for x in range(0, sample.shape[0]):
                for y in range(x, sample.shape[1]):
                    f.write('{}\t{}\t{:.5f}\n'.format(
                        x, y, sample[x, y]
                    ))
    return all_file_paths

def create_parameters(constraints, parameters_output_path, output_path):
    all_parameter_files = []
    for constraint in constraints:
        suffix = constraint.split('/')[-1]
        

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
        all_parameter_files.append(parameter_file)
        with open(parameter_file, 'w') as f:
            f.write(stri)
    return all_parameter_files

def generate_model(parameters_file_path):
    '''
        @params: parameters_file_path <string>, path to folder that contains all the parameters
    '''
    for param in parameters_file_path:
        subprocess.run("java -Xmx5000m -jar "+JAR_LOCATION+" "+param, shell=True) 


def generate_models(data, predicted_cell_line):
    constraints_folder = os.path.join(
        BASE_DIRECTORY,
        'outputs',
        'temp_3d_models',
        'constraints'
    )
    parameters_folder = os.path.join(
        BASE_DIRECTORY,
        'outputs',
        'temp_3d_models',
        'parameters'
    )
    generated_models_folder = os.path.join(
        BASE_DIRECTORY,
        'outputs',
        'temp_3d_models',
        'models'
    )

    create_entire_path_directory(constraints_folder)
    create_entire_path_directory(parameters_folder)
    create_entire_path_directory(generated_models_folder)


    for key in data.keys():
        if key == 'index':
            continue
        # Create constraint files
        constraints = create_constrains_files(
            data[key],
            data['index'],
            constraints_folder,
            key,
            predicted_cell_line
        )
        # Create parameters
        parameters = create_parameters(constraints, parameters_folder, generated_models_folder)
        
        # Generate 3D models
        generate_model(parameters)

    return generated_models_folder

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
    