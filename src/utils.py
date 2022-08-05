import os, shutil
import numpy as np


BASE_DIRECTORY = '/users/gmurtaza/GrapHiC/'
HIC_FILES_DIRECTORY = '/users/gmurtaza/data/gmurtaza/hic_datasets' # Recommended that this path is on some larger storage device
PARSED_HIC_FILES_DIRECTORY = '/users/gmurtaza/data/gmurtaza/parsed_hic_datasets'
DATASET_DIRECTORY = '/users/gmurtaza/data/gmurtaza/generated_datasets'

WEIGHTS_DIRECTORY = os.path.join(BASE_DIRECTORY, 'weights') # Recommended to keep weights on the same directory
GENERATED_DATA_DIRECTORY = os.path.join(BASE_DIRECTORY, 'outputs')


def load_hic_file(path, format='.npz'):
    '''
        A wrapper that reads the HiC file and returns it, it essentially is created to 
        later support multiple HiC file formats, currently the only utilized file format is 
        .npz. 

        @params: 
    '''
    if format == '.npz':
        return np.load(path, allow_pickle=True)
    else:
        print('Provided file format is invalid')
        exit(1)



def delete_files(folder_path):
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        try:
            shutil.rmtree(filepath)
        except OSError:
            os.remove(filepath)




def create_entire_path_directory(path):
    """
        Given a path, creates all the missing directories on the path and 
        ensures that the given path is always a valid path
        @params: <string> path, full absolute path, this function can act wonkily if given a relative path
        @returns: None
    """
    
    path = path.split('/')
    curr_path = '/'
    for dir in path:
        curr_path = os.path.join(curr_path, dir)
        if os.path.exists(curr_path):
            continue
        else:
            os.mkdir(curr_path)


def graph_random_walks(adjacency_matrix):
    pass