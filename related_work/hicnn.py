import sys
sys.path.append('../GrapHiC/')

import os
import numpy as np
import time
import multiprocessing
import torch


from itertools import repeat
from multiprocessing import Pool

from src.matrix_operations import divide
from src.matrix_operations import compactM

from src.utils import load_hic_file
from src.utils import create_entire_path_directory

from src.epigentic_encodings import read_node_encoding_files
from src.normalizations import normalize_hic_matrix
from src.positional_encodings import encoding_methods
from src.noise import noise_types

MULTIPROCESSING = True

from tqdm import tqdm

from src.utils import BASE_DIRECTORY, create_entire_path_directory, delete_files, GENERATED_DATA_DIRECTORY, PREDICTED_FILES_DIRECTORY
from src.matrix_operations import process_graph_batch, spreadM, together
from src.evaluations import MSE, SSIM, PCC
from torch.utils.tensorboard import SummaryWriter
from src.visualizations import visualize

dataset_partitions = {
    'train': [1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15, 16, 17, 18],
    'valid': [8, 19, 10, 22],
    'test': [9, 20, 21, 11],
    'debug_train': [19],
    'debug_test': [20], 
    'all': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
}




def process_compact_idxs(base_compact_idx, target_compact_idx, compact_method='intersection'):
    '''
        Processes the compact indexes based on the method type provided.
        @params: base_compact_idx <List>, Compaction indices of the base matrix
        @params: target_compact_idx <List>, Compaction indices of the target matrix
        @params: compact_method <string>, Compaction strategy, intersection is the default strategy  
        @returns: <List> Compact indices
    '''
    if compact_method == 'ignore':
        compact_indexes = []
            
    elif compact_method == 'intersection':
        compact_indexes = list(set.intersection(set(base_compact_idx), set(target_compact_idx)))
       
    elif compact_method == 'target':
        compact_indexes = target_compact_idx
        
    else:
        print("Invalid value for variable compact_type, please choose one of [ignore, intersection, target].")
        exit(1)

    return compact_indexes

def process_chromosome_files(args):
    return _process_chromosome_files(*args)

def _process_chromosome_files(
        path_to_base_chrom, 
        path_to_target_chrom,
        normalization_params,
        cropping_params,
        compact,
        verbose
    ):
    chromosome = int(path_to_base_chrom.split('/')[-1].split('.')[0].split('chr')[1])

    if verbose: print('Processing chromosome {}'.format(chromosome))
    # Read chromosome files
    try:
        base_data = load_hic_file(path_to_base_chrom)
        target_data = load_hic_file(path_to_target_chrom)
    except FileNotFoundError:
        print('Skipping chromosome {} because file is missing'.format(chromosome))
        return ()

    base_chrom = base_data['hic']
    target_chrom = target_data['hic']
    
    # Ensure that that shapes match, otherwise we can not proceede
    assert(base_chrom.shape[0] == target_chrom.shape[0])
    full_size = base_chrom.shape[0]
    
    # Find the informative rows and remove the rest from the data
    if verbose: print('Removing non-infomative rows and columns')
    base_compact_idx = base_data['compact']
    target_compact_idx = target_data['compact'] 
    
    if len(base_compact_idx) == 0 or len(target_compact_idx) == 0:
        print('Chromosome file {} does not contain valid data'.format(base_compact_idx))
        return ()
        

    compact_indexes = process_compact_idxs(base_compact_idx, target_compact_idx, compact_method=compact)
    base_chrom = compactM(base_chrom, compact_indexes)
    target_chrom = compactM(target_chrom, compact_indexes)
                 
    
    # Normalize the HiC Matrices
    if normalization_params['chrom_wide']:
        if verbose: print('Performing Chromosome wide normalization...')
        base_chrom = normalize_hic_matrix(base_chrom, normalization_params, chromosome=chromosome)
        target_chrom = normalize_hic_matrix(target_chrom, normalization_params, chromosome=chromosome)

        
    # Divide the HiC Matrices
    if verbose: print('Dividing the Chromosomes...')
    bases, _ = divide(base_chrom, chromosome, cropping_params)        
    targets, inds = divide(target_chrom, chromosome, cropping_params)

    return (
            chromosome, 
            bases, 
            targets,
            inds,
            compact_indexes,
            full_size,
        )


def create_dataset(
                    path_to_base_files,
                    path_to_target_files, 
                    path_to_output_folder,
                    cropping_params,
                    normalization_params,
                    compact='intersection',
                    datasets=[
                    'train', 
                    'valid', 
                    'test'
                    ],
                    verbose=True
    ):
    '''
        This function takes in path to the base chromospositional_encoding_methodte path where to store the generated dataset file
        @params: positional_encoding_method <string>, what positional encoding method to use
        @params: nodeargs_encoding_files <string> path to the node encoding files, these files contain the epigenetic signals
        @params: cropping_params <dict> contains the cropping parameters necessary for cropping huge chromosome files
        @params: normalization_params <dict> normalization parameters defining how to normalize the Hi-C matrix
        @params: compact <string>, what matrix compaction method to use to remove rows (and columns) that are non informative. Defaults 'intersection'. 
        @params: dataset <list>, what dataset to construct 
        @returns: None
    '''

    create_entire_path_directory(path_to_output_folder)

    for dataset in datasets:
        start_time = time.time()
        if verbose: print('Creating {} dataset'.format(dataset))

        chromosomes = dataset_partitions[dataset]
        output_file = os.path.join(path_to_output_folder, '{}.npz'.format(dataset))
        
        base_chrom_paths = list(
            map(
                lambda x: os.path.join(path_to_base_files, 'chr{}.npz'.format(x)),
                chromosomes
        ))
        target_chrom_paths = list(
            map(
                lambda x: os.path.join(path_to_target_files, 'chr{}.npz'.format(x)),
                chromosomes
        ))
        
        args = zip(
            base_chrom_paths, 
            target_chrom_paths,
            repeat(normalization_params),
            repeat(cropping_params),
            repeat(compact),
            repeat(verbose),
        )
        num_cpus = multiprocessing.cpu_count() if MULTIPROCESSING else 1
        
        with Pool(num_cpus) as pool:
            results = pool.map(process_chromosome_files, args)
        
        
        results = list(filter(lambda x: len(x) != 0, results))
            

        data = np.concatenate([r[1] for r in results])
        target = np.concatenate([r[2] for r in results])
        inds = np.concatenate([r[3] for r in results])

        compacts = {r[0]: r[4] for r in results}
        sizes = {r[0]: r[5] for r in results}

        
        print('Saving file:', output_file)
        np.savez_compressed(output_file, data=data, target=target, inds=inds, compacts=compacts, sizes=sizes)
        end_time = time.time()

        print('Creating {} dataset took {} seconds!'.format(dataset, end_time-start_time))







def data_info(data):
    indices = data['inds']
    compacts = data['compacts'][()]
    sizes = data['sizes'][()]
    return indices, compacts, sizes

def save_data(predicted_hic, compact, size, file):
    hic = spreadM(predicted_hic, compact, size, convert_int=False, verbose=True)
    np.savez_compressed(file, hic=hic, compact=compact)
    print('Saving file:', file)



def train(model, train_loader, optimizer):
    model.training = True
    epoch_loss = 0.0
    num_batches = 0
    for i, (data) in enumerate(train_loader):
        if i == (len(train_loader) - 1):
            continue
        base, target, ind, _ = data
        base = base.to(model.device)
        target = target.to(model.device)
        optimizer.zero_grad()
        
        output = model(base)
        batch_loss = model.loss(output, target)

        batch_loss.backward()
        optimizer.step()
        epoch_loss = epoch_loss + batch_loss
        num_batches += 1


    # Compute average loss per sample
    avg_loss = float(epoch_loss)/(num_batches*model.hyperparameters['batch_size'])    
    return avg_loss



def validate(model, valid_loader):
    model.training = False
    validation_loss = 0.0
    num_batches = 0

    for i, (data) in enumerate(valid_loader):
        if i == (len(valid_loader) - 1):
            continue
        
        base, target, ind, _ = data
        base = base.to(model.device)
        target = target.to(model.device)

        output = model(base)
        batch_loss = model.loss(output, target)        

        validation_loss = validation_loss + batch_loss.item()
        
        num_batches += 1
    
    validation_loss = float(validation_loss)/(num_batches*model.hyperparameters['batch_size']) 
    
    return validation_loss


def test(model, test_loader, output_path):
    model.training = False

    baseline_mse_score = 0.0
    baseline_ssim_score = 0.0
    baseline_pcc_score = 0.0


    upscaled_mse_score = 0.0
    upscaled_ssim_score = 0.0
    upscaled_pcc_score = 0.0

    total_batches = 0
    
    for data in tqdm(test_loader, desc='Predicting: '):
        base, target, ind, target_full = data

        base = base.to(model.device)
        target = target.to(model.device)
        
        
        outputs = model(base)
        


        baseline_mse_score += MSE(base, target_full)
        baseline_ssim_score += SSIM(base, target_full)
        baseline_pcc_score += PCC(base, target_full)

        upscaled_mse_score += MSE(outputs, target)
        upscaled_ssim_score += SSIM(outputs, target)
        upscaled_pcc_score += PCC(outputs, target)
        

        total_batches += 1
        
        visualize(base, outputs, target, total_batches, os.path.join(output_path, 'visualizations'))


    with open(os.path.join(output_path, 'metrics.dump'), 'a+') as dump_file:
        dump_file.write('Mean Squared Error --- Baseline: {}, Generated: {}\n'.format(
            (baseline_mse_score/total_batches), (upscaled_mse_score/total_batches)
        ))
        dump_file.write('SSIM --- Baseline: {}, Generated: {}\n'.format(
            (baseline_ssim_score/total_batches), (upscaled_ssim_score/total_batches)
        ))
        dump_file.write('Pearson\'s Correlation Coefficient --- Baseline: {}, Generated: {}\n'.format(
            (baseline_pcc_score/total_batches), (upscaled_pcc_score/total_batches)
        ))




def generate_chromosomes(model, file_test, output_path, base):
    model.training = False
    loader = model.load_data(
        file_test, 
        1,
        False
    )


    result_data = []
    result_inds = []
    
    _, compacts, sizes = data_info(np.load(file_test, allow_pickle=True))
    output_path = os.path.join(output_path, base)
    create_entire_path_directory(output_path)

    for i, data in enumerate(loader):
        if i == (len(loader) - 1):
            continue
        base, target, ind, target_full = data

        base = base.to(model.device)
        target = target.to(model.device)
        
        outputs = model(base)

        result_data.append(outputs.detach().to('cpu').numpy())
        result_inds.append(ind.numpy())
    
    
    result_data = np.concatenate(result_data, axis=0)
    result_inds = np.concatenate(result_inds, axis=0).reshape(-1, 4)
    
    predicted = together(result_data, result_inds, tag='Reconstructing: ')

    def save_data_n(key):
        file = os.path.join(output_path, f'chr{key}.npz')
        save_data(predicted[key], compacts[key], sizes[key], file)

    print(f'Saving predicted data as individual chromosome files')

    for key in compacts.keys():
        save_data_n(key)






def run(
        model,
        file_train, 
        file_validation,
        file_test,
        base,
        target,
        retrain=True, 
        debug=True
    ):
    '''
        This function trains, validates, and tests the provided model
    '''
    # Step 0: Setup the tensorboard writer, create or ensure the required directories exist
    # Tensor board writer
    tensor_board_writer_path = os.path.join(BASE_DIRECTORY, 'logdir', model.model_name)
    print(tensor_board_writer_path)

    if not os.path.exists(tensor_board_writer_path):
        create_entire_path_directory(tensor_board_writer_path)
    
    # Create tensorboard writer
    writer = SummaryWriter(os.path.join(tensor_board_writer_path, model.model_name))

    # Weights directory exists
    if not os.path.exists(model.weights_dir):
        create_entire_path_directory(model.weights_dir)
    
    # Clean the existing sets of weights in the model.dir_model directory
    if retrain:
        delete_files(model.weights_dir)

    # Create and clean the output directory 
    output_path = os.path.join(GENERATED_DATA_DIRECTORY, model.model_name)
    output_path_chrom = os.path.join(PREDICTED_FILES_DIRECTORY, base + '@' + target, model.model_name)
    if not os.path.exists(output_path):
        create_entire_path_directory(output_path)

    # Step 1: Initialize the model and the optimizer
    if debug: print('Initializing the model parameters')
    # Move model to the defined device
    model.to(model.device)

    if debug: print('Initializing the Optimizer')
    # Initialize the optimizer  TODO: Make this generic as well
    optimizer = model.create_optimizer()
    
    training_losses = []
    validation_losses = []
    min_loss = None

    # Step 2: Enter the main training loop
    if retrain:
        if debug: print('Loading Training Dataset')
        train_loader = model.load_data(file_train)
        if debug: print('Loading Validation Dataset')
        validation_loader = model.load_data(file_validation)
        
        for epoch in tqdm(range(model.hyperparameters['epochs']), 'Training Epochs'):
            training_loss = train(model, train_loader, optimizer)
            validation_loss = validate(model, validation_loader)

            training_losses.append(training_loss)
            validation_losses.append(validation_loss)
            if min_loss == None:
                print('Updating MinLoss to {}'.format(min_loss))
                min_loss = validation_loss
            
            if validation_loss <= min_loss:
                print('Updating MinLoss to {}'.format(min_loss))
                min_loss = validation_loss
                torch.save(model.state_dict(), os.path.join(model.weights_dir, '{}-epoch_{}-loss_model'.format(0, 0)))
            
            # writer.add_scalar("Loss/Training",      training_loss,      epoch)
            # writer.add_scalar("Loss/Validation",    validation_loss,    epoch)
            # torch.save(model.state_dict(), os.path.join(model.weights_dir, '{}-epoch_{}-loss_model'.format(epoch, validation_loss)))

    print(training_losses)
    print(validation_losses)

    model.load_weights()
    
    # Step 3: Generate chromosome files
    generate_chromosomes(model, file_test, output_path_chrom, base)

    


























