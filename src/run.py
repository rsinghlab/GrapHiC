'''
    This file contains the wrapper scripts for training, validation and testing a provided 
    deep-learning based model.
'''
import os
import torch
from tqdm import tqdm

from src.utils import BASE_DIRECTORY, create_entire_path_directory, delete_files, GENERATED_DATA_DIRECTORY
from src.matrix_operations import process_graph_batch
from src.evaluations import MSE, SSIM, PCC
from torch.utils.tensorboard import SummaryWriter
from src.visualizations import visualize


def train(model, train_loader, optimizer, epoch):
    epoch_loss = 0.0
    num_batches = 0
    for i, (data) in enumerate(train_loader):
        if i == (len(train_loader) - 1):
            continue
        
        data = data.to(model.device)
        optimizer.zero_grad()
        
        output = model(data)
        # I have unified all the data types to be graphs, each model handles it seperately
        targets =  process_graph_batch(model, data.y, data.batch)
        
        batch_loss = model.loss(output, targets)

        batch_loss.backward()
        optimizer.step()
        epoch_loss = epoch_loss + batch_loss
        num_batches += 1


    # Compute average loss per sample
    avg_loss = float(epoch_loss)/(num_batches*model.hyperparameters['batch_size'])    
    return avg_loss



def validate(model, valid_loader):
    validation_loss = 0.0
    mse = 0.0
    ssim = 0.0 
    pcc = 0.0 

    num_batches = 0

    for i, (data) in enumerate(valid_loader):
        if i == (len(valid_loader) - 1):
            continue
        
        data = data.to(model.device)
        output = model(data)
        targets =  process_graph_batch(model, data.y, data.batch)
        
        batch_loss = model.loss(output, targets)

        validation_loss = validation_loss + batch_loss.item()
        
        mse = mse + MSE(output, targets)
        ssim = ssim + SSIM(output, targets)
        pcc = pcc + PCC(output, targets)
        
        num_batches += 1
    
    validation_loss = float(validation_loss)/(num_batches*model.hyperparameters['batch_size']) 
    mse = float(mse)/(num_batches) 
    ssim = float(ssim)/(num_batches) 
    pcc = float(validation_loss)/(num_batches)

    return validation_loss, mse, ssim, pcc


def test(model, test_loader, output_path):
    baseline_mse_score = 0.0
    baseline_ssim_score = 0.0
    baseline_pcc_score = 0.0


    upscaled_mse_score = 0.0
    upscaled_ssim_score = 0.0
    upscaled_pcc_score = 0.0

    total_batches = 0
    
    for data in tqdm(test_loader, desc='Predicting: '):
        data = data.to(model.device)
        outputs = model(data)
        targets = process_graph_batch(model, data.y, data.batch)
        inputs = process_graph_batch(model, data.input, data.batch)



        baseline_mse_score += MSE(inputs, targets)
        baseline_ssim_score += SSIM(inputs, targets)
        baseline_pcc_score += PCC(inputs, targets)

        upscaled_mse_score += MSE(outputs, targets)
        upscaled_ssim_score += SSIM(outputs, targets)
        upscaled_pcc_score += PCC(outputs, targets)
        

        total_batches += 1
        
        visualize(inputs, outputs, targets, total_batches, output_path)



    print('Mean Squared Error --- Baseline: {}, Generated: {}'.format(
        (baseline_mse_score/total_batches), (upscaled_mse_score/total_batches)
    ))
    print('SSIM --- Baseline: {}, Generated: {}'.format(
        (baseline_ssim_score/total_batches), (upscaled_ssim_score/total_batches)
    ))
    print('Pearson\'s Correlation Coefficient --- Baseline: {}, Generated: {}'.format(
        (baseline_pcc_score/total_batches), (upscaled_pcc_score/total_batches)
    ))


def generate_chromosomes():
    pass


def run(
        model,
        file_train, 
        file_validation,
        file_test,
        clean_existing_weights=True, 
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
    if clean_existing_weights:
        delete_files(model.weights_dir)

    # Create and clean the output directory 
    output_path = os.path.join(GENERATED_DATA_DIRECTORY, model.model_name)
    if not os.path.exists(output_path):
        create_entire_path_directory(output_path)

    # Step 1: Initialize the model and the optimizer
    if debug: print('Initializing the model parameters')
    # Move model to the defined device
    model.to(model.device)

    if debug: print('Initializing the Optimizer')
    # Initialize the optimizer  TODO: Make this generic as well
    optimizer = model.create_optimizer()

    # Step 2: Load the datasets into host memory
    if debug: print('Loading Training Dataset')
    train_loader = model.load_data(file_train)
    if debug: print('Loading Validation Dataset')
    validation_loader = model.load_data(file_validation)
    if debug: print('Loading Testing Dataset')
    test_loader = model.load_data(file_test)

    # Step 3: Enter the main training loop
    for epoch in tqdm(range(model.hyperparameters['epochs']), 'Training Epochs'):
        training_loss = train(model, train_loader, optimizer, writer)
        validation_loss, mse, ssim, pcc = validate(model, validation_loader)

        writer.add_scalar("Loss/Training",      training_loss,      epoch)
        writer.add_scalar("Loss/Validation",    validation_loss,    epoch)
        writer.add_scalar("MSE/Validation",     mse,                epoch)
        writer.add_scalar("SSIM/Validation",    ssim,               epoch)
        writer.add_scalar("PCC/Validation",     pcc,                epoch)
        torch.save(model.state_dict(), os.path.join(model.weights_dir, '{}-epoch_{}-loss_model'.format(epoch, training_loss)))

    # Step 4: Test the model on testing set
    # Load the best weight
    model.load_weights()
    test(model, test_loader, output_path)

    # # Step 5: Generate chromosome files
    generate_chromosomes()

    # # Step 6: Generate report and clean up
    # #TODO: 

    


