'''
    This file contains the wrapper scripts for training, validation and testing a provided 
    deep-learning based model.
'''
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.utils import BASE_DIRECTORY, create_entire_path_directory, delete_files, GENERATED_DATA_DIRECTORY, PREDICTED_FILES_DIRECTORY
from src.matrix_operations import process_graph_batch, spreadM, together
from src.evaluations import MSE, SSIM, PCC
from src.visualizations import visualize
from src.npz_to_cool_converter import create_cool_file_from_numpy



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
    model.train()
    
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
    model.eval()
    validation_loss = 0.0

    num_batches = 0

    for i, (data) in enumerate(valid_loader):
        if i == (len(valid_loader) - 1):
            continue
        
        data = data.to(model.device)
        output = model(data)
        targets =  process_graph_batch(model, data.y, data.batch)
        
        batch_loss = model.loss(output, targets)

        validation_loss = validation_loss + batch_loss.item()
    
        
        num_batches += 1
    
    validation_loss = float(validation_loss)/(num_batches*model.hyperparameters['batch_size']) 
    
    return validation_loss


def test(model, file_test, output_path, base, target):
    model.train()
    
    encodings_order = np.load(file_test, allow_pickle=True)['enc_order']
    test_loader = model.load_data(file_test)

    
    baseline_mse_score = 0.0
    baseline_ssim_score = 0.0
    baseline_pcc_score = 0.0


    upscaled_mse_score = 0.0
    upscaled_ssim_score = 0.0
    upscaled_pcc_score = 0.0

    total_batches = 0
    
    input_samples = []
    input_encodings = []
    predicted_samples = []
    target_samples = []
    indices = []

    total_params = sum(
	    param.numel() for param in model.parameters()
    )
    print('Total parameters: {}'.format(total_params))
    
    for data in tqdm(test_loader, desc='Predicting: '):
        data = data.to(model.device)
        outputs = model(data)
        targets = process_graph_batch(model, data.y, data.batch)
        inputs = process_graph_batch(model, data.input, data.batch)
        encodings = process_graph_batch(model, data.x, data.batch)
        
        # We handle pos_idx seperately because its harder to generalize this 
        idx = torch.reshape(data.pos_indx, ((int(data.batch.max()) + 1), 4))
        
        input_samples.append(inputs.to('cpu').numpy())
        input_encodings.append(encodings.to('cpu').numpy())
        
        target_samples.append(targets.to('cpu').numpy())
        predicted_samples.append(outputs.detach().to('cpu').numpy())
        indices.append(idx.to('cpu').numpy())


        baseline_mse_score += MSE(inputs, targets)
        baseline_ssim_score += SSIM(inputs, targets)
        baseline_pcc_score += PCC(inputs, targets)

        upscaled_mse_score += MSE(outputs, targets)
        upscaled_ssim_score += SSIM(outputs, targets)
        upscaled_pcc_score += PCC(outputs, targets)
        

        total_batches += 1
        
        visualize(inputs, outputs, targets, total_batches, os.path.join(output_path, 'visualizations'))


    with open(os.path.join(output_path, 'metrics.dump'), 'a+') as dump_file:
        dump_file.write('Upscaling file: {}'.format(base))
        dump_file.write('Mean Squared Error --- Baseline: {}, Generated: {}\n'.format(
            (baseline_mse_score/total_batches), (upscaled_mse_score/total_batches)
        ))
        dump_file.write('SSIM --- Baseline: {}, Generated: {}\n'.format(
            (baseline_ssim_score/total_batches), (upscaled_ssim_score/total_batches)
        ))
        dump_file.write('Pearson\'s Correlation Coefficient --- Baseline: {}, Generated: {}\n'.format(
            (baseline_pcc_score/total_batches), (upscaled_pcc_score/total_batches)
        ))

    print(PREDICTED_FILES_DIRECTORY, base, target, model.model_name)

    samples_path = os.path.join(PREDICTED_FILES_DIRECTORY, base + '@' + target, model.model_name)
    create_entire_path_directory(samples_path)

    file = os.path.join(samples_path, 'predicted_samples.npz')

    input_samples = np.concatenate(input_samples, axis=0)
    input_encodings = np.concatenate(input_encodings, axis=0)

    predicted_samples = np.concatenate(predicted_samples, axis=0)
    target_samples = np.concatenate(target_samples, axis=0)
    indices = np.concatenate(indices, axis=0).reshape(-1, 4)

    
    print(input_encodings.shape, encodings_order)
    
    np.savez_compressed(
        file, 
        input=input_samples,
        input_encodings=input_encodings, 
        target=target_samples,
        graphic=predicted_samples, 
        index=indices, 
        enc_order=encodings_order
    )
    print('All the predicted samples saved at {}'.format(file))



def save_chromosomes(model, file_test, output_path, base):
    model.eval()

    loader = model.load_data(
        file_test, 
        1,
        False
    )

    result_data = []
    result_inds = []
    
    _, compacts, sizes = data_info(np.load(file_test, allow_pickle=True))
    chroms_path = os.path.join(PREDICTED_FILES_DIRECTORY, model.model_name, base)
    create_entire_path_directory(chroms_path)

    for i, data in enumerate(loader):
        if i == (len(loader) - 1):
            continue
        
        data = data.to(model.device)
        outputs = model(data)
        result_inds.append(data.pos_indx.to('cpu').numpy())
       
        result_data.append(outputs.detach().to('cpu').numpy())
        
    
    result_data = np.concatenate(result_data, axis=0)
    result_inds = np.concatenate(result_inds, axis=0).reshape(-1, 4)
    
    print(sorted(list(np.unique(result_inds[:,0]))))
    
    
    
    predicted = together(result_data, result_inds, tag='Reconstructing: ')

    def save_data_n(key):
        file = os.path.join(chroms_path, f'chr{key}.npz')
        save_data(predicted[key], compacts[key], sizes[key], file)
        from matplotlib.colors import LinearSegmentedColormap
        REDMAP = LinearSegmentedColormap.from_list("bright_red", [(1,1,1),(1,0,0)])

        plt.matshow(predicted[key][:2000, :2000], cmap=REDMAP)
        plt.savefig(os.path.join(chroms_path, 'sample-vis.png'))

    print(f'Saving predicted data as individual chromosome files')

    for key in compacts.keys():
        save_data_n(key)


def plot_loss_curve(training_losses, validation_losses, epochs, output_path):
    plt.plot(epochs, training_losses)
    plt.plot(epochs, validation_losses)
    plt.savefig(os.path.join(output_path, 'loss-curve.png'))
    plt.clf()
    plt.cla()
    plt.close()
    
    


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
    # Weights directory exists
    if not os.path.exists(model.weights_dir):
        create_entire_path_directory(model.weights_dir)
    
    # Clean the existing sets of weights in the model.dir_model directory
    if retrain:
        print("Retraining")
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
    
    epoch_ids = []
    training_losses = []
    validation_losses = []
    
    min_loss = None
    # Step 2: Enter the main training loop if the retrain flag is set
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
            epoch_ids.append(epoch)
            
            
            if min_loss == None:
                print('Updating MinLoss to {}'.format(min_loss))
                min_loss = validation_loss
            
            if validation_loss <= min_loss:
                print('Updating MinLoss to {}'.format(min_loss))
                min_loss = validation_loss
                torch.save(model.state_dict(), os.path.join(model.weights_dir, '{}-epoch_{}-loss_model'.format(0, 0)))
            
        plot_loss_curve(training_losses, validation_losses, epoch_ids, output_path)
    
    
    # Load the testing loader
    if debug: print('Loading Testing Dataset')
    
    # Step 4: Test the model on testing set
    # Load the best weight
    model.load_weights()
    test(model, file_test, output_path, base, target)

    # Step 5: we save all the samples instead of the chromosomes for easier evaluation
    save_chromosomes(model, file_test, output_path, base)

    



