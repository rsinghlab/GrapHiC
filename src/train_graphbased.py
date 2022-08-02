import torch
import torch.nn as nn
from torch.nn import Conv2d
import torch.nn.functional as F
import torch.optim as optim
from src import utils
import numpy as np
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

REDMAP = LinearSegmentedColormap.from_list("bright_red", [(1,1,1),(1,0,0)])

def validation_loss(model, validation_loader, epoch=0):
    validation_loss = 0.0
    num_batches = 0
    for i, (data) in enumerate(validation_loader):
        if i == (len(validation_loader) - 1):
            continue
        
        data = data.to(model.device)
        output = model(data.x, data.edge_index, data.edge_attr, data.batch)
        targets = model.process_graph_batch(data.y, data.batch)
        targets = targets.reshape(targets.shape[0], 1, targets.shape[1], targets.shape[2])
        batch_loss = model.loss(output, targets)
        validation_loss = validation_loss + batch_loss.item()
        num_batches += 1
        
        if i == 0:
            plt.matshow(output[0, 0, :, :].detach().to('cpu').numpy(), cmap=REDMAP)
            plt.savefig('outputs/{}/output_epoch_{}_i_{}.png'.format(model.model_name, epoch, i))
            plt.close()
            plt.matshow(targets[0, 0, :, :].to('cpu').numpy(), cmap=REDMAP)
            plt.savefig('outputs/{}/targets_epoch_{}_i_{}.png'.format(model.model_name, epoch, i))
            plt.close()
        


    return float(validation_loss)/num_batches

def train(model, file_train, file_valid, experiment_name, clean_existing_weights=False, debug=True):
    writer = SummaryWriter('logdir/{}'.format(experiment_name))
    
    # create output directory for the model
    utils.create_entire_path_directory(os.path.join('/home/murtaza/Documents/temp_graphic/GrapHiC/outputs/', model.model_name))
    
    # Clean the existing sets of weights in the model.dir_model directory
    if clean_existing_weights:
        utils.delete_files(model.dir_model)
    
    if debug: print('Initializing the model parameters')
    # Move model to the defined device
    model.to(model.device)

    if debug: print('Initializing the Optimizer')
    # Initialize the optimizer  TODO: Make this generic as well
    optimizer = model.create_optimizer()
    
    # Dataloader function, this is defined by the model
    if debug: print('Loading the Training dataset')
    train_loader = model.load_data(file_train)
    # if debug: print('Loading the Validation dataset')
    validation_loader = model.load_data(file_valid)

    if debug: print('Entering the main training loop')
    # Main training loop
    for epoch in tqdm(range(model.hyperparameters['epochs']), 'Training Epochs'):
        epoch_loss = 0.0
        # For each batch in the training loader we train the model
        num_batches = 0
        for i, (data) in enumerate(train_loader):
            if i == (len(train_loader) - 1):
                continue
            
            data = data.to(model.device)
            optimizer.zero_grad()
            
            output = model(data.x, data.edge_index, data.edge_attr, data.batch)

            targets = model.process_graph_batch(data.y, data.batch)
            targets = targets.reshape(targets.shape[0], 1, targets.shape[1], targets.shape[2])

            batch_loss = model.loss(output, targets)

            batch_loss.backward()
            optimizer.step()
            epoch_loss = epoch_loss + batch_loss
            num_batches += 1

        valid_loss = validation_loss(model, validation_loader, epoch)

        writer.add_scalar("Loss/Training", float(epoch_loss)/num_batches, epoch)
        writer.add_scalar("Loss/Validation", valid_loss, epoch) 
        torch.save(model.state_dict(), os.path.join(model.dir_model, '{}-epoch_{}-loss_model'.format(epoch, valid_loss)))
    writer.flush()
