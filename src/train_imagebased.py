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


def validation_loss(model, validation_loader):
    validation_loss = 0.0
    num_batches = 0
    for i, (data) in enumerate(validation_loader):
        x = data[0]
        targets = data[1]
        
        x = x.to(model.device)
        targets = targets.to(model.device)
        
        output = model(x)
        batch_loss = model.loss(output, targets)
        validation_loss = validation_loss + batch_loss.item()
        num_batches += 1

    return float(validation_loss)/num_batches

def train(model, file_train, file_valid, experiment_name, clean_existing_weights=False, debug=True):
    writer = SummaryWriter('logdir/{}'.format(experiment_name))

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
            x = data[0]
            targets = data[1]
            
            x = x.to(model.device)
            targets = targets.to(model.device)
            optimizer.zero_grad()
            
            output = model(x)

            batch_loss = model.loss(output, targets)

            batch_loss.backward()
            optimizer.step()
            epoch_loss = epoch_loss + batch_loss.item()
            num_batches += 1

        valid_loss = validation_loss(model, validation_loader)
        writer.add_scalar("Loss/Training", float(epoch_loss)/num_batches, epoch)
        writer.add_scalar("Loss/Validation", valid_loss, epoch) 
        torch.save(model.state_dict(), os.path.join(model.dir_model, '{}-epoch_{}-loss_model'.format(epoch, valid_loss)))
    
    writer.flush()
