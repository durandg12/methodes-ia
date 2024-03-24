import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import time
import matplotlib.pyplot as plt
import numpy as np
from model import MaskedConv
import streamlit as st



def train_loop(architecture, input_channels, output_channels,  h, residual_block, end_activation, p, optimizer, criterion, device, lr, trainloader,  epochs,mean,std):
    """
        Function to train our models.

        Parameters
        ----------
        architecture : function
            The architecture of the neural network to be trained
        in_channels : int
            Number of input channels, 1 (black & white) or 3 (RGB).
        out_channels : int
            The number of classes in our multinomial distribution (256), the value a pixel can take.
        h : int
            Number of channels in the layers.
        residual_block : function
            The residual block to be used in our architecture, which defines the model.
        end_activation : torch.nn
            The activation function to be used at the end of our architecture.
        p : int
            Number of residual blocks.
            5 if PixelCNN
            6 if PixelRNN
        optimizer : torch.optim.optimizer.Optimizer
            The optimizer to use
        criterion : nn.modules._Loss
            The loss function to minimize
        device : string
            The device to use.
        lr : float
            The learning rate
        trainloader : torch.utils.data.DataLoader
            The train DataLoader
        n_epochs : int, default = 10
            Number of epochs to do

        Returns
        -------
        model : nn.Module
            The trained model.
    """
    
    #Build the model
    model = architecture(input_channels, output_channels, h, residual_block, end_activation).to(device = device)
    optimizer = optimizer(model.parameters(), lr = lr)
    st.write("l'entraînement a commencé")
    progress_bar = st.progress(0)  # Initialize the progress bar
    for epoch in range(epochs ):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, (x,y) in enumerate(trainloader):
        # assign a device for the inputs
            inputs = x.to(device)
      
            if input_channels == 1 :
                label = torch.round((x.to(device) * std[0] + mean[0]) * 255) #denormalize to obtain original labels (0-255)
                labels = torch.reshape(label, (16, 1, 28, 28)).squeeze().long()
            else :
                denorm = transforms.Normalize(mean = [-0.4915/0.2470, -0.4823/0.2435, -0.4468/0.2616], std = [1/0.2470, 1/0.2435, 1/0.2616])
                label = torch.round(denorm(x.to(device))*255).to(torch.int) #denormalize to obtain original labels (0-255)
                labels = label.long()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(inputs, p = p) # dim = (16,256,28,28) or (16, 3*256, 32, 32), we need to go back to 3d tensor for the loss function
            loss = criterion(outputs, labels)

            # backward
            loss.backward()

            # optimize
            optimizer.step()

            running_loss += loss.item()
            progress = (epoch + 1) / epochs
        progress_bar.progress(progress)  # Update the progress bar
        
    st.write(f'Epoch {epoch + 1} / {epochs} | Loss: {running_loss / len(trainloader)}')
    running_loss = 0.0
    return(model)