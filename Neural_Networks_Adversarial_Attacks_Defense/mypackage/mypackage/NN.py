# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 09:18:44 2021

@author: Daniel Omola
"""

import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
import math		


# Define the target device (CPU or GPU) to manage tensors
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(device)


# ///////////////////////////////// Class use for building a 3 layers fully connected NN  /////////////////////////////// 

# define a neural network with 3 fully connected layers
class FullyConnectedModel(torch.nn.Module):
  """
  #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
  #
  #						Class for building a Linear Neural Network 
  #	 						with 3 Fully Connected layers 
  #
  #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
  """

  # Default constructor
  def __init__(self, input_dim, output_dim, hidden_dim):

    super(FullyConnectedModel, self).__init__()
    
    self.fc1 = nn.Linear(input_dim, hidden_dim) # Input layer
    self.fc2 = nn.Linear(hidden_dim, hidden_dim)
    self.fc3 = nn.Linear(hidden_dim, hidden_dim)
    self.fc_out = nn.Linear(hidden_dim, output_dim) # Output layer
    

  # Forward pass
  def forward(self, x):
    """
	#######################################################################################
	#
	#						Function for Forward pass performing
	#
	#######################################################################################
	"""	    

    x = x.view(x.shape[0], -1) # Reshape the input in order to make it flatten

    x = F.relu(self.fc1(x))
    
    x = F.relu(self.fc2(x))

    x = F.relu(self.fc3(x))

    output = self.fc_out(x)

    return output
	
	
	
# ///////////////////////////////// Class use for building a CONV 2D NN  /////////////////////////////// 


class ConvModel(torch.nn.Module):
  """
  #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
  #
  #				Class to build a Neural Network with convolutional model with: 
  #	 						- 2 convolution, 
  #							- 2 max pooling layer,
  #						- 3 fully connected layer
  #				Result : 2x (conv -> max pooling -> relu) -> 2x (fc -> relu) -> fc
  #
  #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
  """	  
# convolutional model with 2 convolution, 2 max pooling layer, 3 fully connected layer
# the model should be: 2x (conv -> max pooling -> relu) -> 2x (fc -> relu) -> fc

  # Default constructor
  def __init__(self, input_dim, output_dim, hidden_dim, num_filters, kernel_size, stride, padding, pool_size):

    super(ConvModel, self).__init__()
    
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=kernel_size, stride=stride, padding=padding) # Conv1
    self.conv2 = nn.Conv2d(in_channels=num_filters, out_channels=num_filters*2, kernel_size=kernel_size, stride=stride, padding=padding) # Conv2
    
    self.max_pool_1 = nn.MaxPool2d(kernel_size=pool_size) # Max Pooling
    self.max_pool_2 = nn.MaxPool2d(kernel_size=pool_size) # Max Pooling
    #self.conv2_drop = nn.Dropout2d() # Dropout
  

    self.fc1 = nn.Linear(64 * 7 * 7, hidden_dim)
    #self.fc1 = nn.Linear(, hidden_dim) # Fully Connected 1 (input calculation formula: https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_convolutional_neuralnetwork/)
    self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # Fully Connected 2
    self.fc_out = nn.Linear(hidden_dim, output_dim)  # Fully Connected 3 (output)
   

  def forward(self, x):
    """
	#######################################################################################
	#
	#						Function for Forward pass performing
	#
	#######################################################################################
	"""    

    x = self.conv1(x)

    x = F.relu( self.max_pool_1(x) )

    x = self.conv2(x)

    x = F.relu( self.max_pool_2(x) )

    x = x.view(x.shape[0], -1) # Reshape the input in order to make it flat

    x = F.relu(self.fc1(x))

    x = F.relu(self.fc2(x))

    output = F.relu(self.fc_out(x))

    return output



def train_model(model, criterion, optimizer, loader, epochs, attack=None):

  """
  #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
  #
  #						Function to train a Model
  #
  #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
  """

  for epoch in range(epochs):
    total_loss = 0.0
    accuracy = 0.0

    for batch_n, (images, labels) in enumerate(loader):
      images = images.to(device) # get features
      labels = labels.to(device) # get labels

      # ############ Predictions with attack ############
      if attack != None:
        delta = attack.compute(images, labels) # Get the perturbation
        prediction = model(images + delta) # forward pass (predictions) with a perturbation

      # ############ Predictions without Attack ############
      else: 
        prediction = model(images) # forward pass (predictions)

		
      # ######### Loss calculation (compute softmax -> loss) ###
      loss = criterion(prediction, labels) 

      if optimizer:
        optimizer.zero_grad() # reset the gradients
        loss.backward() # Compute gradients
        optimizer.step() # Parameters update

      # ############ Final Prediction ############
      _, predicted = torch.max(prediction.data, 1)

      # ############ Performance Metrics incrementation ############	  
      total_loss += loss.item() * images.shape[0] # Compute global Loss
      accuracy += (predicted == labels).sum() # Compute global Accuracy

    # ############ Final Performance Metrics ############	
    loss = total_loss/len(loader.dataset)
    accuracy = accuracy/len(loader.dataset)

    print(f'\tEpoch: {epoch+1}/{epochs} \t| Loss: {loss:.4f} \t | Accuracy: {accuracy:.2%}')
  
  return loss, accuracy  




def eval_model(model, loader, criterion,attack=None):

  """
  #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
  #
  #						Function to evaluate your model with specific loader (Test set)
  #
  #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
  """
  
  total_loss = 0.0
  accuracy = 0.0

  """with torch.no_grad():""" # Disable the Gradient calculation

  for n_batch, (images, labels) in enumerate(loader):

    images = images.to(device) # Get Features/Images
    labels = labels.to(device) # Get Labels


    # ############ Predictions with attack ############
    if attack != None:
      delta = attack.compute(images, labels) # Get the perturbation      
      prediction = model(images + delta) # forward pass (predictions) with a perturbation
    

    # ############ Predictions without Attack ############
    else: 
      prediction = model(images) # forward pass (predictions)

    # ############ Loss Calculation ############
    loss = criterion(prediction, labels) # Loss calculation (compute softmax -> loss)
	
    # ############ Final Prediction ############
    _, predicted = torch.max(prediction.data, 1)

    # ############ Performance Metrics incrementation ############
    total_loss += loss.item() * images.shape[0] # Compute global Loss
    accuracy += (predicted == labels).sum() # Compute global Accuracy

  # ############ Final Performance Metrics ############
  loss = total_loss/len(loader.dataset)
  accuracy = accuracy/len(loader.dataset)

  if attack != None:
    return loss, accuracy.item(), delta
  
  else:
    return loss, accuracy.item()