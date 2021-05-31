# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 10:45:01 2021

@author: danie
"""
import torch
import torch.nn as nn

class FastGradientSignMethod:
  """Class to compute the optimal perturbation of an image by FGSM mechanism"""


  # Default constructor
  def __init__(self, model, eps, criterion):
    self.eps = eps
    self.model = model
    self.criterion = criterion


  # Optimal perturbation calculation
  def compute(self, x, y):
    """ Construct FGSM adversarial perturbation for examples x"""

    delta = torch.zeros_like(x, requires_grad=True) # Perturbation initialization
    y_adversarial = self.model(x + delta) # Adversarial label

    loss = self.criterion(y_adversarial, y) # Loss calculation
    loss.backward() # Compute gradient


    return self.eps * delta.grad.detach().sign() # Get the optimal perturbation such as: delta = epsilon * sign(delta)




class ProjectedGradientDescent:
  """Class to compute the optimal perturbation of an image by PGD mechanism"""


  # Default constructor
  def __init__(self, model, eps, alpha, num_iter, random, criterion):
    self.model = model
    self.eps = eps
    self.alpha = alpha
    self.num_iter = num_iter
    self.random = random
    self.criterion = criterion
    

  
  # Optimal perturbation calculation
  def compute(self, x, y):
    """ Construct PGD adversarial pertubration on the examples x"""

    if self.random:
      delta = torch.rand_like(x, requires_grad=True) # Perturbation (Delta) random initialization
      delta.data = delta.data * 2 * self.eps - self.eps

    else:
      delta = torch.zeros_like(x, requires_grad=True) # Perturbation (Delta) initialization to zero

    #print(delta)

    for epoch in range(self.num_iter):

      y_adversarial = self.model(x + delta) # Adversarial label
      

      loss = self.criterion(y_adversarial, y) # Loss calculation
      loss.backward() # Compute gradient (backpropagation)

      # alpha est une fraction d'epsilon, car il est logique de choisir le Learning Rate <= epsilon
      delta.data = (delta + self.alpha * delta.grad.detach().sign()).clamp(-self.eps, self.eps) # Update delta through a step size (alpha)

      delta.grad.zero_() # Reset gradients


    #return delta.detach
    return delta