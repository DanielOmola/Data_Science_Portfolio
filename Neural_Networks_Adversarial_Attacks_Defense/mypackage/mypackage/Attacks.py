# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 10:45:01 2021

@author: danie
"""
import torch
import torch.nn as nn

class FastGradientSignMethod:
  """
  #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
  #
  #			Class that implement the Fast Gradient Sign (FGSM) mechanism 
  #
  #					http://www.diva-portal.org/smash/get/diva2:1355328/FULLTEXT01.pdf
  #	
  #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
  """

  def __init__(self, model, eps, criterion):
    self.eps = eps
    self.model = model
    self.criterion = criterion


  # Optimal perturbation calculation
  def compute(self, x, y):
    """
	#######################################################################################
	#
	#	Construct an FGSM adversarial perturbation for an image x 
	#
	#######################################################################################
	"""
	# ############ Perturbation initialization ############
    delta = torch.zeros_like(x, requires_grad=True)
    y_adversarial = self.model(x + delta) # Adversarial label
	
	# ############ Loss between perturbated and true image ############
    loss = self.criterion(y_adversarial, y) # Loss calculation
	
	# ############ Compute gradients (with backpropagation) ############
    loss.backward()
	# ############ return perturbation ############
    return self.eps * delta.grad.detach().sign() # Get the optimal perturbation such as: delta = epsilon * sign(delta)




class ProjectedGradientDescent:
  """
  #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
  #
  #			Class that implement the Projected Gradient Descent(PGD) mechanism  
  #
  #		http://www.diva-portal.org/smash/get/diva2:1355328/FULLTEXT01.pdf
  #	
  #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
  """

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
    """
	#######################################################################################
	#
	#			Construct a PGD adversarial perturbation for an image x
	#
	#######################################################################################
	"""
	# ############ Random initialization of Perturbation (Delta)  ############
    if self.random:
      delta = torch.rand_like(x, requires_grad=True) 
      delta.data = delta.data * 2 * self.eps - self.eps
	  
	# ############ Initialization of Perturbation (Delta) with Zeros ############
    else:
      delta = torch.zeros_like(x, requires_grad=True) # Perturbation (Delta) initialization to zero


    for epoch in range(self.num_iter):
	
      # ############ Create an Adversarial label
      y_adversarial = self.model(x + delta) 
      
      # ############ Loss between perturbated and true image ############
      loss = self.criterion(y_adversarial, y) # Loss calculation
	  
	  # ############ Compute gradients (with backpropagation) ############
      loss.backward() # 
		
	  # ############ Compute final delta (Perturbation) ############
      #              - alpha (learning rate) is a fraction of epsilon
	  #              Update delta through a step size (alpha)
      delta.data = (delta + self.alpha * delta.grad.detach().sign()).clamp(-self.eps, self.eps) 
	  
	  # ############ Reset gradients ####################################
      delta.grad.zero_()
	  
	# ############ return perturbation ############
    return delta