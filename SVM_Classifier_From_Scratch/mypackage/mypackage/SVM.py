# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 10:45:01 2021

@author: daniel omola
"""

import numpy as np

class SVM:
  """
  # ######################################################################################
  #
  #			Class for building a simple SVM Classifier (linear kernel)
  #		hyperparameters : learning rate (lr), lambda(lambda_param), number of iterations (n_iters)
  # 
  # ######################################################################################
  """
  def __init__( self,
                 lr = 0.01,
                 lambda_param = 0.01,
                 n_iters = 10000):
      
      self.lr = lr
      self.lambda_param = lambda_param
      self.n_iters= n_iters
      self.w = None
      self.b = None


  def fit(self,X,y):
    """
	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
	#
	#						Function for training the SVM
	#
	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
	"""	
    y_=np.where(y<=0,-1,1)
    n_samples,n_features =X.shape
    
    # ############### Instanciate W and b with 0
    self.w = np.zeros(n_features)
    self.b = 0

    W,b=self.w,self.b
    self.history = [(W,b)]
    width = 0
    
    M = [np.array([1,1]),np.array([-1,1]),np.array([-1,-1]),np.array([1,-1])]
    
    for m in M :

        self.w,self.b= self.w * m ,self.b

    	# ############### Gradient descent optimization ################
        #for _ in range(self.n_iters):
        for _ in range(self.n_iters):
            
            for idx, x_i in enumerate(X):
                
    			# ###### Check if the point is well classified or missclassified  ################
                condition=y_[idx]*(np.dot(x_i,self.w) - self.b) >= 1
                
                if condition:
    				# ###### W update for well classified points
                    self.w-= self.lr*(2 * self.lambda_param * self.w)
                else :
    				# ###### W and b update for misclassified points
                    self.w -= self.lr * ( 2 * self.lambda_param * self.w - np.dot(y_[idx],x_i))
                    self.b -= self.lr * y_[idx]
    				
                self.history.append((self.w,self.b))

        if 1/self.w.dot(self.w) > width:
            width = 1/self.w.dot(self.w) 
            self.w_star, self.b_star = self.w,self.b
            
        else :
            self.w_star, self.b_star = self.w,self.b
        
        
            
  def predict(self,X):
    """
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    #
    #						Function for predicting the class of of a data point
    #
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    """	
    linear_output = np.sign(np.dot(X,self.w))
    return linear_output

