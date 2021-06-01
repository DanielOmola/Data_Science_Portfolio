# -*- coding: utf-8 -*-
"""
Created on Thu May  6 12:52:04 2021

@author: danie
"""

import numpy as np

class LDA:
    """
    # ######################################################################################
    #
    #   Class for building LDA algorithm
    #
    #   hyperparameters : n_components (number of LDAs to be returned)
    #                   
    # ######################################################################################
    """
    
    def __init__(self,n_components):
        self.n_components= n_components
        self.linear_discriminants = None


    def fit(self, X,y):
        """
    	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    	#       Function for training 
    	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    	"""	
        # ########### Get number of features
        n_features = X.shape[1]
        
        # ########### Get number of class
        class_labels = np.unique(y)
        
        
        # ########### Get the global mean
        # ########### of each dimension
        mean_overall = np.mean(X, axis=0)
        
        # S_W, S_B
        S_W =np.zeros((n_features,n_features))
        S_B =np.zeros((n_features,n_features))
        
        # ########### loop through class
        for c in class_labels:
            
          # ########### get data points
          # ########### belonging to the class
          X_c = X[y==c]
        
          # ########### Get the mean of each
          # ###########  dimension inside the class
          mean_c = np.mean(X_c, axis=0)
                   
          # ########### Compute the variances of
          # ########### features inside the class
          S_W += (X_c - mean_c).T.dot(X_c - mean_c)
        
          # ##### get the number of element
          # ##### inside the class
          n_c = X_c.shape[0]
            
          # #### compute the difference between 
          # #### the means of class and the global means
          mean_diff = (mean_c-mean_overall).reshape(n_features,1)
                    
          # #### compute the square of the difference between 
          # #### the class means and the global means
          S_B += n_c * (mean_diff).dot(mean_diff.T)
        
        # ####################################
        #  divivide the sum of all class
        #  square mean differences 
        #  by the sum of all class variances
        #  to get a matrix A
        # ####################################
        A = np.linalg.inv(S_W).dot(S_B)
        
        # ####################################
        #  get the eigenvalues,eigenvectors
        #  of matrix A
        #  eigenvectors are vector that maximise
        #  the difference between class means
        #  and global mean while minimizing variation
        #  inside each class
        eigenvalues,eigenvectors = np.linalg.eig(A) 

        # #### get the n most important LDAs
        # #### (n first eigenvectors)
        eigenvectors= eigenvectors.T
        idxs = np.argsort(abs(eigenvalues))[::-1]
        eigenvalues= eigenvalues[idxs]
        eigenvectors= eigenvectors[idxs]

        
        
        # ### store n first LDAs
        self.linear_discriminants = eigenvectors[0:self.n_components]
 
    def transform(self,X):
        """
    	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    	#       Project the data to the new dimensions (LDAs)
    	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    	"""	
        return np.dot(X,self.linear_discriminants.T)