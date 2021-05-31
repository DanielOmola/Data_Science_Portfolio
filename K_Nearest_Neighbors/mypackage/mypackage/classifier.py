# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 10:45:01 2021

@author: daniel omola
"""

import numpy as np
#import pandas as pd
from collections import Counter

def optimal_k(X,y):
    """
    # ######################################
    #   Function helping find optimal k                  
    # ######################################
    """
    range_k = range(2,50)
    acc_k = [accuracy(X,y,KNN(k=i).fit(X,y)) for i in range_k]
    best = max(acc_k)
    k = range_k[acc_k.index(best)]
    return k
	
def accuracy(X,y,model):
    """
    # ######################################
    #   Function for accuracy calculation                  
    # ######################################
    """
    predictions = model.predict(X)
    return np.sum(predictions == y)/len(y)*100
	
	
def euclidean_distance(x1,x2):
    """
    # ######################################
    #   Function computing distance                 
    # ######################################
    """
    return np.sqrt(np.sum((x1-x2)**2))

class KNN:
    """
    # ######################################################################################
    #
    #   Class for building KNN Classifier
    #
    #   hyperparameters : k number of neighbors
    #                   
    # ######################################################################################
    """
    def __init__(self,k=3):
        self.k = k
  
    def fit(self,X,y):
        """
    	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    	#       Function for training (just load the train set for future prediction)
    	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    	"""	
        self.X_train = X
        self.y_train = y
        return self

    def predict(self,X):
        """
    	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    	#						Function for prediction
    	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    	"""	
        predicted_label = [self._predicted(x) for x in X]
        return np.array(predicted_label)
  
    def _predicted(self,x):
        """
    	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    	#						Function for single prediction
    	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    	"""	
        # ########### Compute the distances with training data
        distances=[euclidean_distance(x,x_train) for x_train in self.X_train]

        # ########### get k nearest labels/class
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # ########### Majority vote (most common class win)
        most_common = Counter(k_nearest_labels).most_common(1)
        
        return most_common[0][0]
		

