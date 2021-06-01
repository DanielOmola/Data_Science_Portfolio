# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 10:45:01 2021

@author: daniel omola
"""

import numpy as np
#import pandas as pd
import tensorflow as tf
tf.__version__


class LogisticRegressionNP:
    """
    # ######################################################################################
    #
    #   Class for building Logistic Regression Model with gradient descent optimization
    #
    #   hyperparameters and parameters : 
    #                   learning rate (lr) float, number of iterations (num_iter) float,
    #                   penality (integer/float), fit_intercept (boolean), verbose (boolean)
    # 
    # ######################################################################################
    """
    def __init__(self, lr=0.01,threshold=0.5, num_iter=100000, fit_intercept=True, verbose=False):
        self.threshold = threshold
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose
    
    def __add_intercept(self, X):
        """
    	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    	#						Function for adding intercept
    	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    	"""	
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def __sigmoid(self, z):
        """
    	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    	#						Sigmoid/Logistic Function
    	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    	"""	
        return 1 / (1 + np.exp(-z))
    
    def __loss(self, h, y):
        """
    	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    	#						Function for loss calculation
    	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    	"""	
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    def fit(self, X, y):
        """
    	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    	#						Function for training
    	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    	"""	
        if self.fit_intercept:
            X = self.__add_intercept(X)
        
        # ########### random weights initialization
        self.theta = np.random.rand(X.shape[1])#np.zeros(X.shape[1])
        
        # ########### Gradient Descent process
        for i in range(self.num_iter):
            
            # ########### linear combinaison of features with parameters (theta)
            z = np.dot(X, self.theta)
            
            # ########### pass linear combinaison to sigmoid function for non linearity
            h = self.__sigmoid(z)
            
            # ########### compute gradients
            gradient = np.dot(X.T, (h - y)) / y.size
            
            # ########### update parameters
            self.theta -= self.lr * gradient
            
            if(self.verbose == True and i % 10000 == 0):
                z = np.dot(X, self.theta)
                h = self.__sigmoid(z)
                print(f'loss: {self.__loss(h, y)} \t')
                
        print("\n########### LOGISTIC REGRESSION : TRAINING RESULT ###########\n") 
        print(f'\t\t\tFinal Loss: {self.__loss(h, y)} \t')
        print(f'\t\t\tFinal Accuracy: {self.accuracy(X, y)} \t')
        print(f'\n\t\t\t- Threshold: {self.threshold}\n\t\t\t- Intercept: {self.theta[0]}\n\t\t\t- b1: {self.theta[1]}\n\t\t\t- b2: {self.theta[2]}')  
        return self.theta

    def predict_prob(self, X):
        """
    	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    	#		Function for probability of belonging to class 1
    	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    	"""	
        if self.fit_intercept:
            X = self.__add_intercept(X)
        return self.__sigmoid(np.dot(X, self.theta))
    
    def predict(self, X):
        """
    	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    	#						Function for prediction (input : threshold )
    	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    	"""
        return self.predict_prob(X) >= self.threshold

    def accuracy(self, X,y):
        """
    	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    	#						Function for accuracy calculation
    	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    	"""
        preds = self.predict(X[:,1:])
        return (preds == y).mean()*100
		
class LogisticRegressionTF:
    """
    # ######################################################################################
    #
    #   Class for building Logistic Regression Model with gradient descent optimization
    #
    #   hyperparameters and parameters : 
    #                   learning rate (lr) float, number of iterations (num_iter) float,
    #                   penality (integer/float), fit_intercept (boolean), verbose (boolean)
    # 
    # ######################################################################################
    """
    def __init__(self,lr = 0.01,threshold=.5,epochs=500,verbose=False):
        self.lr = lr
        self.epochs = epochs
        self.threshold = threshold
        self.verbose = verbose
        
    def fit(self,X,Y):
        """
    	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    	#						Function for building and training the model
    	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    	"""	
        tf.random.set_seed(1234)
        
        # ###########  Build the model ########### 

        model = tf.keras.Sequential([

                               tf.keras.layers.Dense(units = 1,
                                                     input_shape=[X.shape[1]],
                                                     activation='sigmoid')])

        # ########### Compile and Fit the Model ###########

        # ---------- Chose an Optimiser
        # optimizer=tf.keras.optimizers.SGD(learning_rate=self.lr,momentum=0.0, nesterov=False)
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01)
        

        # ---------- Compile the model
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'])

        # --------------- Fit the model
        model.fit(X,Y,epochs=self.epochs,verbose=self.verbose)
        self.model = model
        
    def predict(self,X):
        """
    	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    	#						Function for prediction
    	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    	"""	
        return ((self.model.predict(X)>= self.threshold)*1).reshape(-1)
