# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 10:45:01 2021

@author: daniel omola
"""

import numpy as np
#import pandas as pd

class LinearRegression:
    """
    # ######################################################################################
    #
    #    Class for building LinearRegression Model with gradient descent optimization
    #
    #    hyperparameters : 
    #                    learning rate (lr) float, number of iterations (num_iter) float,
    #                    fit_intercept (boolean), verbose (boolean)
    # 
    # ######################################################################################
    """
    def __init__(self, lr=0.01, num_iter=100000, fit_intercept=True, verbose=False):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose
    
    def add_intercept(self, X):
        """
    	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    	#						Function for adding intercept
    	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    	"""	
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
        
    
    def loss(self,X,h, y):
        """
    	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    	#						Function for loss calculation
    	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    	"""	
        h = np.dot(X,self.theta.T).reshape(X.shape[0],1)
        y = y.reshape(y.shape[0],1)
        return ((y-h)**2).sum()
    
    def fit(self, X, y):
        """
    	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    	#						Function for training
    	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    	"""	
        if self.fit_intercept:
            X = self.add_intercept(X)

        y = y.reshape(y.shape[0],1)
        
        # ########### random weights initialization
        self.theta = np.random.rand(X.shape[1])
        
        # ########### Gradient Descent process
        for i in range(self.num_iter):

            # ########### predictions with parameters (theta)
            h = np.dot(X, self.theta.T).reshape(X.shape[0],1)
            
            # ########### compute gradients
            gradient = -2*np.dot(X.T,(y-h))#-2*(y-h) / y.size
            self.theta -= self.lr * gradient.reshape(-1)
        print("\n########### LINEAR REGRESSION : RESULT FROM GRADIENT DESCENT OPTIMIZATION ###########\n")    
        print(f'\tIntercept: {self.theta[0]}\n\tSlope: {self.theta[1]}')    
        print(f'\tTotal loss: {self.loss(X,h, y)} \n')    
        
                
        return self.theta

    def predict(self, X):
        """
    	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    	#						Function for prediction
    	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    	"""	
        if self.fit_intercept:
            X = self.add_intercept(X)
        return np.dot(X, self.theta)


# ////////////////////////////////////////////////////////////////////////////////////////////////////////
    
class RidgeRegression(LinearRegression):
    """
    # ######################################################################################
    #
    #   Class for building RidgeRegression Model with gradient descent optimization
    #
    #   hyperparameters : 
    #                   learning rate (lr) float, number of iterations (num_iter) float,
    #                   penality (integer/float), fit_intercept (boolean), verbose (boolean)
    # 
    # ######################################################################################
    """
    def __init__(self, lr=0.01, num_iter=100000, penality = 1, fit_intercept=True, verbose=False):
        LinearRegression.__init__(self, lr=0.01, num_iter=100000, fit_intercept=True, verbose=False)
        self.lr = lr
        self.num_iter = num_iter
        self.penality = penality
        self.fit_intercept = fit_intercept
        self.verbose = verbose
       
    def loss(self,X,h, y):
        """
    	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    	#						Function for loss calculation
    	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    	"""
        #h = np.dot(x,theta.T).reshape(X.shape[0],1)
        #y = y.reshape(y.shape[0],1)
        #print(self.penality)
        #print(LinearRegression.loss(self,X,h, y)+(self.penality*self.theta**2).sum())
        return ((y-h)**2).sum() + (self.penality*self.theta**2).sum()#/ (2*y.size)
    
    def fit(self, X, y):
        """
    	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    	#						Function for training
    	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    	"""
        if self.fit_intercept:
            X = self.add_intercept(X)
        
        # weights initialization
        self.theta = np.random.rand(X.shape[1])
        
        y = y.reshape(y.shape[0],1)
        
        for i in range(self.num_iter):
            h = np.dot(X, self.theta.T).reshape(X.shape[0],1)
            gradient = -2*(np.dot(X.T, (y - h))) + 2*self.penality * self.theta.reshape(X.shape[1],1)/ y.size#/ y.size
            #gradient =  - self.lr * (1/y.size)* (  (X.T @ (h-y)) + self.penality * self.theta )
            #print(gradient)
            #print(-2*(np.dot(X.T, (y - h))))
            #print(2*self.penality* self.theta.reshape(X.shape[1],1))
            #print(self.theta)
            self.theta -= self.lr * gradient.reshape(-1)
            
            if(self.verbose == True and i % 2 == 0):
                h = np.dot(X, self.theta)

        print("\n########### RIDGE REGRESSION : RESULT FROM GRADIENT DESCENT OPTIMIZATION ###########\n") 
        print(f'\tPenalty: {self.penality}')
        print(f'\tIntercept: {self.theta[0]}\n\tSlope: {self.theta[1]}')    
        print(f'\tTotal loss: {self.loss(X,h, y)} \n')
        return self.theta

    def predict(self, X):
        """
    	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    	#						Function for prediction
    	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    	"""	
        if self.fit_intercept:
            X = self.__add_intercept(X)
        return np.dot(X, self.theta)

# ////////////////////////////////////////////////////////////////////////////////////////////////////////

class LassoRegression(LinearRegression):
    """
    # ######################################################################################
    #
    #   Class for building LassoRegression Model with gradient descent optimization
    #
    #   hyperparameters : 
    #                   learning rate (lr) float, number of iterations (num_iter) float,
    #                   penality (integer/float), fit_intercept (boolean), verbose (boolean)
    # 
    # ######################################################################################
    """
    def __init__(self, lr=0.01, num_iter=100000, penality = 1, fit_intercept=True, verbose=False):
        LinearRegression.__init__(self, lr=0.01, num_iter=100000, fit_intercept=True, verbose=False)
        self.lr = lr
        self.num_iter = num_iter
        self.penality = penality
        self.fit_intercept = fit_intercept
        self.verbose = verbose
      
    def loss(self,X,h, y):
        #print(LinearRegression.loss(self,X,h, y)+abs((self.penality*self.theta)).sum())
        return ((y-h)**2).sum() + abs((self.penality*self.theta)).sum()
    
    def fit(self, X, y):
        """
    	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    	#						Function for training
    	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    	"""        
        if self.fit_intercept:
            X = self.add_intercept(X)
        self.theta = np.random.rand(X.shape[1])
        
        y = y.reshape(y.shape[0],1)
        
        for i in range(self.num_iter):
            h = np.dot(X, self.theta.T).reshape(X.shape[0],1)
            #gradient = -2*np.dot(X.T, (y - h)) + self.penality* self.theta.sum()/ y.size
            gradient = -2*np.dot(X.T, (y - h)) + self.penality * np.sign(self.theta).reshape(X.shape[1],1)/ y.size
            self.theta -= self.lr * gradient.reshape(-1)
            
            if(self.verbose == True and i % 2 == 0):
                h = np.dot(X, self.theta)

        print("\n########### LASSO REGRESSION : RESULT FROM GRADIENT DESCENT OPTIMIZATION ###########\n") 
        print(f'\tPenalty: {self.penality}')
        print(f'\tIntercept: {self.theta[0]}\n\tSlope: {self.theta[1]}')    
        print(f'\tTotal loss: {self.loss(X,h, y)} \n')
        return self.theta

    def predict(self, X):
        """
    	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    	#						Function for prediction
    	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    	"""	
        if self.fit_intercept:
            X = self.__add_intercept(X)
        return np.dot(X, self.theta)
