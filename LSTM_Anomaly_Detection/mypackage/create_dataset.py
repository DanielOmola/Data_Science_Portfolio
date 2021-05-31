# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 10:45:01 2021

@author: daniel omola
"""


#import pandas as pd
import numpy as np


def create_datasetX(X, time_steps=1):
	"""
	####################################################################################
	#
	#		Create X (features) overlaping sequences based on a number of steps
	#
	####################################################################################
	"""
	Xs = []
	for i in range(len(X) - time_steps):
		v = X.iloc[i:(i + time_steps)].values
		Xs.append(v)
	return np.array(Xs)


	
def create_datasety(y, time_steps=1):
	"""
	####################################################################################
	#
	#		Create Y (target) sequences based on a number of steps
	#
	####################################################################################
	"""
	ys = []
	for i in range(len(y) - time_steps):
		w = y.iloc[i + time_steps].values
		ys.append(w)
	return np.array(ys)


	
def get_data_sequences(train,test,col='returns_spx',TIME_STEPS = 30):
	"""
	####################################################################################
	#
	#		Create X (features) and Y (target) sequences
	#
	####################################################################################
	"""
	X_train = create_datasetX(train[[col]],TIME_STEPS )
	X_test = create_datasetX(test[[col]],TIME_STEPS )
	y_train = create_datasety(train[[col]],TIME_STEPS )
	y_test = create_datasety(test[[col]],TIME_STEPS )
	return X_train, X_test, y_train, y_test
