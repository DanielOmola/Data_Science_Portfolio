# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 10:45:01 2021

@author: daniel omola
"""

from statsmodels.tsa.stattools import adfuller,pacf
import numpy as np

sig_test = lambda tau_h, T : np.abs(tau_h)>2.58/np.sqrt(T)

def check_stationarity(data,col='close_spx'):
	"""
	####################################################################################
	#
	#				  Run an Augmented Dickey Fuller Test
	#				to check Stationarity of data provided
	#
	####################################################################################
	"""
	X = data[col].values
	result = adfuller(X)
	print('ADF Statistic: %f\n' % result[0])
	print('p-value: %f\n' % result[1])
	print('Critical Values:\n')
	for key, value in result[4].items():
		print('\t%s: %.3f' % (key, value))

		
def get_number_steps(data,col):
	"""
	####################################################################################
	#
	#				  Get the recommended number of steps
					based on Partial Autocorrelation Function (PACF)
	#
	####################################################################################
	"""
	T=len(data[col])
	partial_acf = pacf(data[col])
	for i in range(len(partial_acf)):
		if sig_test(partial_acf[i],T)==False:
			n_steps=i-1
			print('=> Recommanded number of steps for %s : %d'%(col,n_steps))
			break