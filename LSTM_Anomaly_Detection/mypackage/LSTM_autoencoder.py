# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 10:45:01 2021

@author: daniel omola
"""

import keras
import tensorflow as tf
import tensorflow.compat.v1.keras.backend as K

session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
K.set_session(sess)

from keras.models import Sequential
#from keras.layers import Dense

import numpy as np
import pandas as pd


#import plotly
#import plotly.express as px
import plotly.graph_objects as go

import sys
from .mypackage import create_dataset as cd




class LSTM_autoencoder :
  """
  #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
  #
  #						Class for building LSTM Autoencoder
  # 
  #							1 LSTM layer for encoding,
  #							1 LSTM layer for decoding
  #
  #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
  """
  def __init__(self, train,test):
    self.train = train
    self.test = test

  def build_model(self,X_train,
					nb_hidden_units = 128,
					droupout_rate = 0.2,
					optimizer = tf.keras.optimizers.Adam(
                                              learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True,
                                              name='Adam'), verbose=True):
											  											  
    """
	#######################################################################################
	#
	#						Function for building LSTM Autoencoder Model
	# 							output : LSTM Autoencoder Model
	#
	#######################################################################################
	"""											  

    model = Sequential()

    # ############ first LSTM layer: encoder ########################
	
    model.add(keras.layers.LSTM(
								units=nb_hidden_units,
								input_shape=(X_train.shape[1],
								X_train.shape[2])))

								
    # ######################## add dropout ##########################
    model.add(keras.layers.Dropout(rate=droupout_rate))

	
    # ######################## add a "RepeatVector" #################
    model.add(keras.layers.RepeatVector(n=X_train.shape[1]))

	
    # ######################## last LSTM layer: decoder ##############
    model.add(keras.layers.LSTM(units=nb_hidden_units, return_sequences=True))

	
    # ######################## add dropout ###########################
    model.add(keras.layers.Dropout(rate=droupout_rate))

	
    # ######################## add a "TimeDistributed" ################
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(units=X_train.shape[2])))
	

    # ######################## print summary ########################
    if verbose:
      print(model.summary())
	  

    # ######################## compile the Model ########################
    model.compile(loss='mae', optimizer=optimizer)

    self.model = model


  def train_it(self,X_train, y_train,nb_epochs=10,batch_size=32,verbose=True):
    """
	#######################################################################################
	#
	#						Function for Model training
	# 
	#######################################################################################
	"""	
    self.history = self.model.fit(
                          X_train, y_train,
                          epochs=nb_epochs,
                          batch_size=nb_epochs,
                          validation_split=0.2,
                          shuffle=False,
                          verbose=verbose
                        )

  def predict(self,X):
    """
	#######################################################################################
	#
	#						Function for prediction
	# 
	#######################################################################################
	"""	
    self.predictions = self.model.predict(X)
    self.mae_loss = np.mean(np.abs(self.predictions - X), axis=1)
    

  def get_anomaly_prediction_df(self,X,set_type='train',THRESHOLD=.02,TIME_STEPS=30,is_spx=True):
    """
	#######################################################################################
	#
	#						Function for anomaly prediction
	# 
	#######################################################################################
	"""	
    # right threshold for the anomaly detection
    # pred with the train set

    if is_spx:
      col = 'close_spx'
    else : 
      col = 'close_vix'

    #col ='close'
    predictions = self.model.predict(X)
    mae_loss = np.mean(np.abs(predictions - X), axis=1)

    if set_type=='train':
      score_df = pd.DataFrame(index=self.train[TIME_STEPS:].index)
      print(score_df.columns)
      score_df['close'] = self.train[TIME_STEPS:][col]
      score_df['close_vix'] = self.train[TIME_STEPS:]['close_vix']
      score_df['close_spx'] = self.train[TIME_STEPS:]['close_spx']
      score_df['period'] = self.train[TIME_STEPS:]['period']

    else :
      score_df = pd.DataFrame(index=self.test[TIME_STEPS:].index)
      print(score_df.columns)
      score_df['close'] = self.test[TIME_STEPS:][col]
      score_df['close_vix'] = self.test[TIME_STEPS:]['close_vix']
      score_df['close_spx'] = self.test[TIME_STEPS:]['close_spx']
      score_df['period'] = self.test[TIME_STEPS:]['period']


    score_df['loss'] = mae_loss
    score_df['threshold'] = THRESHOLD

    score_df['anomaly'] = score_df.loss > THRESHOLD

    return score_df



  def plot_history(self):
    """
	#######################################################################################
	#
	#	Plot training and validation performance metrics history of the Model
	# 
	#######################################################################################
	"""	
    fig = go.Figure()

    fig.add_trace(go.Scatter(
                        y=self.history.history['loss'],
                        mode='lines',
                        name='train loss'))
    
    fig.add_trace(go.Scatter(
                            y=self.history.history['val_loss'],
                            mode='lines',
                            name='validation loss'))
    
    fig.show() 
"""
# //////////////// Helper function ////////////////
def best_param(model,train,test,
                   nb_hidden_units = 128,
                   nb_epochs=10,
                   TIME_STEPS = 2,
                   droupout_rate = 0.2,
                   batch_size=32,
                   col='returns_spx',
                   verbose = False):

				   
  X_train, X_test, y_train, y_test = cd.get_data_sequences(train,test,col= col,TIME_STEPS = 2)
  model = LSTM_autoencoder(train, test)
  model.build_model(X_train,nb_hidden_units = nb_hidden_units,droupout_rate = droupout_rate,verbose = verbose)
  model.train_it(X_train, y_train,nb_epochs=nb_epochs,batch_size=batch_size,verbose=verbose)
  model.predict(X_train)
  return model.history
"""


  
# //////////////// Helper function ////////////////
def get_best_param(model,train,test,
                    nb_hidden_units,
                    nb_epochs,
                    TIME_STEPS,
                    droupout_rate,
                    batch_size,
                    col,
                     verbose = False):
	"""
	#######################################################################################
	#
	#						Find the best parameters among those provided
	#
	#######################################################################################
	"""
	
	
	def best_param(model,train,test,
					   nb_hidden_units = 128,
					   nb_epochs=10,
					   TIME_STEPS = 2,
					   droupout_rate = 0.2,
					   batch_size=32,
					   col='returns_spx',
					   verbose = False):

	  """
	  #######################################################################################
	  #
	  #						Helper function
	  #
	  #######################################################################################
	  """					   
	  X_train, X_test, y_train, y_test = cd.get_data_sequences(train,test,col= col,TIME_STEPS = 2)
	  model = LSTM_autoencoder(train, test)
	  model.build_model(X_train,nb_hidden_units = nb_hidden_units,droupout_rate = droupout_rate,verbose = verbose)
	  model.train_it(X_train, y_train,nb_epochs=nb_epochs,batch_size=batch_size,verbose=verbose)
	  model.predict(X_train)
	  return model.history  
	  
	  
	  
	  
	param_DF = pd.DataFrame()
	n = len(nb_hidden_units)
	for i in range(n):
		sys.stdout.write('\r')
		sys.stdout.write("%d/%d parameter sets" % (i+1,n))
		sys.stdout.flush()
    
	H = best_param(model,train,test,
                      nb_hidden_units = nb_hidden_units[i],
                      nb_epochs=nb_epochs[i],
                      TIME_STEPS = TIME_STEPS[i],
                      droupout_rate = droupout_rate[i],
                      batch_size=batch_size[i],
                      col=col,
                      verbose = False)
    
	loss = H.history['val_loss'][-1]
	param_DF = param_DF.append(
                        {
                        'nb_hidden_units': nb_hidden_units[i],
                        'nb_epochs':nb_epochs[i],
                        'TIME_STEPS' : TIME_STEPS[i],
                        'droupout_rate' : droupout_rate[i],
                        'batch_size':batch_size[i],
                        'loss':loss
                        }, ignore_index=True
                    )
	param_DF= param_DF.sort_values(by=['loss'],ascending=True)
	print('\n\n/////////////////// Best Parameters ////////////////////')
	return param_DF.iloc[0]	