# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 10:45:01 2021

@author: daniel omola
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler 
from mypackage import ploter as plt

def get_data(path='./data/',verbose=False):
    """
    #######################################################################################
    #
    #		Load Raw Data and run some preprocessing task
	#					Files :
    #						   - S&P500 Data : data_spx.csv
    #						   - VIX Data : VIX.csv
    #
    #######################################################################################
    """
	
	###################### Helper Function #################################
	
    to_float=(lambda x : float(x.replace(",",".")))
	
	###################### Get and preprocess SP500 Data ###################
    data_spx = pd.read_csv(path + 'data_spx.csv',sep =";")
    data_spx['date'] = data_spx['date'].astype('datetime64[ns]') 
    data_spx['year'] = data_spx['date'].dt.year
    data_spx['month'] = data_spx['date'].dt.month
    data_spx['period'] = data_spx[['year','month']].apply(lambda r : "%d_%d"%(r[0],r[1] ),axis=1)
    data_spx.set_index('date', inplace = True)
    data_spx.close = data_spx.close.apply(to_float) 
    data_spx['returns'] = data_spx['close'].pct_change()
    data_spx. dropna(inplace=True) 
	
	###################### Get and preprocess VIX Data ###################	
    data_vix = pd.read_csv(path + 'VIX.csv')
    new_name ={'Date':'date',
           'Close':'close_vix'}
    data_vix.rename(columns = new_name, inplace = True)
    data_vix['date'] = data_vix['date'].astype('datetime64[ns]') 
    data_vix.set_index('date', inplace = True)
    data_vix['returns_vix'] = data_vix['close_vix'].pct_change()
    data_vix.dropna(inplace=True)
    data_vix=data_vix[['close_vix','returns_vix']]
    data_spx=data_spx.merge(data_vix,how='left',on='date')
    data_spx. dropna(inplace=True) 

	###################### Change column name ###################
    new_name ={'returns':'returns_spx',
               'close':'close_spx'}
    data_spx.rename(columns = new_name, inplace = True)

	###################### Print  ###################	
    if verbose:
        print(f" //////////// Initial Data Shape  ////////////\n\n\t {data_spx.shape}")
        print(f"\n//////////// Describe ////////////\n\n {data_spx.describe()}")
        print(f"\n//////////// Data Types ////////////\n\n {data_spx.dtypes}")
    return data_spx


def split_data(data, pct_train_test = 0.95):
  """
  ####################################################################################
  #
  #				Split Data for training and testing
  #
  ####################################################################################
  """
  train_size = int(len(data) * pct_train_test)
  test_size = len(data) - train_size
  train, test = data.iloc[0:train_size], data.iloc[train_size:len(data)]

  return train, test
  


def scale_data(train, test, col = 'returns_spx', scaler = MinMaxScaler(),verbose = False):
  """
  ####################################################################################
  #
  #				Scale Data based on a specified scaler 
  #
  ####################################################################################
  """
  #scaler = MinMaxScaler()
  scaler = scaler.fit(train[[col]])
  train[col] = scaler.transform(train[[col]])
  test[col] = scaler.transform(test[[col]])
  if verbose :
    plt.hist(data = train, col = col)
  return train, test
