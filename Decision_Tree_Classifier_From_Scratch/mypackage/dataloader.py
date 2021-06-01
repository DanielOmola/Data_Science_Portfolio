# -*- coding: utf-8 -*-
"""
Created on Sun May 16 11:22:34 2021

@author: danie
"""
import zipfile
#import numpy as np
import pandas as pd

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

def encode(df,col):
    for cl in col:
        df[cl] = le.fit_transform(df[cl])
        #print(df[cl])
        #df[cl] = df[cl].astype('category')
    return(df)


def WomanOrChild(r):
    if (r[0]=='female') or (r[1]<18) :
        return 1
    return 0

def get_data(zip_folder = 'Titanic_Data.zip',split_rate = .7,encoded=False):
    
    # #### Loop through zip folder to read csv files
    with zipfile.ZipFile(zip_folder) as z:

        with z.open("train.csv") as f:
            train = pd.read_csv(f, header=0, delimiter=",")
            #print('------- Train ------- \n\tShape :%s\n\tHead \n%s'%(train.shape,train.head()))
    
        with z.open("test.csv") as f:
            test = pd.read_csv(f, header=0, delimiter=",")
            #print('------- Test ------- \n\tShape :%s\n\tHead \n%s'%(test.shape,test.head()))
        
    # #### drop NaN rows   
    train = train.dropna(subset=['Embarked','Age'])
    test = test.dropna(subset=['Embarked','Age'])
    
    # #### define features columns
    features = ['Pclass','Sex', 'Age', 'SibSp',
           'Parch', 'Fare',  'Embarked']
    
    # #### define categorical columns
    categorical = ['Pclass','Sex','Parch', 'Embarked']
    
    # #### reset index on X_train and y_train
    X_train = train[features].reset_index(drop=True)
    y_train = train['Survived'].reset_index(drop=True)
    
    X_train['WomanChild']=X_train[['Sex','Age']].apply(lambda r : WomanOrChild(r),axis=1)
    
    
    # #### split between train set 
    # #### and validation set
    split_rate = split_rate
    split_index = int(split_rate * len(X_train))
    
    X_validation=X_train.iloc[split_index:]
    y_validation=y_train.iloc[split_index:]
    
    X_train=X_train.iloc[:split_index]
    y_train=y_train.iloc[:split_index]
    
    # #### set categorical columns
    for cl in features:
      if cl in  categorical :
        X_train.loc[:,cl] =  X_train.loc[:,cl].astype('category')
        X_validation.loc[:,cl] =  X_validation.loc[:,cl].astype('category') 
        
    # #### encode categorical columns if specified
    if encoded:
        X_train = encode(X_train,categorical)
        X_validation = encode(X_validation,categorical)

    return X_train, X_validation,y_train,y_validation