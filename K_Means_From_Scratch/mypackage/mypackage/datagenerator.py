# -*- coding: utf-8 -*-
"""
Created on Thu May  6 12:52:04 2021

@author: danie
"""

import numpy as np
np.random.seed(0)

def GenerateData():
    x1=np.random.randn(50,2)
    x2x=np.random.randn(80,1)+12
    x2y=np.random.randn(80,1)
    x2=np.column_stack((x2x,x2y))
    x3=np.random.randn(100,2)+8
    x4=np.random.randn(120,2)+15
    data=np.concatenate((x1,x2,x3,x4))
    noise=np.random.randn(50,2)*4 +10
    noisy_Data=np.concatenate((data,noise))
    return noisy_Data