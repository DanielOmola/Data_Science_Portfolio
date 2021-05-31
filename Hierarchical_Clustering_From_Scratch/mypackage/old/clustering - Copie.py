# -*- coding: utf-8 -*-
"""
Created on Thu May  6 12:52:04 2021

@author: danie
"""

import numpy as np
#np.random.seed(0)
import itertools
import re

def dist(X,x,y,linkage_method='single'):
    D = []
    if type(x)==int:
        x = [x]
    if type(y)==int:
        y = [y]
    for i in itertools.product(x,y):
        D.append(sum((X[i[0]]-X[i[1]])**2)**.5)
    if linkage_method=='single':
        return(min(D))
    if linkage_method=='complete':
        #print(x,y,D,max(D))
        #print(D,max(D))
        return(max(D))
    if linkage_method=='average':
        #print(x,y,D,np.mean(D))
        #print(D,np.mean(D))
        return(np.mean(D))
        #m = lambda x1,x2 :.5*np.array(x1)+ .5*np.array(x2)

def flatten(TheList):
    a = str(TheList)
    b,crap = re.subn(r'[\[,\]]', ' ', a)
    c = b.split()
    d = [int(x) for x in c]
    return list(d)

class HClustering:
    """
    # ######################################################################################
    #
    #    Class for building Hierarchical Clustering Model
    #
    #    hyperparameters : 
    #             - linkage_method, string (number of cluster ) ,
    #             - ...,
    #             - ...
    # 
    # ######################################################################################
    """    
    def __init__(self):
        self.unused = []
        self.next_process = []
        self.clusters = [[],[]]

    def fit(self,X,linkage_method='single'):
        """
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        #			Function for training : finds the optimal centroids
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        """
        self.unused = [i for i in range(X.shape[0])]
        self.next_process = [i for i in range(X.shape[0])]

        while len(self.unused)!=0: 
            count = 0
            dist_list =[]
            
            # ###### ------------- Distance Matrix calculation -----------------
            while count != (len(self.next_process)*(len(self.next_process)-1)/2):
                for c in self.next_process[count+1:]:
                    dist_list.append((dist(X,self.next_process[count],c,linkage_method=linkage_method),[self.next_process[count],c]))
                count+=1
            dist_list.sort(reverse=False)

            # ######------------ Minimum Distance pair (can be 2 clusters) ------------------
            cl = dist_list[0][1]

            # --------- Loop trough the elements of the minimum distance pair ------------
            for i in list(set(flatten(cl))): 
                # ###### try to remove element of min distance pair from the unused points list
                try:
                    self.unused.remove(i)
                    # ###### if the element is a previous cluster, the element
                    # ###### is not in the unused list, so nothing happen
                except :
                    pass

            try:
                # ###### try to remove the elements of min distance pair from the next_process points list...
                for i in cl:
                    try:
                        self.next_process.remove(i) 
                    except :
                        pass
            except :
                # ###### ...in case of failure, that mean that the element is a cluster
                # ###### that need to be flatened before processed
                for i in list(set(flatten(cl))):
                    try:
                        self.next_process.remove(i) 
                    except :
                        pass

            # ###### add minimum distance pair to the next process list
            self.next_process.append(flatten(cl))
            #print("-------- Next Process --------\n\t"+str(next_process))

            # ###### add minimum distance pair to the cluster list
            self.clusters.append(flatten(cl))
            print('cl : %s'%str(cl))

            
    def fit2(self,X,linkage_method='single'):
            """
            # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
            #			Function for training : finds the optimal centroids
            # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
            """
            self.unused = [i for i in range(X.shape[0])]
            self.next_process = [i for i in range(X.shape[0])]
    
            #while len(self.unused)!=0: 
            while len(set(flatten(self.clusters[-2:])))!=X.shape[0]:
                #print(' - unused',self.unused)
                count = 0
                dist_list =[]
                
                # ###### ------------- Distance Matrix calculation -----------------
                while count != (len(self.next_process)*(len(self.next_process)-1)/2):
                    for c in self.next_process[count+1:]:
                        dist_list.append((dist(X,self.next_process[count],c,linkage_method=linkage_method),[self.next_process[count],c]))
                    count+=1
                dist_list.sort(reverse=False)
    
                # ######------------ Minimum Distance pair (can be 2 clusters) ------------------
                cl = dist_list[0][1]
    
                # --------- Loop trough the elements of the minimum distance pair ------------
                for i in list(set(flatten(cl))): 
                    # ###### try to remove element of min distance pair from the unused points list
                    try:
                        self.unused.remove(i)
                        # ###### if the element is a previous cluster, the element
                        # ###### is not in the unused list, so nothing happen
                    except :
                        pass
    
                try:
                    # ###### try to remove the elements of min distance pair from the next_process points list...
                    for i in cl:
                        try:
                            self.next_process.remove(i) 
                        except :
                            pass
                except :
                    # ###### ...in cas of failure, that mean that the element is a cluster
                    # ###### that need to be flatened before processed
                    for i in list(set(flatten(cl))):
                        try:
                            self.next_process.remove(i) 
                        except :
                            pass
    
                # ###### add minimum distance pair to the next process list
                self.next_process.append(flatten(cl))
                #print("-------- Next Process --------\n\t"+str(next_process))
    
                # ###### add minimum distance pair to the cluster list
                self.clusters.append(cl)
                #print(self.clusters)
                #self.unused.extend(flatten(cl))
                #self.next_process.extend(flatten(cl))
            self.clusters=self.clusters[2:]