# -*- coding: utf-8 -*-
"""
Created on Thu May  6 12:52:04 2021

@author: danie
"""

import numpy as np
import scipy as scipy
np.random.seed(0)

class DBSCAN:
    
    def __init__(self,eps = 1.5,Min_neighbor = 30 ):
        self.eps = eps
        self.Min_neighbor = Min_neighbor
        self.Cluster = []
        self.Clusters_List = []
        self.Cores = []
        self.Outliers = []
        
    def fit(self,data):
        self.m,self.n = data.shape
        self.DistanceMatrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(data, 'euclidean'))
        #Cluster = []
        #Clusters_List = []
        #Cores = []
        #Outliers = []
        self.visited=np.zeros(self.m,'int')

        # #### loop trough all data points
        for i in range(len(data)):

          # ### if the data point has not been visited   
          if self.visited[i]==0:

            # ### mark it as visited 
            self.visited[i]=1

            # ### find its neighbors
            Neighbors = np.where(self.DistanceMatrix[i]<=self.eps)
            N = Neighbors[0].tolist()
            nb_Neighbors = len(N)

            # ###### if the number of neighbors if < minimum required...
            # ###### ...the data point is considered as an outlier and... 
            # ###### ...added to the outliers list 
            if nb_Neighbors < self.Min_neighbor:
              self.Outliers.append((i,N))

            # ###### if the number of neighbors is >= minimum required...
            # ###### ...the data point is considered as a Core and can be ... 
            # ###### ... added to the core list and its neighbors list...
            # ###### ... can be expand through the ExpandClsuter function
            # ###### ... and then added to the Clusters_List 
            else :    
              cor = (i,N)
              self.Cores.append(cor)
              N = self.ExpandClsuter(N,self.Min_neighbor,self.eps)
              self.Clusters_List.append(N)

        self.cluster_labels = np.zeros(self.m,'int')
        for i, L in enumerate(self.Clusters_List):
          #print(i+1,L)
          for j in L:
            self.cluster_labels[j]= i+1

                
        return self.Clusters_List,self.cluster_labels

        # ########## Function for expanding the cluster
    def ExpandClsuter(self,N,Min_neighbor,esp):

        # ###### initialize neighbors list 
        Neighbors=[]

        # ###### initialize unvisited data points list
        unvisited_N = N[:]

        # ###### loop untill unvisited data points list is empty
        while unvisited_N:

            # ###### get the last data point of unvisited list 
            i=unvisited_N.pop()

            # ###### if the data point has not been visited...
            if self.visited[i]==0:
			
                # ### ... mark the data point as visited 
                self.visited[i]=1

                # ###### get the neighbor of the unvisited data point base on esp  
                Neighbors=np.where(self.DistanceMatrix[i]<self.eps)[0].tolist()

                # ###### if the number of neighbors if > minimum required...
                # ###### ...the data point is considered as a Core and can be ... 
                # ###### ... extend with the neighbors list       
                if len(Neighbors)>=self.Min_neighbor:
				
                    # ###### merge previous neighbors list with new neighbors
                    N.extend(Neighbors)
					
                    # ###### remove redundancy 
                    N = list(set(N))

                    # #### loop trough new neighbors...
                    # #### ...to find unvisited data points...
                    # #### ... to be append to unvisited list
                    for j in Neighbors:
                        if self.visited[j]==0:
                            unvisited_N.append(j)
                            unvisited_N = list(set(unvisited_N))
        return N