# -*- coding: utf-8 -*-
"""
Created on Thu May  6 12:52:04 2021

@author: danie
"""

from mypackage import ploter as plt
import numpy as np
np.random.seed(0)

class KMeans:
    """
    # ######################################################################################
    #
    #    Class for building Kmeans Model
    #
    #    hyperparameters : 
    #             - k, int (number of cluster ) ,
    #             - tol, float (if the centroid is not moving more than the tolerance value),
    #             - max_iter, int (max number of iteration we're willing to run)
    # 
    # ######################################################################################
    """
    def __init__(self, k=2, tol=0.001, max_iter=300,verbose = False):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
    def fit(self,data):
        """
    	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    	#			Function for training : finds the optimal centroids
    	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    	"""
        self.n = data.shape[0]
        # ########### start an empty dictionary for centroids storing
        self.centroids = {}
        
        # ########### define our starting centroids as...
        # ###########  ... the first two data samples in our data
        
        #for i in range(self.k):
        for i,j in zip(range(self.k),np.random.randint(0,self.n,self.k)):
            self.centroids[i] = data[j]
            
        if self.verbose:    
            print("---- Initial Centroids ---- \n\t%s"%str(self.centroids))
        
        # ########### iterate through our max_iter value
        for i in range(self.max_iter):
            
            # ###### define a empty dictionnary
            # ###### used for storing classifications
            self.classifications = {} 
            
            # ###### create dictionnary keys
            # ###### (by iterating through range of self.k)
            # ###### with values as empty list that will
            # ###### data point according to its belonging class
            for j in range(self.k): 
                self.classifications[j] = []
                
            if self.verbose:     
                print("\n- Classifications - %d/%d : \n\t%s"%(i+1, self.max_iter,str(self.classifications)))
            
            # ###### iterate through data points
            for featureset in data: 

              # ###### compute distance between each data points
              # ###### and each current centroids
              distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
            
              # ###### get the index of closer centroid
              # ######  for each data points  
              classification = distances.index(min(distances))
                
              # ###### add data points to the closer centroid  
              self.classifications[classification].append(featureset) #and classify them as such
                
            if self.verbose: 
                print("\n- Classifications - %d/%d : \n\t%s"%(i+1, self.max_iter,str(self.classifications)))
            
            # ###### store the previous centroids
            # ###### including their data points
            prev_centroids = dict(self.centroids)

            # ###### compute the average of each centroids
            # ###### that will be retained as the new centroids
            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification],axis=0)
            
            if self.verbose:     
                print("\n---- New Centroids ---- \n\t%s"%str(self.centroids))   

            # ###### check if the algorithm is optimized
            # ###### start by considering that it is optimized
            optimized = True
            
            
            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                
                # ###### if the percentage change between current and previous centroids is...
                # ###### ... greater than tol, the algorithm is not considered optimized 
                #print(np.sum((current_centroid-original_centroid)/original_centroid))
                if abs(np.sum((current_centroid-original_centroid)/original_centroid*100.0)) > self.tol:
                    #print(np.sum((current_centroid-original_centroid)/original_centroid*100.0))
                    optimized = False
                    
            # ###### break the loop when the algorithm is optimized         
            if optimized:
                print("\n##   Optimization terminated after %d iterrations.  ##"%i)
                break

    def predict(self,data):
        """
    	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    	#						Function for prediction
    	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    	"""
        distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification
    
def train_and_plot(clf,data,title = ''):
    # ##################################
    #
    #  train the model/cluster the data
    #   and plot the result 
    #
    # ################################## 
    clf.fit(data)

    colors = 10*['blueviolet', 'lightblue', 'lightgreen', 'purple','turquoise','green']
    colors_data = []
    i = 0
    X_ = np.zeros(shape=(data.shape[0],data.shape[1]))
    for classification in clf.classifications:
        color = colors[classification]
        for v in clf.classifications[classification]:
            X_[i]=v
            colors_data.append(color)
            i+=1
    centroids_merge = [list(v) for v in clf.centroids.values()]
    centroids_merge=np.array(centroids_merge)
    plt.plot_scatter(data=X_,centroids=centroids_merge, color = colors_data,title=title)