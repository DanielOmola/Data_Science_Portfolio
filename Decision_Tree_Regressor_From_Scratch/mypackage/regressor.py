# -*- coding: utf-8 -*-
"""
Created on Thu May  6 12:52:04 2021

@author: danie
"""

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt



def rmse(h, y):
  """
  # ###################################################
  #    Function for root mean square computation
  # ###################################################
  """
  return sqrt(mean_squared_error(h, y))

def performance(preds, y):
  """
  # ###################################################
  #    Function for rmse and R2 display
  # ###################################################
  """
  loss=rmse(preds, y)
  r2 = r2_score(y, preds)
  print("----------- Performance Metrics ------------\n- RMSE : %f\n- R2 :%f"%(loss,r2))
  
  

class MyDecisionTreeRegressor:
  """
  # ######################################################################################
  #
  #    Class for building Decision Tree Regressor
  #
  #    hyperparameters : 
  #             - min_leaf, int (number data point for a node to be considered as a leaf)
  # 
  # ######################################################################################
  """
  def fit(self, X, y, min_leaf = 5):
    """
    # ----------------------------------------------------------------------
    #			Function for building decision tree with recursive Node class
    # ----------------------------------------------------------------------
    """
    self.dtree = Node(X, y, np.array(np.arange(len(y))), min_leaf)
    return self

  def predict(self, X):
    """
    # ----------------------------------------------------------------------
    #			Function for predicting new entries
    # ----------------------------------------------------------------------
    """
    return self.dtree.predict(X.values)





class Node:
  """
  # ######################################################################################
  #
  #    Class for building Decision Tree Regressor
  #
  #    hyperparameters : 
  #             - min_leaf, int (number data point for a node to be considered as a leaf)
  # 
  # ######################################################################################
  """
  def __init__(self, x, y, idxs, min_leaf=5,type_node='Principal'):
      #print("\n////////////////////////// New Node \\\\\\\\\\\\\\\\\\\\\\")
      self.x = x
      self.y = y
      self.idxs = idxs
      self.min_leaf = min_leaf
      self.row_count = len(idxs)
      self.col_count = x.shape[1]
      self.mean_node_val = np.mean(y[idxs])
      self.score = float('inf')
      self.find_varsplit()

  @property
  def split_col(self): return self.x.values[self.idxs,self.var_idx]

  @property
  def is_leaf(self): return self.score == float('inf')
  
  
  def find_varsplit(self):
    """
    # ----------------------------------------------------------------------
    #			Function for predicting new entries
    # ----------------------------------------------------------------------
    """

    # ###### loop through each columns
    for c in range(self.col_count): 

      # ###### to find row of columns split
      self.find_better_split(c)

    # --------------------------------------------
    #   if the node is a leaf
    #   just return the node
    # --------------------------------------------
    if self.is_leaf: 
      return

    # --------------------------------------------
    #   if the node can be splited in new nodes
    #   get the split value
    # --------------------------------------------
    x = self.split_col

    # #### get the left node index
    lhs = np.nonzero(x <= self.split)[0]

    # #### get the right node index
    rhs = np.nonzero(x > self.split)[0]

    # #### build left Node recursively untill reaching a leaf
    self.lhs = Node(self.x, self.y, self.idxs[lhs], self.min_leaf)

    # #### build right Node recursively untill reaching a leaf
    self.rhs = Node(self.x, self.y, self.idxs[rhs], self.min_leaf)


  def find_better_split(self, var_idx):

      # ##### instanciate column values
      x = self.x.values[self.idxs, var_idx]

      # ##### loop through columns values 
      for r in range(self.row_count):

          # #### get values lower than split candidate value
          lhs = x <= x[r]

          # #### get values higher than split candidate value
          rhs = x > x[r]

          # --------------------------------------------
          # Check if lower or upper split violate the min_leaf
          # constrainte in case of violation go to the next candidate 
          # --------------------------------------------
          if rhs.sum() < self.min_leaf or lhs.sum() < self.min_leaf: continue



          # --------------------------------------------
          #   if the row value is a valid candidat
          #   check the score 
          # --------------------------------------------
          curr_score = self.find_score(lhs, rhs)


          # --------------------------------------------
          #   if the current score
          #   is better than the previous one
          #   the column and the row are the 
          #   better candidates for the split
          # --------------------------------------------
          if curr_score < self.score:
            
            # #### the column is the best column candidate 
            self.var_idx = var_idx

            # #### the value is the best value/row candidate 
            self.split = x[r]
            
            # #### the current score become the best score
            self.score = curr_score
  
  def find_score(self, lhs, rhs):
    """
    # ----------------------------------------------------------------------
    #			Function to compute the score of split
    # ----------------------------------------------------------------------
    """
    # #### get the y based on index
    y = self.y[self.idxs]
    # #### get the std of left values
    lhs_std = y[lhs].std()
    # #### get the std of right values
    rhs_std = y[rhs].std()
    # #### return the weighted standard deviation as score
    return lhs_std * lhs.sum() + rhs_std * rhs.sum()


  def predict(self, x):
    """
    # ----------------------------------------------------------------------
    #			Function for single prediction
    # ----------------------------------------------------------------------
    """
    return np.array([self.predict_row(xi) for xi in x])

  def predict_row(self, xi):
    """
    # ----------------------------------------------------------------------
    #			Function for multiple prediction
    # ----------------------------------------------------------------------
    """
    # --------------------------------------------
    #   if the node is a leaf
    #   return the node mean value
    # --------------------------------------------
    if self.is_leaf: 
      return self.mean_node_val

    # --------------------------------------------
    #   if the node is not a leaf
    #   pass the value to the next left Node if true,
    #   to the right next Node if false
    # --------------------------------------------
    node = self.lhs if xi[self.var_idx] <= self.split else self.rhs
    return node.predict_row(xi)