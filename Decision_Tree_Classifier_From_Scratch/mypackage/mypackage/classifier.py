# -*- coding: utf-8 -*-
"""
Created on Thu May  6 12:52:04 2021

@author: danie
"""

import numpy as np

from sklearn.metrics import log_loss, accuracy_score, recall_score
from sklearn.metrics import precision_score, f1_score, confusion_matrix
#from math import sqrt

def cross_entropy(h, y):
  """
  # ###################################################
  #    Function for logloss computation
  # ###################################################
  """
  return log_loss(y_pred=h, y_true=y)





def performance(preds, y):
  """
  # ###################################################
  #    Function for root mean square computation
  # ###################################################
  """
  print("----------- Performance Metrics ------------")
  print("- Log Loss:",cross_entropy(preds, y))
  print("- Accuracy:",accuracy_score(preds, y)*100)
  print("- Recall:",recall_score(preds, y, average='macro')*100)
  print("- Precision:",precision_score(preds, y, average='macro')*100)
  print("- F1 Score:",f1_score(preds, y, average='macro', zero_division=1)*100)
  print("\n----------- Confusion Matrix ------------")
  print(confusion_matrix( y,preds))

    


def vote(cls):
  counts = np.bincount(cls)
  return np.argmax(counts)




class MyDecisionTreeClassifier:
  """
  # ######################################################################################
  #
  #    Class for building Decision Tree Clssifier
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
  def __init__(self, x, y, idxs, min_leaf=1):
      #print("\n////////////////////////// New Node \\\\\\\\\\\\\\\\\\\\\\")
      self.x = x
      self.y = y
      self.idxs = idxs
      self.min_leaf = min_leaf
      self.row_count = len(idxs)
      self.col_count = x.shape[1]
      self.col = x.columns
      # #### Majority vote win
      self.val = vote(y[idxs])
      # #### Impurity GINI or entropy
      self.score = float(1)
      self.find_varsplit()

  @property
  def split_col(self): 
    #print(self.idxs,self.var_idx)
    return self.x.values[self.idxs,self.var_idx]

  @property
  def is_leaf(self): return self.score == float(1)

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

    try :
      self.val_type=str(self.x.iloc[:,self.var_idx].dtypes)
    except:
      self.val_type=str(self.x.iloc[:,0].dtypes)

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

    if self.val_type == 'category':
      # #### get the left node index
      lhs = np.nonzero(x == self.split)[0]
      # #### get the right node index
      rhs = np.nonzero(x != self.split)[0]  
  
    else:
      # #### get the left node index
      lhs = np.nonzero(x <= self.split)[0]
      # #### get the right node index
      rhs = np.nonzero(x > self.split)[0]

    # #### build left Node recursively untill reaching a leaf
    self.lhs = Node(self.x, self.y, self.idxs[lhs], self.min_leaf)
    
    # #### build right Node recursively untill reaching a leaf
    self.rhs = Node(self.x, self.y, self.idxs[rhs], self.min_leaf)


  def find_better_split(self, var_idx):
    """
    # ----------------------------------------------------------------------
    #			Function for best split candidates (columns and value)
    # ----------------------------------------------------------------------
    """
    # ##### instanciate column values
    x = self.x.values[self.idxs, var_idx]

    # ##### instanciate column data type
    self.val_type=str(self.x.iloc[:,var_idx].dtypes)

    # ##### loop through columns values 
    for r in range(self.row_count):
        #print(x[r])

        if self.val_type=='category':

          # #### get values equal 
          # #### to candidate value
          lhs = x == x[r]

          # #### get values different
          # #### from candidate value
          rhs = x != x[r]
        else:
          # #### get values lower 
          # #### than candidate value
          lhs = x <= x[r]
          # #### get values higher 
          # #### than candidate value
          rhs = x > x[r]
        
        # --------------------------------------------
        # Check if lower or upper split violate the min_leaf
        # constraint in case of violation go to the next candidate 
        # --------------------------------------------
        if rhs.sum() < self.min_leaf or lhs.sum() < self.min_leaf:
          continue


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

            # #### the value is the best value/row candidate
            self.score = curr_score

              


  def find_score(self, lhs, rhs):
    """
    # ----------------------------------------------------------------------
    #			Function to compute the score of split
    # ----------------------------------------------------------------------
    """

    # #### get the y based on index
    y = self.y[self.idxs]

    # #### get the 0/1 proba of left values
    lhs_count = len(y[lhs])
    proba_0 = y[lhs][y[lhs]==0.].sum()/lhs_count
    proba_1 = y[lhs][y[lhs]==1.].sum()/lhs_count
    # #### compute the left impurity   
    impurity_lhs = 1- proba_0**2 - proba_1**2

    # #### get the 0/1 proba of right values
    rhs_count = len(y[rhs])
    proba_0 =  y[rhs][y[rhs]==0.].sum()/rhs_count
    proba_1 = y[rhs][y[rhs]==1.].sum()/rhs_count
    # #### compute the right impurity
    impurity_rhs = 1- proba_0**2 - proba_1**2

    # #### compute the global impurity
    impurity = (impurity_lhs*lhs_count + impurity_rhs*rhs_count)/(lhs_count + rhs_count)
    return impurity


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
      return self.val


    # --------------------------------------------
    #   if the node is not a leaf
    #   pass the value to the next left Node if true,
    #   to the right next Node if false
    # --------------------------------------------
    node = self.lhs if xi[self.var_idx] <= self.split else self.rhs

    return node.predict_row(xi)