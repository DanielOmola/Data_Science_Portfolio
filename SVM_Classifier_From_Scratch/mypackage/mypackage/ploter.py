# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 10:45:01 2021

@author: daniel omola
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np



def plot_svm(X,y,W,b,title_plt):
    """
	####################################################################################
	#
	#		Plot : data points, hyperplane and boundaries(1 and -1 lines) 
	#
	####################################################################################
	"""
    
    # ################# Helper Function ###################
    def f(x0, W, b, c=0):
        return (-(W[0] * x0 + c) + b ) / W[1]

    def f_plus(x0, W, b, c=-1):
        return (-(W[0] * x0 + c) + b ) / W[1]

    def f_minus(x0, W, b, c=+1):
        return (-(W[0] * x0 + c) + b ) / W[1]
    
    # ############ Plot the data points ##################
    x1,x2 = np.split(X, 2, axis=1)
    color= ['red' if x ==-1 else 'green' for x in y]
    width=0.5

    fig1 = go.Scatter(
      x = x1.reshape(x1.shape[0],),
      y = x2.reshape(x2.shape[0],),
      marker=dict(color=color),
      mode='markers',
      name='y vs. X'
  )

    # ######## data preparation for line plot #######
    #c = 0.
    x = np.linspace(0,10,11)

    y= f(x,W, b, c=0)
    y_plus= f_plus(x,W, b, c=-1)
    y_minus= f_minus(x,W, b, c=1)

    # ######## Plot the hyper plan line  #######  
    fig2 = go.Scatter(
      x = x.reshape(x.shape[0],),
      y = y.reshape(x.shape[0],),
      mode='lines',
      name='0',
      line=dict(color="black",width=width,dash='dot')
  )

    # ######## Plot the boundary line for positive class (+1) #######
    fig3 = go.Scatter(
      x = x.reshape(x.shape[0],),
      y = y_plus.reshape(y_plus.shape[0],),
      mode='lines',
      name ='+1',
      line=dict(color="green",width=width)
  )


    # ######## Plot the boundary line for negative class (-1) #######
    fig4 = go.Scatter(
      x = x.reshape(x.shape[0],),
      y = y_minus.reshape(x.shape[0],),
      mode='lines',
      name='-1',
      line=dict(color="red",width=width)
  )


    # ######## Set plot final parameters #######
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(fig1)

    fig.add_trace(fig2,secondary_y=False)
    fig.add_trace(fig3,secondary_y=False)
    fig.add_trace(fig4,secondary_y=False)

    fig.update_layout(
      title = title_plt,
      xaxis = dict(
          tickmode = 'auto',
          tick0 = 0.5,
          dtick = 0.75,
          showticklabels = True
      )
  )

    fig.show()  
    
def plot_svm_2(X,y,W,b,title_plt):
    """
	####################################################################################
	#
	#		Plot : points identified as anomaly and close price of S&P500 and VIX
	#
	####################################################################################
	"""
    
    # ################# Helper Function ###################

    def f(x0, W, b, c=0):
        return (-W[0] * x0 - b ) / W[1]
    
    def f_plus(x0, W, b, c=-1):
        return (1-W[0] * x0 - b ) / W[1]
    
    def f_minus(x0, W, b, c=+1):
        return (-1-W[0] * x0 - b ) / W[1]
    
    # ############ Plot the data points ##################
    x1,x2 = np.split(X, 2, axis=1)
    color= ['red' if x ==-1 else 'green' for x in y]
    width=0.5

    fig1 = go.Scatter(
      x = x1.reshape(x1.shape[0],),
      y = x2.reshape(x2.shape[0],),
      marker=dict(color=color),
      mode='markers',
      name='y vs. X'
  )

    # ######## data preparation for line plot #######
    #c = 0.
    x = np.linspace(0,10,11)

    y= f(x,W, b, c=0)
    y_plus= f_plus(x,W, b, c=0)
    y_minus= f_minus(x,W, b, c=0)

    # ######## Plot the hyper plan line  #######  
    fig2 = go.Scatter(
      x = x.reshape(x.shape[0],),
      y = y.reshape(x.shape[0],),
      mode='lines',
      name='0',
      line=dict(color="black",width=width,dash='dot')
  )

    # ######## Plot the bundary line for positive class (+1) #######
    fig3 = go.Scatter(
      x = x.reshape(x.shape[0],),
      y = y_plus.reshape(y_plus.shape[0],),
      mode='lines',
      name ='+1',
      line=dict(color="green",width=width)
  )


    # ######## Plot the bundary line for negative class (-1) #######
    fig4 = go.Scatter(
      x = x.reshape(x.shape[0],),
      y = y_minus.reshape(x.shape[0],),
      mode='lines',
      name='-1',
      line=dict(color="red",width=width)
  )


    # ######## Set plot final parameters #######
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(fig1)

    fig.add_trace(fig2,secondary_y=False)
    fig.add_trace(fig3,secondary_y=False)
    fig.add_trace(fig4,secondary_y=False)

    fig.update_layout(
      title = title_plt,
      xaxis = dict(
          tickmode = 'auto',
          tick0 = 0.5,
          dtick = 0.75,
          showticklabels = True
      )
  )

    fig.show()  