# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 10:45:01 2021

@author: daniel omola
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_scatter(data, color = None,title=''):

  if color :
    fig1 = go.Scatter(
        x = data[:,0],
        y = data[:,1],
        mode='markers',
        name='',
                marker=dict(
              color=color ))
  else:
    fig1 = go.Scatter(
    x = data[:,0],
    y = data[:,1],
    mode='markers',
    name='')

  fig = make_subplots(specs=[[{"secondary_y": True}]])
  fig.add_trace(fig1)


  fig.update_layout(
  title=title,
      xaxis = dict(
          tickmode = 'auto',
          tick0 = 0.5,
          dtick = 0.75,
          showticklabels = True
      )
  )

  fig.show()