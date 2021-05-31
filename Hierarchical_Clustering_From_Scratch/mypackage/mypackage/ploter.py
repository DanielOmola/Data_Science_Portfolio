# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 10:45:01 2021

@author: daniel omola
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot(X,Y):
  text=[i for i in range(len(X))]
  fig1 = go.Scatter(
      x = X,
      y = Y,
      mode='markers+text',
      text=text,
    textposition="bottom center",
      name='y vs. X'
  )


  fig = make_subplots(specs=[[{"secondary_y": True}]])
  fig.add_trace(fig1)

  fig.update_layout(
      title = None,
      xaxis = dict(
          tickmode = 'auto',
          tick0 = 0.5,
          dtick = 0.75,
          showticklabels = True
      )
  )
  fig.update_yaxes(
    scaleanchor = "x",
    scaleratio = 1,
  )

  fig.show()