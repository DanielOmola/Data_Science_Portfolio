# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 10:45:01 2021

@author: daniel omola
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def hist(data,col="close_spx"):
    """
	####################################################################################
	#
	#				Plot histogram on specified column
	#
	####################################################################################
	"""
    fig = px.histogram(data, x=col)
    fig.update_layout(
        autosize=False,
        width=600,
        height=400,)
    fig.show()
  

def plot_loss_tresshold(df):
    """
	####################################################################################
	#
	#				Plot time sery of prediction error and a line as a tresshold 
	#
	####################################################################################
	"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df.loss,
        mode='lines',
        name='error'))
  
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df.threshold,
        mode='lines',
        name='threshold'))
    fig.show()
  

def plot_anomalies(df,is_spx=True, secondary_y = 'close_vix' ):
    """
	####################################################################################
	#
	#		Plot : points identified as anomaly and close price of S&P500 and VIX
	#
	####################################################################################
	"""
    col = 'close'
    anomalies = df.copy()
    anomalies.close = anomalies[[col,'anomaly']].apply(lambda r : r[0] if r[1]==True else 0,axis=1)
    anomalies=anomalies[anomalies.anomaly==True]

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])


    fig.add_trace(go.Scatter(
                      x=df.index,
                      y=df[col],
                      mode='lines',
                      name='close'))



    fig.add_trace(go.Scatter(
                          x=anomalies.index,
                          y=anomalies[col],
                          mode='markers',
                          name='anomaly'))
    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df[secondary_y],
                        mode='lines',
                          marker_color='lightgrey',
                        name=secondary_y),
                secondary_y=True,)
  
    # Set x-axis title
    fig.update_xaxes(title_text="xaxis title")

  # Set y-axes titles
    if secondary_y == 'close_vix':
        fig.update_yaxes(title_text="<b>SP500</b> close", secondary_y=False)
        fig.update_yaxes(title_text="<b>VIX</b> close", secondary_y=True)
    else:
        fig.update_yaxes(title_text="<b>VIX</b> close", secondary_y=False)
        fig.update_yaxes(title_text="<b>SP500</b> close", secondary_y=True)
    fig.show()