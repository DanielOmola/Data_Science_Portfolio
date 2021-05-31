# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 10:45:01 2021

@author: daniel omola
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def plot_regressors(X,y,theta,theta_ridge,theta_lasso,theta_sub_optimal= np.array([0.5, 2.5 ])):
    fig1 = go.Scatter(
    x = X.reshape(X.shape[0],),
    y = y.reshape(X.shape[0],),
    mode='markers',
    name='y vs. X')

    x = X.reshape(X.shape[0],1)
    intercept = np.ones((x.shape[0], 1))
    x = np.concatenate((intercept, x), axis=1)


    fig2 = go.Scatter(
        x = X.reshape(X.shape[0],),
        y = np.dot(x,theta),
        mode='lines',
        name='Regression'
    )


    theta_sub_optimal = theta_sub_optimal
    fig3 = go.Scatter(
        x = X.reshape(X.shape[0],),
        y = np.dot(x,theta_sub_optimal),
        mode='lines',
        name='Sub Optimale Line'
    )


    fig4 = go.Scatter(
        x = X.reshape(X.shape[0],),
        y = np.dot(x,theta_ridge),
        mode='lines',
        name='Ridge Regression'
    )

    fig5 = go.Scatter(
        x = X.reshape(X.shape[0],),
        y = np.dot(x,theta_lasso),
        mode='lines',
        name='Lasso Regression'
    )


    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(fig1)
    fig.add_trace(fig2,secondary_y=False)
    fig.add_trace(fig3,secondary_y=False)
    fig.add_trace(fig4,secondary_y=False)
    fig.add_trace(fig5,secondary_y=False)

    fig.update_layout(
        title = 'Linear Regression',
        xaxis = dict(
            tickmode = 'auto',
            tick0 = 0.5,
            dtick = 0.75,
            showticklabels = True
        )
    )

    fig.show()