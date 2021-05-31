# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 10:45:01 2021

@author: daniel omola
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_scatter(data,centroids=None,color=None,title=''):
  #print(centroids[:,0])
  #print(color)
  if color  :
    fig1 = go.Scatter(
        x = data[:,0],
        y = data[:,1],
        mode='markers',
        name='data',
                marker=dict(color=color ))
    
    fig2 = go.Scatter(
    x = centroids[:,0],
    y = centroids[:,1],
    mode='markers',
    name='centroids',
            marker=dict(
                    size=10,
          color='red' ))
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(fig1)
    fig.add_trace(fig2,secondary_y=False)
    
    
  else:
    fig1 = go.Scatter(
    x = data[:,0],
    y = data[:,1],
    mode='markers',
    name='data')

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(fig1)


  fig.update_layout(
      title = title,
      xaxis = dict(
          tickmode = 'auto',
          tick0 = 0.5,
          dtick = 0.75,
          showticklabels = True
      )
  )

  fig.show()
  
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