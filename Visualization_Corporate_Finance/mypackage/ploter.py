# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 10:45:01 2021

@author: daniel omola
"""

import pandas as pd
import numpy as np
import plotly.express as px

def pivot(df,values,index):
  """
  ###############################################################
  #
  #				Function to generate pivot table
  #					 based on value and index
  #  
  ###############################################################
  """
  ################## Helper Function ###################
  def sub_pivot(df,values,index):
    table = pd.pivot_table(
                          df,
                          values=values,
                          index=index,
                          columns=['Fiscal Year'],
                          aggfunc=np.sum
                        )

    table = table/1000000
    table=table.fillna(0)
    table = table.sort_values(by=[table.columns[-1]],ascending=False)
    return table
    #####################################
    
  return sub_pivot(df,values=values,index=[index])  


def show_macro_evolution(data,col = 'Total_Debt'):
  """
  #######################################################################
  #
  #		Function to generate the interactive evolution chart of a 
  #					specific metric for all Sectors
  #  
  #######################################################################
  """
  data['Fiscal Year']=data['Fiscal Year'].astype(int)
  sector_evolution = pivot(data[data['Fiscal Year'].astype(int)>2009],values=col,index='Sector')

  sector_evolution=(sector_evolution.pct_change(axis=1,fill_method ='ffill'))

  sector_evolution=sector_evolution.replace([np.inf, -np.inf], np.nan)
  sector_evolution=sector_evolution.fillna(0)

  sector_evolution=(1 + sector_evolution).cumprod(axis=1)*100

  sector_evolution.T
  sector_evolution=sector_evolution.reset_index()
  sector_evolution



  df_1 = pd.DataFrame()
  for c in sector_evolution.columns[1:]:
    df_2 = sector_evolution[[c]].reset_index()
    df_2['Fiscal Year'] = c
    df_2['Sector'] = sector_evolution.Sector
    df_2.drop(columns='index',inplace=True)
    df_2.columns= ['Value','Year','Sector']
    df_1=df_1.append(df_2)
  df_1

  fig = px.line(df_1, x="Year",y='Value',color='Sector',title= col+" - by Sector")
  fig.show()


def show_micro_evolution(df,company,value = 'Total_Debt',index_value = 'Company'):
  """
  #######################################################################
  #
  #		Function to generate the interactive evolution chart of a 
  #					specific metric for one company
  #  
  #######################################################################
  """
  df['Fiscal Year']=df['Fiscal Year'].astype('int')
  evolution = pivot(df[(df['Fiscal Year']>2009)&(df['Company Name']==company)],values=value,index=index_value)

  evolution=(evolution.pct_change(axis=1,fill_method ='ffill'))

  evolution=evolution.replace([np.inf, -np.inf], np.nan)
  evolution=evolution.fillna(0)

  evolution=evolution.reset_index()
  Ev = evolution.T.reset_index()
  Ev.columns=['Year','percent']

  fig = px.bar(Ev, x="Year",y='percent',title= company+" - "+value)
  fig.show()
    


def get_sunburst(view,path,values):
  """
  ####################################################################################
  #
  #		Function to generate a sunburst interactive chart based on a path and a value 
  #
  ####################################################################################
  """
  fig = px.sunburst(view, path=path, values=values)
  fig.show()
