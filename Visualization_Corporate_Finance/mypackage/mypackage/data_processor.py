# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 10:45:01 2021

@author: daniel omola
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
#from sklearn.preprocessing import StandardScaler, MinMaxScaler 
#from mypackage import ploter as plt
import plotly.express as px

def load_raw_data():
  """
  #######################################################################################
  #
  #						Load Raw Data from zip files
  # 
  #				us-companies.zip, industries.zip, markets.zip,
  #				us-balance-annual.zip, us-income-annual.zip
  #
  #######################################################################################
  """
  companies = pd.read_csv('./data/us-companies.zip',compression='zip', header= 0, sep=';')
  industries = pd.read_csv('./data/industries.zip',compression='zip', header= 0, sep=';')
  markets = pd.read_csv('./data/markets.zip',compression='zip', header= 0, sep=';')
  balance_sheet = pd.read_csv('./data/us-balance-annual.zip',compression='zip', header= 0, sep=';')
  income = pd.read_csv('./data/us-income-annual.zip',compression='zip', header= 0, sep=';')

  return companies,industries,markets,balance_sheet,income



def manage_column_type(data):
  """
  ####################################################################################
  #
  #				Change some columns data type to category
  #		'Company Name','Currency','Fiscal Year','Industry','IndustryId'...
  #
  ####################################################################################
  """
  categorical = ['Company Name','Currency','Fiscal Year','Industry','IndustryId',
                'Market Name','MarketId','Fiscal Period','Publish Date','Report Date','Restated Date',
                'Sector','SimFinId','Ticker','key']
				
  numerical = pd.Series(data.columns)     
  numerical=numerical[~numerical.isin(categorical)] 

  for c in categorical:
    data[c]=data[c].astype('category') 

  return data  

  
###################### Helper Function : Join 2 DF #################################
  
def jointure(df_G,df_D,left_on,right_on,how='left',verbose=False):

  cols_to_use = df_D.columns.difference(df_G.columns).tolist()
  cols_to_use.append(right_on)
	  
  data = df_G.merge(df_D[cols_to_use],
						how=how,
						left_on=left_on,
						right_on=right_on)
  if verbose:
    print("DF Gauche : %s | DF Droite : %s"%(str(df_G.shape),str(df_D.shape)))
    print(" => DF Final : %s "%str(data.shape))

  return data
  

def join_raw_data(industries, markets, balance_sheet, income, companies,verbose=False):
  """
  ####################################################################################
  #
  #				Join sequentially all available DF
  #
  ####################################################################################
  """


  # ################################## First Join ####################################  
  
  data = jointure(df_G = income,
                  df_D = companies,
                  left_on = 'SimFinId',
                  right_on = 'SimFinId',
                  how='left',verbose=verbose)


  # ############# Join previous join result with industries DF #######################
  
  data = jointure(df_G = data,
                  df_D = industries,
                  left_on = 'IndustryId',
                  right_on = 'IndustryId',
                  how='left',verbose=verbose)

  # ############# Join previous join result with balance_sheet DF #######################
  
  balance_sheet['key']=balance_sheet[['SimFinId','Fiscal Year']].apply(lambda x : '%s_%s'%(str(x[0]),str(x[1])),axis=1)
  data['key']=data[['SimFinId','Fiscal Year']].apply(lambda x : '%s_%s'%(str(x[0]),str(x[1])),axis=1)

  data = jointure(df_G = data,
                  df_D = balance_sheet,
                  left_on = 'key',
                  right_on = 'key',
                  how='left',verbose=verbose)

  # ############# Join previous join result with markets DF #############################
  
  data = jointure(df_G = data,
                  df_D = markets,
                  left_on = 'Currency',
                  right_on = 'Currency',
                  how='left',verbose=verbose)
				  
  # ############# Replace NaN by 'other' for Sector and Industry columns ################
  
  data['Industry']=data['Industry'].astype(str)
  data['Industry']=data['Industry'].apply(lambda v : 'other' if v=='nan' else v )
  data['Sector']=data['Sector'].astype(str)
  data['Sector']=data['Sector'].apply(lambda v : 'other' if v=='nan' else v )
  data = manage_column_type(data)

  return data

  
def check_unbalanced(data):
  """
  ####################################################################################
  #
  #				Get and Plot the unballanced Balance Cheets Data
  #
  ####################################################################################
  """
  
  # ############# Get the unballanced Balance Cheets Data ############################
  
  unbalanced_BS = data[(data['Total Assets']!=data['Total Liabilities & Equity'])]
  unbalanced_BS['gape'] = unbalanced_BS['Total Assets'] - unbalanced_BS['Total Liabilities & Equity']

  
  # ############# Plot the unballence Balance Cheets Data ############################  
  
  print('=> Number of unconsistent Balance Sheet : %d\n'%len(unbalanced_BS.gape))
  
  fig = px.box(unbalanced_BS,
				y='gape')
  fig.show()
  
  
  fig = px.box(unbalanced_BS, x="Fiscal Year",
				y='gape',color='Fiscal Year')
  fig.show()
  
  
  fig = px.violin(unbalanced_BS, x="Fiscal Year",
					y='gape',color='Fiscal Year')
  fig.show()
  
  
  fig = px.scatter(unbalanced_BS, x="Fiscal Year",
					y='gape',color='Fiscal Year')
  fig.show()
  
  
  stats_unbalanced = unbalanced_BS.groupby(['Fiscal Year']).agg({'gape':['mean','count','std']}).reset_index()
  stats_unbalanced.columns = ['Fiscal Year','gape_mean','gape_count','gape_std']
  stats_unbalanced=stats_unbalanced[stats_unbalanced.gape_count!=0]
  
  
  fig = px.bar(stats_unbalanced, x="Fiscal Year",
				y='gape_mean',color='Fiscal Year')
  fig.show()
  
  
  fig = px.bar(stats_unbalanced, x="Fiscal Year",
				y='gape_count',color='Fiscal Year')
  fig.show()
  
 
def check_revenue(data):
  """
  ####################################################################################
  #
  #				Get and Plot the inconsistent Revenue Data
  #
  ####################################################################################
  """
  
  # ################# Get the inconsistent Revenue Data ##############################
  
  unconsistant_R = data[data.Revenue<=0]
  unconsistant_R=unconsistant_R[['Fiscal Year','Revenue']]
  
  
  # ################# Plot the unballence Balance Cheets Data ########################
  
  print('Number of unconsistent Income Statment : %d\n'%len(unconsistant_R))
  unconsistant_R_grp = unconsistant_R.groupby(['Fiscal Year']).agg({'Revenue':['count']}).reset_index()
  unconsistant_R_grp.columns = ['Fiscal Year','Revenue']
  
  fig = px.bar(unconsistant_R_grp, x="Fiscal Year",
				y='Revenue',color='Fiscal Year')
  fig.show()
  
def plot_NA(df):
  """
  ####################################################################################
  #
  #				Check and plot NaN on selected column
  #
  ####################################################################################
  """
  ################# Helper Functions ####################
  def check_na_col(df):
	  cl = df.columns[df.isna().any()].tolist()
	  return cl

  def check_number_na(df,na_col):
	  r = df[na_col].isna().sum()
	  return r
		
  def check_percent_na(df,na_col):
	  return (abs(df[na_col].count()-df[na_col].count().max()))/df[na_col].shape[0]*100
	#######################################################
	
  na_col = check_na_col(df)
	
  NA=check_number_na(df,na_col)
	
  fig = px.bar(NA)
  fig.show()


def check_availability_year(data,values='Short Term Debt',col='Company Name'):
  """
  ####################################################################################
  #
  #				Check and plot data availability by Year
  #
  ####################################################################################
  """
  short_debt = pivot(data,values=values,index=col)
  short_debt = short_debt[short_debt!=0]
  fig = px.bar(short_debt.count(axis=0),title=values)
  fig.show()

  
def check_availability_company(data,values='Short Term Debt',col='Company Name'):
  """
  ####################################################################################
  #
  #				Check and plot data availability by Company
  #
  ####################################################################################
  """
  company_data_availability = pivot(data,values=values,index='Company Name')
  company_data_availability = company_data_availability[company_data_availability!=0]
  company_data_availability.count(axis=1)
  fig = px.bar(company_data_availability.count(axis=1).sort_values(ascending=False))
  fig.show()
  
  

def get_consistent_data(data,initial_year = 2015,values='Short Term Debt'):
  """
  ####################################################################################
  #
  #				Generate consistent data set
  #			same data availability in each year and for each company
  #
  ####################################################################################
  """
  short_debt = pivot(data,values=values,index='Company Name')
  short_debt = short_debt[short_debt!=0]
  companies_to_keep = short_debt.index[(short_debt.loc[:,initial_year:].isna().sum(axis=1)==0)]
  data = data[data['Company Name'].isin(companies_to_keep)]
  data = data[data['Fiscal Year'].astype(int)>=initial_year]
  print('%d companies kept.'%len(companies_to_keep))
  return data    

 
def pivot(df,values,index):
  """
  ####################################################################################
  #
  #				Get Pivot Table for a selected columns
  #			with Year as columns
  #
  ####################################################################################
  """
  ################# Helper Function ################
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
  
   
def features_engineering(data):
  """
  ####################################################################################
  #
  #				Create new columns containing Profitability, Leverage ratios
  #
  ####################################################################################
  """
  categorical = ['Company Name','Currency','Fiscal Year','Industry','IndustryId',
                'Market Name','MarketId','Fiscal Period','Publish Date','Report Date','Restated Date',
                'Sector','SimFinId','Ticker','key']
  numerical = pd.Series(data.columns)     
  numerical=numerical[~numerical.isin(categorical)]
  
  # ################# Prepare average_total_equity ##############################
  
  total_equity = pivot(data,values='Total Equity',index='Company Name')
  average_total_equity=total_equity.rolling(window=2,axis=1).mean()
  average_total_equity=average_total_equity.sort_index(ascending=True)
  
  
  # ################# Prepare average_total_assets ##############################
  
  total_asset = pivot(data,values='Total Assets',index='Company Name')
  average_total_assets=total_asset.rolling(window=2,axis=1).mean()
  average_total_assets=average_total_assets.sort_index(ascending=True)
  
  
  # ################# Get operating_income ##############################
  
  operating_income = pivot(data,values='Operating Income (Loss)',index='Company Name')
  operating_income = operating_income.sort_index(ascending=True)

  operating_income = pivot(data,values='Operating Income (Loss)',index='Company Name')
  operating_income = operating_income.sort_index(ascending=True)

  
  # ################# Get net_income ##############################
  
  net_income = pivot(data,values='Net Income',index='Company Name')
  net_income = net_income.sort_index(ascending=True)

  
  # ################# Compute operating_ROA ##############################
  
  operating_ROA = operating_income.div(average_total_assets,axis=1)
  operating_ROA.head(1)
  operating_ROA=operating_ROA.stack().reset_index(name='op_roa').rename(columns={'Company Name':'company','Fiscal Year':'year'})
  operating_ROA['key']=operating_ROA[['company','year']].apply(lambda r : '%s%s'%(r[0],r[1]),axis=1)

  
  # ################# Compute ROA ##############################
  
  ROA = net_income.div(average_total_assets,axis=1)
  ROA.head(1)
  ROA=ROA.stack().reset_index(name='roa').rename(columns={'Company Name':'company','Fiscal Year':'year'})
  ROA['key']=ROA[['company','year']].apply(lambda r : '%s%s'%(r[0],r[1]),axis=1)

  
  # ################# Compute ROE ##############################
  
  ROE = net_income.div(average_total_equity,axis=1)
  ROE=ROE.stack().reset_index(name='roe').rename(columns={'Company Name':'company','Fiscal Year':'year'})
  ROE['key']=ROE[['company','year']].apply(lambda r : '%s%s'%(r[0],r[1]),axis=1)

  data['key']=data[['Company Name','Fiscal Year']].apply(lambda r : '%s%s'%(r[0],r[1]),axis=1)
  
  
  # ################# Join operating_ROA, ROA and ROE with initial Data ######
  
  data = jointure(df_G = data,
                  df_D = operating_ROA.iloc[:,2:],
                  left_on = 'key',
                  right_on = 'key',
                  how='left')
  
  data = jointure(df_G = data,
                  df_D = ROE.iloc[:,2:],
                  left_on = 'key',
                  right_on = 'key',
                  how='left')

  data = jointure(df_G = data,
                  df_D = ROA.iloc[:,2:],
                  left_on = 'key',
                  right_on = 'key',
                  how='left')
 
  # ################# Compute Profitability ratios ##############################

  data['Total_Debt']=data['Long Term Debt']+data['Short Term Debt']
  data['Gross_Profit_Margin']=data['Gross Profit']/data['Revenue']
  data['Operating_Profit_Margin']=data['Operating Income (Loss)']/data['Revenue']
  data['Pretax_Income']=data['Pretax Income (Loss)']/data['Revenue']
  data['Net_Profit_Margin']=data['Net Income']/data['Revenue']

  # ################# Compute Leverage ratios ###################################

  data['Debt_to_Assets']=data['Total_Debt']/data['Total Assets']
  data['Debt_to_Capital']=data['Total_Debt']/(data['Total_Debt']+data['Total Equity'])

  data['ROC']=data['Operating Income (Loss)']/(data['Total_Debt']+data['Total Equity'])

  data=data.replace([np.inf, -np.inf], np.nan)
  data.loc[:,numerical]=data.loc[:,numerical].fillna(0)
  numerical = pd.Series(data.columns)     
  numerical=numerical[~numerical.isin(categorical)]

  return data

def filter_data(data,ratios):
    """
    ####################################################################################
    #
    #				Get a Data Set containing only the specified ratios
    #
    ####################################################################################
    """
    credit_data=data[ratios]
    credit_data = credit_data[credit_data!=0]                          
    credit_data = data[ratios].dropna()
    credit_data = data[ratios].dropna()
     
    new_col = {'Fiscal Year':'Year',
               'Company Name':'Company',
               }
    credit_data=credit_data.rename(columns=new_col)
    return credit_data