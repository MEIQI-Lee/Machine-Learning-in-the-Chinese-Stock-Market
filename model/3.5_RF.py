#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import math
import sklearn
import sklearn.preprocessing
import datetime
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error

import warnings
warnings.simplefilter(action='ignore', category=Warning)
import seaborn as sns
sns.set()
pd.options.mode.chained_assignment = None  # default='warn'

#Program that executes the rolling window train-val-test split 
get_ipython().run_line_magic('run', 'TimeBasedCV.ipynb')


# In[2]:


df = pd.read_csv(r'../Monthly_features/cleaned_merged_Monthly.csv')
#Convert to float 32 (format needed for the most ML models)
df[df.columns[2:]] = df[df.columns[2:]].astype('float32')
#Sort observations by date and stock id
df = df.sort_values(by = ['证券代码', '交易月份'], ascending = True)
df['交易月份'] = pd.to_datetime(df['交易月份'])


# In[3]:


large_stocks = pd.read_csv('large_stocks.csv')
small_stocks = pd.read_csv('small_stocks.csv')

df_large_stocks = df[df['证券代码'].isin(large_stocks['证券代码'])]
df_small_stocks = df[df['证券代码'].isin(small_stocks['证券代码'])]

large_shareholder = pd.read_csv('large_shareholder.csv')
small_shareholder = pd.read_csv('small_shareholder.csv')

df_large_shareholder = df[df['证券代码'].isin(large_shareholder['证券代码'])]
df_small_shareholder = df[df['证券代码'].isin(small_shareholder['证券代码'])]

soe_stocks = pd.read_csv('soe_stocks.csv')
non_soe_stocks = pd.read_csv('non_soe_stocks.csv')

df_soe_stocks = df[df['证券代码'].isin(soe_stocks['证券代码'])]
df_non_soe_stocks = df[df['证券代码'].isin(non_soe_stocks['证券代码'])]


# In[4]:


#所有股票的变量
features = df.columns[~df.columns.isin(["证券代码","交易月份","mom1m"])].tolist()
#Ranks in [0,1]interval
df[features]=df.groupby("交易月份")[features].rank(pct=True)
#Multiply by 2 and substract 1 to get ranks in interval [-1,1] 
df[features] = 2*df[features] - 1

#大股票
features = df_large_stocks.columns[~df_large_stocks.columns.isin(["证券代码","交易月份","mom1m"])].tolist()
df_large_stocks[features]=df_large_stocks.groupby("交易月份")[features].rank(pct=True)
df_large_stocks[features] = 2*df_large_stocks[features] - 1

#小股票
features = df_small_stocks.columns[~df_small_stocks.columns.isin(["证券代码","交易月份","mom1m"])].tolist()
df_small_stocks[features]=df_small_stocks.groupby("交易月份")[features].rank(pct=True)
df_small_stocks[features] = 2*df_small_stocks[features] - 1

#大股东
features = df_large_shareholder.columns[~df_large_shareholder.columns.isin(["证券代码","交易月份","mom1m"])].tolist()
df_large_shareholder[features]=df_large_shareholder.groupby("交易月份")[features].rank(pct=True)
df_large_shareholder[features] = 2*df_large_shareholder[features] - 1

#小股东
features = df_small_shareholder.columns[~df_small_shareholder.columns.isin(["证券代码","交易月份","mom1m"])].tolist()
df_small_shareholder[features]=df_small_shareholder.groupby("交易月份")[features].rank(pct=True)
df_small_shareholder[features] = 2*df_small_shareholder[features] - 1

#国企
features = df_soe_stocks.columns[~df_soe_stocks.columns.isin(["证券代码","交易月份","mom1m"])].tolist()
df_soe_stocks[features]=df_soe_stocks.groupby("交易月份")[features].rank(pct=True)
df_soe_stocks[features] = 2*df_soe_stocks[features] - 1

#非国企
features = df_non_soe_stocks.columns[~df_non_soe_stocks.columns.isin(["证券代码","交易月份","mom1m"])].tolist()
df_non_soe_stocks[features]=df_non_soe_stocks.groupby("交易月份")[features].rank(pct=True)
df_non_soe_stocks[features] = 2*df_non_soe_stocks[features] - 1


# # Random Forest（随机森林）
# 
# ### Tuning parameters:
# 
# 
# - the depth of trees **𝐿**: **max_depth**
# - the number of trees **𝐵** added to the ensemble prediction: **n_estimators** 
# - the number of predictors **𝑀** randomly considered as potential split variables: **max_features** 
# ## 所有股票

# In[5]:


tscv = TimeBasedCV(train_period=60,
                   val_period=24,
                   test_period=12,
                   freq='months')

features = df.columns[~df.columns.isin(['证券代码',"mom1m"])].tolist()
X = df[features]
y = df["mom1m"]

predictions = []
y_test_list =[]
dates = []
dic_r2_all = {}

# Model’s complexity: dictionary to save the depth of the trees over time
dic_max_depth_all = {}

# Grid of prespecified values for each hyperparameter  
param_grid = {'max_depth': [1,2,3,4,5,6], 'max_features': [3, 6, 12, 24, 46, 49]}
#convert grid to list containing all posible hyperparameter combinations to iterate over
grid = list(ParameterGrid(param_grid))
# Empty container to save the objective loss function (mean squared error) for each hyperparameter combination
mse = np.full((len(grid),1),np.nan, dtype = np.float32)


for train_index, val_index, test_index in tscv.split(X, first_split_date= datetime.date(2005,1,31), second_split_date= datetime.date(2013,1,31)):

    X_train   = X.loc[train_index].drop('交易月份', axis=1)
    y_train = y.loc[train_index]
    
    X_val   = X.loc[val_index].drop('交易月份', axis=1)
    y_val = y.loc[val_index]

    X_test    = X.loc[test_index].drop('交易月份', axis=1)
    y_test  = y.loc[test_index]
    
    #Loop over the list containing all pyperparameter combinations, fit on the training sample and use 
    #validation set to generate predictions:
    for i in range(len(grid)):
        #n_jobs: number of jobs to run in parallel (-1: use all processors)
        # random_state: to ensure that results remain stable
        RFR_val = RandomForestRegressor(bootstrap = True,random_state=42, n_jobs=-1,
                                        max_depth= grid[i]["max_depth"], 
                                        max_features=grid[i]["max_features"],
                                        n_estimators = 100)
        RFR_val.fit(X_train, y_train)
        Yval_predict=RFR_val.predict(X_val)
        #calculate mean squared error for each hyperparameter combination
        mse[i,0] = mean_squared_error(y_val,Yval_predict)


    #The optimal combination of hyperparameters is the one that causes the lowest loss 
    optim_param = grid[np.argmin(mse)]

    #Fit again using the train and validation set and the optimal value for each hyperparameter 
    RFR = RandomForestRegressor(bootstrap = True,random_state=42, n_jobs=-1,
                                max_depth=optim_param["max_depth"], 
                                max_features=optim_param["max_features"],
                                n_estimators = 100)
    RFR.fit(np.concatenate((X_train, X_val)), np.concatenate((y_train, y_val)))
    preds=RFR.predict(X_test)


    #Save predictions, dates and the true values of the dependent variable to list
    predictions.append(preds)
    dates.append(y_test.index)
    y_test_list.append(y_test)

    #Calculate OOS model performance the for current window
    r2 = 1-sum(pow(y_test-preds,2))/sum(pow(y_test,2))
    #Save OOS model performance and the respective month to dictionary
    dic_r2_all["r2." + str(y_test.index)] = r2
    # Save the depth of the trees to dictionary to explore how model complexity evolves over time
    dic_max_depth_all["depth." + str(y_test.index)]= optim_param["max_depth"]
   
    
    
#Concatenate to get results over the whole OOS test period (Jan 2010-Dec 2019)
predictions_all= np.concatenate(predictions, axis=0)
y_test_list_all= np.concatenate(y_test_list, axis=0) 
dates_all= np.concatenate(dates, axis=0)

R2OOS_RF = 1-sum(pow(y_test_list_all-predictions_all,2))/sum(pow(y_test_list_all,2))
print("R2OOS random forest: ", R2OOS_RF)


# # 大股票 

# In[8]:


tscv = TimeBasedCV(train_period=60,
                   val_period=24,
                   test_period=12,
                   freq='months')


features = df_large_stocks.columns[~df_large_stocks.columns.isin(['证券代码',"mom1m"])].tolist()
X = df_large_stocks[features]
y = df_large_stocks["mom1m"]

predictions_large_stocks = []
y_test_list_large_stocks =[]
dates_large_stocks = []
dic_r2_all_large_stocks = {}


param_grid = {'max_depth': [1,2,3,4,5,6], 'max_features': [3, 6, 12, 24, 46, 49]}
grid = list(ParameterGrid(param_grid))
mse = np.full((len(grid),1),np.nan, dtype = np.float32)


for train_index, val_index, test_index in tscv.split(X, first_split_date= datetime.date(2005,1,31), second_split_date= datetime.date(2013,1,31)):

    X_train   = X.loc[train_index].drop('交易月份', axis=1)
    y_train = y.loc[train_index]
    
    X_val   = X.loc[val_index].drop('交易月份', axis=1)
    y_val = y.loc[val_index]

    X_test    = X.loc[test_index].drop('交易月份', axis=1)
    y_test  = y.loc[test_index]
    
    for i in range(len(grid)):
        RFR_val = RandomForestRegressor(bootstrap = True,random_state=42, n_jobs=-1,
                                        max_depth= grid[i]["max_depth"], 
                                        max_features=grid[i]["max_features"],
                                        n_estimators = 100)
        RFR_val.fit(X_train, y_train)
        Yval_predict=RFR_val.predict(X_val)
        mse[i,0] = mean_squared_error(y_val,Yval_predict)

    optim_param = grid[np.argmin(mse)]

    RFR = RandomForestRegressor(bootstrap = True,random_state=42, n_jobs=-1,
                                max_depth=optim_param["max_depth"], 
                                max_features=optim_param["max_features"],
                                n_estimators = 100)
    RFR.fit(np.concatenate((X_train, X_val)), np.concatenate((y_train, y_val)))
    preds=RFR.predict(X_test)



    predictions_large_stocks.append(preds)
    dates_large_stocks.append(y_test.index)
    y_test_list_large_stocks.append(y_test)

    
    
predictions_all_large_stocks= np.concatenate(predictions_large_stocks, axis=0)
y_test_list_all_large_stocks= np.concatenate(y_test_list_large_stocks, axis=0) 
dates_all_large_stocks= np.concatenate(dates_large_stocks, axis=0)

R2OOS_RF_large_stocks = 1-sum(pow(y_test_list_all_large_stocks-predictions_all_large_stocks,2))/sum(pow(y_test_list_all_large_stocks,2))
print("R2OOS random forest Large Stocks: ", R2OOS_RF_large_stocks)


# # 小股票

# In[9]:


tscv = TimeBasedCV(train_period=60,
                   val_period=24,
                   test_period=12,
                   freq='months')


features = df_small_stocks.columns[~df_small_stocks.columns.isin(['证券代码',"mom1m"])].tolist()
X = df_small_stocks[features]
y = df_small_stocks["mom1m"]

predictions_small_stocks = []
y_test_list_small_stocks =[]
dates_small_stocks = []
dic_r2_all_small_stocks = {}


param_grid = {'max_depth': [1,2,3,4,5,6], 'max_features': [3, 6, 12, 24, 46, 49]}
grid = list(ParameterGrid(param_grid))
mse = np.full((len(grid),1),np.nan, dtype = np.float32)


for train_index, val_index, test_index in tscv.split(X, first_split_date= datetime.date(2005,1,31), second_split_date= datetime.date(2013,1,31)):

    X_train   = X.loc[train_index].drop('交易月份', axis=1)
    y_train = y.loc[train_index]
    
    X_val   = X.loc[val_index].drop('交易月份', axis=1)
    y_val = y.loc[val_index]

    X_test    = X.loc[test_index].drop('交易月份', axis=1)
    y_test  = y.loc[test_index]
    
    for i in range(len(grid)):
        RFR_val = RandomForestRegressor(bootstrap = True,random_state=42, n_jobs=-1,
                                        max_depth= grid[i]["max_depth"], 
                                        max_features=grid[i]["max_features"],
                                        n_estimators = 100)
        RFR_val.fit(X_train, y_train)
        Yval_predict=RFR_val.predict(X_val)
        mse[i,0] = mean_squared_error(y_val,Yval_predict)

    optim_param = grid[np.argmin(mse)]

    RFR = RandomForestRegressor(bootstrap = True,random_state=42, n_jobs=-1,
                                max_depth=optim_param["max_depth"], 
                                max_features=optim_param["max_features"],
                                n_estimators = 100)
    RFR.fit(np.concatenate((X_train, X_val)), np.concatenate((y_train, y_val)))
    preds=RFR.predict(X_test)



    predictions_small_stocks.append(preds)
    dates_small_stocks.append(y_test.index)
    y_test_list_small_stocks.append(y_test)

    
    
predictions_all_small_stocks= np.concatenate(predictions_small_stocks, axis=0)
y_test_list_all_small_stocks= np.concatenate(y_test_list_small_stocks, axis=0) 
dates_all_small_stocks= np.concatenate(dates_small_stocks, axis=0)

R2OOS_RF_small_stocks = 1-sum(pow(y_test_list_all_small_stocks-predictions_all_small_stocks,2))/sum(pow(y_test_list_all_small_stocks,2))
print("R2OOS random forest Small Stocks: ", R2OOS_RF_small_stocks)


# # 大股东

# In[10]:


tscv = TimeBasedCV(train_period=60,
                   val_period=24,
                   test_period=12,
                   freq='months')


features = df_large_shareholder.columns[~df_large_shareholder.columns.isin(['证券代码',"mom1m"])].tolist()
X = df_large_shareholder[features]
y = df_large_shareholder["mom1m"]

predictions_large_shareholder = []
y_test_list_large_shareholder =[]
dates_large_shareholder = []
dic_r2_all_large_shareholder = {}


param_grid = {'max_depth': [1,2,3,4,5,6], 'max_features': [3, 6, 12, 24, 46, 49]}
grid = list(ParameterGrid(param_grid))
mse = np.full((len(grid),1),np.nan, dtype = np.float32)


for train_index, val_index, test_index in tscv.split(X, first_split_date= datetime.date(2005,1,31), second_split_date= datetime.date(2013,1,31)):

    X_train   = X.loc[train_index].drop('交易月份', axis=1)
    y_train = y.loc[train_index]
    
    X_val   = X.loc[val_index].drop('交易月份', axis=1)
    y_val = y.loc[val_index]

    X_test    = X.loc[test_index].drop('交易月份', axis=1)
    y_test  = y.loc[test_index]
    
    for i in range(len(grid)):
        RFR_val = RandomForestRegressor(bootstrap = True,random_state=42, n_jobs=-1,
                                        max_depth= grid[i]["max_depth"], 
                                        max_features=grid[i]["max_features"],
                                        n_estimators = 100)
        RFR_val.fit(X_train, y_train)
        Yval_predict=RFR_val.predict(X_val)
        mse[i,0] = mean_squared_error(y_val,Yval_predict)

    optim_param = grid[np.argmin(mse)]

    RFR = RandomForestRegressor(bootstrap = True,random_state=42, n_jobs=-1,
                                max_depth=optim_param["max_depth"], 
                                max_features=optim_param["max_features"],
                                n_estimators = 100)
    RFR.fit(np.concatenate((X_train, X_val)), np.concatenate((y_train, y_val)))
    preds=RFR.predict(X_test)



    predictions_large_shareholder.append(preds)
    dates_large_shareholder.append(y_test.index)
    y_test_list_large_shareholder.append(y_test)

    
    
predictions_all_large_shareholder= np.concatenate(predictions_large_shareholder, axis=0)
y_test_list_all_large_shareholder= np.concatenate(y_test_list_large_shareholder, axis=0) 
dates_all_large_shareholder= np.concatenate(dates_large_shareholder, axis=0)

R2OOS_RF_large_shareholder = 1-sum(pow(y_test_list_all_large_shareholder-predictions_all_large_shareholder,2))/sum(pow(y_test_list_all_large_shareholder,2))
print("R2OOS random forest Large Shareholder: ", R2OOS_RF_large_shareholder)


# # 小股东

# In[11]:


tscv = TimeBasedCV(train_period=60,
                   val_period=24,
                   test_period=12,
                   freq='months')


features = df_small_shareholder.columns[~df_small_shareholder.columns.isin(['证券代码',"mom1m"])].tolist()
X = df_small_shareholder[features]
y = df_small_shareholder["mom1m"]

predictions_small_shareholder = []
y_test_list_small_shareholder =[]
dates_small_shareholder = []
dic_r2_all_small_shareholder = {}


param_grid = {'max_depth': [1,2,3,4,5,6], 'max_features': [3, 6, 12, 24, 46, 49]}
grid = list(ParameterGrid(param_grid))
mse = np.full((len(grid),1),np.nan, dtype = np.float32)


for train_index, val_index, test_index in tscv.split(X, first_split_date= datetime.date(2005,1,31), second_split_date= datetime.date(2013,1,31)):

    X_train   = X.loc[train_index].drop('交易月份', axis=1)
    y_train = y.loc[train_index]
    
    X_val   = X.loc[val_index].drop('交易月份', axis=1)
    y_val = y.loc[val_index]

    X_test    = X.loc[test_index].drop('交易月份', axis=1)
    y_test  = y.loc[test_index]
    
    for i in range(len(grid)):
        RFR_val = RandomForestRegressor(bootstrap = True,random_state=42, n_jobs=-1,
                                        max_depth= grid[i]["max_depth"], 
                                        max_features=grid[i]["max_features"],
                                        n_estimators = 100)
        RFR_val.fit(X_train, y_train)
        Yval_predict=RFR_val.predict(X_val)
        mse[i,0] = mean_squared_error(y_val,Yval_predict)

    optim_param = grid[np.argmin(mse)]

    RFR = RandomForestRegressor(bootstrap = True,random_state=42, n_jobs=-1,
                                max_depth=optim_param["max_depth"], 
                                max_features=optim_param["max_features"],
                                n_estimators = 100)
    RFR.fit(np.concatenate((X_train, X_val)), np.concatenate((y_train, y_val)))
    preds=RFR.predict(X_test)



    predictions_small_shareholder.append(preds)
    dates_small_shareholder.append(y_test.index)
    y_test_list_small_shareholder.append(y_test)

    
    
predictions_all_small_shareholder= np.concatenate(predictions_small_shareholder, axis=0)
y_test_list_all_small_shareholder= np.concatenate(y_test_list_small_shareholder, axis=0) 
dates_all_small_shareholder= np.concatenate(dates_small_shareholder, axis=0)

R2OOS_RF_small_shareholder = 1-sum(pow(y_test_list_all_small_shareholder-predictions_all_small_shareholder,2))/sum(pow(y_test_list_all_small_shareholder,2))
print("R2OOS random forest Small Shareholder: ", R2OOS_RF_small_shareholder)


# # 国企

# In[12]:


tscv = TimeBasedCV(train_period=60,
                   val_period=24,
                   test_period=12,
                   freq='months')


features = df_soe_stocks.columns[~df_soe_stocks.columns.isin(['证券代码',"mom1m"])].tolist()
X = df_soe_stocks[features]
y = df_soe_stocks["mom1m"]

predictions_soe_stocks = []
y_test_list_soe_stocks =[]
dates_soe_stocks = []
dic_r2_all_soe_stocks = {}


param_grid = {'max_depth': [1,2,3,4,5,6], 'max_features': [3, 6, 12, 24, 46, 49]}
grid = list(ParameterGrid(param_grid))
mse = np.full((len(grid),1),np.nan, dtype = np.float32)


for train_index, val_index, test_index in tscv.split(X, first_split_date= datetime.date(2005,1,31), second_split_date= datetime.date(2013,1,31)):

    X_train   = X.loc[train_index].drop('交易月份', axis=1)
    y_train = y.loc[train_index]
    
    X_val   = X.loc[val_index].drop('交易月份', axis=1)
    y_val = y.loc[val_index]

    X_test    = X.loc[test_index].drop('交易月份', axis=1)
    y_test  = y.loc[test_index]
    
    for i in range(len(grid)):
        RFR_val = RandomForestRegressor(bootstrap = True,random_state=42, n_jobs=-1,
                                        max_depth= grid[i]["max_depth"], 
                                        max_features=grid[i]["max_features"],
                                        n_estimators = 100)
        RFR_val.fit(X_train, y_train)
        Yval_predict=RFR_val.predict(X_val)
        mse[i,0] = mean_squared_error(y_val,Yval_predict)

    optim_param = grid[np.argmin(mse)]

    RFR = RandomForestRegressor(bootstrap = True,random_state=42, n_jobs=-1,
                                max_depth=optim_param["max_depth"], 
                                max_features=optim_param["max_features"],
                                n_estimators = 100)
    RFR.fit(np.concatenate((X_train, X_val)), np.concatenate((y_train, y_val)))
    preds=RFR.predict(X_test)



    predictions_soe_stocks.append(preds)
    dates_soe_stocks.append(y_test.index)
    y_test_list_soe_stocks.append(y_test)

    
    
predictions_all_soe_stocks= np.concatenate(predictions_soe_stocks, axis=0)
y_test_list_all_soe_stocks= np.concatenate(y_test_list_soe_stocks, axis=0) 
dates_all_soe_stocks= np.concatenate(dates_soe_stocks, axis=0)

R2OOS_RF_soe_stocks = 1-sum(pow(y_test_list_all_soe_stocks-predictions_all_soe_stocks,2))/sum(pow(y_test_list_all_soe_stocks,2))
print("R2OOS random forest Soe Stocks: ", R2OOS_RF_soe_stocks)


# # 非国企

# In[13]:


tscv = TimeBasedCV(train_period=60,
                   val_period=24,
                   test_period=12,
                   freq='months')


features = df_non_soe_stocks.columns[~df_non_soe_stocks.columns.isin(['证券代码',"mom1m"])].tolist()
X = df_non_soe_stocks[features]
y = df_non_soe_stocks["mom1m"]

predictions_non_soe_stocks = []
y_test_list_non_soe_stocks =[]
dates_non_soe_stocks = []
dic_r2_all_non_soe_stocks = {}


param_grid = {'max_depth': [1,2,3,4,5,6], 'max_features': [3, 6, 12, 24, 46, 49]}
grid = list(ParameterGrid(param_grid))
mse = np.full((len(grid),1),np.nan, dtype = np.float32)


for train_index, val_index, test_index in tscv.split(X, first_split_date= datetime.date(2005,1,31), second_split_date= datetime.date(2013,1,31)):

    X_train   = X.loc[train_index].drop('交易月份', axis=1)
    y_train = y.loc[train_index]
    
    X_val   = X.loc[val_index].drop('交易月份', axis=1)
    y_val = y.loc[val_index]

    X_test    = X.loc[test_index].drop('交易月份', axis=1)
    y_test  = y.loc[test_index]
    
    for i in range(len(grid)):
        RFR_val = RandomForestRegressor(bootstrap = True,random_state=42, n_jobs=-1,
                                        max_depth= grid[i]["max_depth"], 
                                        max_features=grid[i]["max_features"],
                                        n_estimators = 100)
        RFR_val.fit(X_train, y_train)
        Yval_predict=RFR_val.predict(X_val)
        mse[i,0] = mean_squared_error(y_val,Yval_predict)

    optim_param = grid[np.argmin(mse)]

    RFR = RandomForestRegressor(bootstrap = True,random_state=42, n_jobs=-1,
                                max_depth=optim_param["max_depth"], 
                                max_features=optim_param["max_features"],
                                n_estimators = 100)
    RFR.fit(np.concatenate((X_train, X_val)), np.concatenate((y_train, y_val)))
    preds=RFR.predict(X_test)



    predictions_non_soe_stocks.append(preds)
    dates_non_soe_stocks.append(y_test.index)
    y_test_list_non_soe_stocks.append(y_test)

    
    
predictions_all_non_soe_stocks= np.concatenate(predictions_non_soe_stocks, axis=0)
y_test_list_all_non_soe_stocks= np.concatenate(y_test_list_non_soe_stocks, axis=0) 
dates_all_non_soe_stocks= np.concatenate(dates_non_soe_stocks, axis=0)

R2OOS_RF_non_soe_stocks = 1-sum(pow(y_test_list_all_non_soe_stocks-predictions_all_non_soe_stocks,2))/sum(pow(y_test_list_all_non_soe_stocks,2))
print("R2OOS random forest non-soe Stocks: ", R2OOS_RF_non_soe_stocks)


# In[14]:


#不同数据集下的R2oos
chart = np.array([[R2OOS_RF],
                  [R2OOS_RF_large_stocks],
                  [R2OOS_RF_small_stocks],
                  [R2OOS_RF_large_shareholder],
                  [R2OOS_RF_small_shareholder],
                  [R2OOS_RF_soe_stocks],
                  [R2OOS_RF_non_soe_stocks]])
                     
r2oos_rf = pd.DataFrame(chart, columns=['RF'],
                              index=["All", "Top 70%", "Bottom 30%", 
                                     "A.M.C.P.S. Top 70%", "A.M.C.P.S. Bottom 30%", "SOE", "Non-SOE"])

r2oos_rf


# In[16]:


#Save the model's performance measures to compare with other models later.
r2oos_rf.to_csv(r'r2oos results/r2oos_rf.csv')

