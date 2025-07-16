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
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression

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
df.head()


# In[3]:


df['交易月份'] = pd.to_datetime(df['交易月份'])


# In[4]:


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


# ### 连续的股票特征进行横截面排序，并将它们映射到[−1,1]区间
# 
# 股票存量特征逐月横截面排序，并将等级映射到 [-1,1] 区间，从而将特征转化为均匀分布并增加对异常值的不敏感性

# In[5]:


#所有股票的变量
features = df.columns[~df.columns.isin(["证券代码","交易月份","mom1m"])].tolist()
#Ranks in [0,1]interval
df[features]=df.groupby("交易月份")[features].rank(pct=True)
#Multiply by 2 and substract 1 to get ranks in interval [-1,1] 
df[features] = 2*df[features] - 1


# In[6]:


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


# # Partial Least Squares（偏最小二乘法）
# 
# ## 所有股票

# In[7]:


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


numpc_time = {}


numpc = np.arange(1, 18, 1).tolist()

mse = np.full((len(numpc),1),np.nan, dtype = np.float32)


for train_index, val_index, test_index in tscv.split(X, first_split_date= datetime.date(2005,1,31), second_split_date= datetime.date(2013,1,31)):

    X_train   = X.loc[train_index].drop('交易月份', axis=1)
    y_train = y.loc[train_index]
    
    X_val   = X.loc[val_index].drop('交易月份', axis=1)
    y_val = y.loc[val_index]

    X_test    = X.loc[test_index].drop('交易月份', axis=1)
    y_test  = y.loc[test_index]
    
    
    #Loop over the list containing potential number of components, fit on the training sample and use 
    #validation set to generate predictions
    for i in range(len(numpc)):
        pls_val = PLSRegression(n_components = numpc[i], scale = False)
        pls_val.fit(X_train, y_train)
        Yval_predict=pls_val.predict(X_val)
        #predictions are transformed into 1D array 
        Yval_predict = Yval_predict.ravel()
        #calculate mean squared error for each potential value of the numpc hyperparameter
        mse[i,0] = np.sqrt(mean_squared_error(y_val, Yval_predict))
       
       
      
    #The optimal value of the numpc hyperparameter is the value that causes the lowest loss
    optim_numpc = numpc[np.argmin(mse)]
    
    #Fit again using the train and validation set and the optimal numpc parameter
    pls = PLSRegression(n_components=optim_numpc, scale = False)
    pls.fit(np.concatenate((X_train, X_val)), np.concatenate((y_train, y_val)))
    #Use test set to generate final predictions
    preds = pls.predict(X_test)
    #predictions are transformed into 1D array 
    preds = preds.ravel()

    #Save predictions, dates and the true values of the dependent variable to list  
    predictions.append(preds)
    dates.append(y_test.index)
    y_test_list.append(y_test)
    
    #Calculate OOS model performance the for current window
    r2 = 1-sum(pow(y_test-preds,2))/sum(pow(y_test,2))
    #Save OOS model performance and the respective month to dictionary
    dic_r2_all["r2." + str(y_test.index)] = r2
    # Save the number of components to inspect  model's complexity over time 
    numpc_time["numpc." + str(y_test.index)] = optim_numpc

   
        
#Concatenate to get results over the whole OOS test period (Jan 2010-Dec 2019)
predictions_all= np.concatenate(predictions, axis=0)
y_test_list_all= np.concatenate(y_test_list, axis=0) 
dates_all= np.concatenate(dates, axis=0)

#Calculate OOS model performance over the entire test period in line with Gu et al (2020)
R2OOS_PLS = 1-sum(pow(y_test_list_all-predictions_all,2))/sum(pow(y_test_list_all,2))
print("R2OOS partial least squares: ", R2OOS_PLS)


# # 大股票

# In[8]:


from sklearn.metrics import r2_score

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

numpc =np.arange(1, 18, 1).tolist()
mse = np.full((len(numpc),1),np.nan, dtype = np.float32)


for train_index, val_index, test_index in tscv.split(X, first_split_date= datetime.date(2005,1,31), second_split_date= datetime.date(2013,1,31)):
    X_train   = X.loc[train_index].drop('交易月份', axis=1)
    y_train = y.loc[train_index]
    
    X_val   = X.loc[val_index].drop('交易月份', axis=1)
    y_val = y.loc[val_index]

    X_test    = X.loc[test_index].drop('交易月份', axis=1)
    y_test  = y.loc[test_index]
    
    for i in range(len(numpc)):
        pls_val = PLSRegression(n_components = numpc[i], scale = False)
        pls_val.fit(X_train, y_train)
        Yval_predict=pls_val.predict(X_val)
        Yval_predict = Yval_predict.ravel()
        mse[i,0] = np.sqrt(mean_squared_error(y_val, Yval_predict))

    optim_numpc = numpc[np.argmin(mse)]
    
    pls = PLSRegression(n_components=optim_numpc, scale = False)
    pls.fit(np.concatenate((X_train, X_val)), np.concatenate((y_train, y_val)))
    preds = pls.predict(X_test)
    preds = preds.ravel()

    predictions_large_stocks.append(preds)
    dates_large_stocks.append(y_test.index)
    y_test_list_large_stocks.append(y_test)
    
  
    r2_large_stocks = r2_score(y_test, preds)
    dic_r2_all_large_stocks["r2." + str(y_test.index)] = r2


predictions_all_large_stocks= np.concatenate(predictions_large_stocks, axis=0)
y_test_list_all_large_stocks= np.concatenate(y_test_list_large_stocks, axis=0) 
dates_all_large_stocks= np.concatenate(dates_large_stocks, axis=0)

R2OOS_PLS_large_stocks = r2_score(y_test_list_all_large_stocks, predictions_all_large_stocks)
print("R2OOS partial least squares Large Stocks: ", R2OOS_PLS_large_stocks)


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

numpc =np.arange(1, 18, 1).tolist()
mse = np.full((len(numpc),1),np.nan, dtype = np.float32)


for train_index, val_index, test_index in tscv.split(X, first_split_date= datetime.date(2005,1,31), second_split_date= datetime.date(2013,1,31)):
    X_train   = X.loc[train_index].drop('交易月份', axis=1)
    y_train = y.loc[train_index]
    
    X_val   = X.loc[val_index].drop('交易月份', axis=1)
    y_val = y.loc[val_index]

    X_test    = X.loc[test_index].drop('交易月份', axis=1)
    y_test  = y.loc[test_index]
    
    for i in range(len(numpc)):
        pls_val = PLSRegression(n_components = numpc[i], scale = False)
        pls_val.fit(X_train, y_train)
        Yval_predict=pls_val.predict(X_val)
        Yval_predict = Yval_predict.ravel()
        mse[i,0] = np.sqrt(mean_squared_error(y_val, Yval_predict))

    optim_numpc = numpc[np.argmin(mse)]
    
    pls = PLSRegression(n_components=optim_numpc, scale = False)
    pls.fit(np.concatenate((X_train, X_val)), np.concatenate((y_train, y_val)))
    preds = pls.predict(X_test)
    preds = preds.ravel()

    predictions_small_stocks.append(preds)
    dates_small_stocks.append(y_test.index)
    y_test_list_small_stocks.append(y_test)
    
  
    r2_small_stocks = r2_score(y_test, preds)
    dic_r2_all_small_stocks["r2." + str(y_test.index)] = r2


predictions_all_small_stocks= np.concatenate(predictions_small_stocks, axis=0)
y_test_list_all_small_stocks= np.concatenate(y_test_list_small_stocks, axis=0) 
dates_all_small_stocks= np.concatenate(dates_small_stocks, axis=0)

R2OOS_PLS_small_stocks = r2_score(y_test_list_all_small_stocks, predictions_all_small_stocks)
print("R2OOS partial least squares Small Stocks: ", R2OOS_PLS_small_stocks)


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

numpc =np.arange(1, 18, 1).tolist()
mse = np.full((len(numpc),1),np.nan, dtype = np.float32)


for train_index, val_index, test_index in tscv.split(X, first_split_date= datetime.date(2005,1,31), second_split_date= datetime.date(2013,1,31)):
    X_train   = X.loc[train_index].drop('交易月份', axis=1)
    y_train = y.loc[train_index]
    
    X_val   = X.loc[val_index].drop('交易月份', axis=1)
    y_val = y.loc[val_index]

    X_test    = X.loc[test_index].drop('交易月份', axis=1)
    y_test  = y.loc[test_index]
    
    for i in range(len(numpc)):
        pls_val = PLSRegression(n_components = numpc[i], scale = False)
        pls_val.fit(X_train, y_train)
        Yval_predict=pls_val.predict(X_val)
        Yval_predict = Yval_predict.ravel()
        mse[i,0] = np.sqrt(mean_squared_error(y_val, Yval_predict))

    optim_numpc = numpc[np.argmin(mse)]
    
    pls = PLSRegression(n_components=optim_numpc, scale = False)
    pls.fit(np.concatenate((X_train, X_val)), np.concatenate((y_train, y_val)))
    preds = pls.predict(X_test)
    preds = preds.ravel()

    predictions_large_shareholder.append(preds)
    dates_large_shareholder.append(y_test.index)
    y_test_list_large_shareholder.append(y_test)
    
  
    r2_large_shareholder = r2_score(y_test, preds)
    dic_r2_all_large_shareholder["r2." + str(y_test.index)] = r2


predictions_all_large_shareholder= np.concatenate(predictions_large_shareholder, axis=0)
y_test_list_all_large_shareholder= np.concatenate(y_test_list_large_shareholder, axis=0) 
dates_all_large_shareholder= np.concatenate(dates_large_shareholder, axis=0)

R2OOS_PLS_large_shareholder = r2_score(y_test_list_all_large_shareholder, predictions_all_large_shareholder)
print("R2OOS partial least squares Large Shareholder: ", R2OOS_PLS_large_shareholder)


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

numpc =np.arange(1, 18, 1).tolist()
mse = np.full((len(numpc),1),np.nan, dtype = np.float32)


for train_index, val_index, test_index in tscv.split(X, first_split_date= datetime.date(2005,1,31), second_split_date= datetime.date(2013,1,31)):
    X_train   = X.loc[train_index].drop('交易月份', axis=1)
    y_train = y.loc[train_index]
    
    X_val   = X.loc[val_index].drop('交易月份', axis=1)
    y_val = y.loc[val_index]

    X_test    = X.loc[test_index].drop('交易月份', axis=1)
    y_test  = y.loc[test_index]
    
    for i in range(len(numpc)):
        pls_val = PLSRegression(n_components = numpc[i], scale = False)
        pls_val.fit(X_train, y_train)
        Yval_predict=pls_val.predict(X_val)
        Yval_predict = Yval_predict.ravel()
        mse[i,0] = np.sqrt(mean_squared_error(y_val, Yval_predict))

    optim_numpc = numpc[np.argmin(mse)]
    
    pls = PLSRegression(n_components=optim_numpc, scale = False)
    pls.fit(np.concatenate((X_train, X_val)), np.concatenate((y_train, y_val)))
    preds = pls.predict(X_test)
    preds = preds.ravel()

    predictions_small_shareholder.append(preds)
    dates_small_shareholder.append(y_test.index)
    y_test_list_small_shareholder.append(y_test)
    
  
    r2_small_shareholder = r2_score(y_test, preds)
    dic_r2_all_small_shareholder["r2." + str(y_test.index)] = r2


predictions_all_small_shareholder= np.concatenate(predictions_small_shareholder, axis=0)
y_test_list_all_small_shareholder= np.concatenate(y_test_list_small_shareholder, axis=0) 
dates_all_small_shareholder= np.concatenate(dates_small_shareholder, axis=0)

R2OOS_PLS_small_shareholder = r2_score(y_test_list_all_small_shareholder, predictions_all_small_shareholder)
print("R2OOS partial least squares Small Shareholder: ", R2OOS_PLS_small_shareholder)


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

numpc =np.arange(1, 18, 1).tolist()
mse = np.full((len(numpc),1),np.nan, dtype = np.float32)


for train_index, val_index, test_index in tscv.split(X, first_split_date= datetime.date(2005,1,31), second_split_date= datetime.date(2013,1,31)):
    X_train   = X.loc[train_index].drop('交易月份', axis=1)
    y_train = y.loc[train_index]
    
    X_val   = X.loc[val_index].drop('交易月份', axis=1)
    y_val = y.loc[val_index]

    X_test    = X.loc[test_index].drop('交易月份', axis=1)
    y_test  = y.loc[test_index]
    
    for i in range(len(numpc)):
        pls_val = PLSRegression(n_components = numpc[i], scale = False)
        pls_val.fit(X_train, y_train)
        Yval_predict=pls_val.predict(X_val)
        Yval_predict = Yval_predict.ravel()
        mse[i,0] = np.sqrt(mean_squared_error(y_val, Yval_predict))

    optim_numpc = numpc[np.argmin(mse)]
    
    pls = PLSRegression(n_components=optim_numpc, scale = False)
    pls.fit(np.concatenate((X_train, X_val)), np.concatenate((y_train, y_val)))
    preds = pls.predict(X_test)
    preds = preds.ravel()

    predictions_soe_stocks.append(preds)
    dates_soe_stocks.append(y_test.index)
    y_test_list_soe_stocks.append(y_test)
    
  
    r2_soe_stocks = r2_score(y_test, preds)
    dic_r2_all_soe_stocks["r2." + str(y_test.index)] = r2


predictions_all_soe_stocks= np.concatenate(predictions_soe_stocks, axis=0)
y_test_list_all_soe_stocks= np.concatenate(y_test_list_soe_stocks, axis=0) 
dates_all_soe_stocks= np.concatenate(dates_soe_stocks, axis=0)

R2OOS_PLS_soe_stocks = r2_score(y_test_list_all_soe_stocks, predictions_all_soe_stocks)
print("R2OOS partial least squares Soe Stocks: ", R2OOS_PLS_soe_stocks)


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

numpc =np.arange(1, 18, 1).tolist()
mse = np.full((len(numpc),1),np.nan, dtype = np.float32)


for train_index, val_index, test_index in tscv.split(X, first_split_date= datetime.date(2005,1,31), second_split_date= datetime.date(2013,1,31)):
    X_train   = X.loc[train_index].drop('交易月份', axis=1)
    y_train = y.loc[train_index]
    
    X_val   = X.loc[val_index].drop('交易月份', axis=1)
    y_val = y.loc[val_index]

    X_test    = X.loc[test_index].drop('交易月份', axis=1)
    y_test  = y.loc[test_index]
    
    for i in range(len(numpc)):
        pls_val = PLSRegression(n_components = numpc[i], scale = False)
        pls_val.fit(X_train, y_train)
        Yval_predict=pls_val.predict(X_val)
        Yval_predict = Yval_predict.ravel()
        mse[i,0] = np.sqrt(mean_squared_error(y_val, Yval_predict))

    optim_numpc = numpc[np.argmin(mse)]
    
    pls = PLSRegression(n_components=optim_numpc, scale = False)
    pls.fit(np.concatenate((X_train, X_val)), np.concatenate((y_train, y_val)))
    preds = pls.predict(X_test)
    preds = preds.ravel()

    predictions_non_soe_stocks.append(preds)
    dates_non_soe_stocks.append(y_test.index)
    y_test_list_non_soe_stocks.append(y_test)
    
  
    r2_non_soe_stocks = r2_score(y_test, preds)
    dic_r2_all_non_soe_stocks["r2." + str(y_test.index)] = r2


predictions_all_non_soe_stocks= np.concatenate(predictions_non_soe_stocks, axis=0)
y_test_list_all_non_soe_stocks= np.concatenate(y_test_list_non_soe_stocks, axis=0) 
dates_all_non_soe_stocks= np.concatenate(dates_non_soe_stocks, axis=0)

R2OOS_PLS_non_soe_stocks = r2_score(y_test_list_all_non_soe_stocks, predictions_all_non_soe_stocks)
print("R2OOS partial least squares non-Soe Stocks: ", R2OOS_PLS_non_soe_stocks)


# In[14]:


#不同数据集下的R2oos
chart = np.array([[R2OOS_PLS],
                  [R2OOS_PLS_large_stocks],
                  [R2OOS_PLS_small_stocks],
                  [R2OOS_PLS_large_shareholder],
                  [R2OOS_PLS_small_shareholder],
                  [R2OOS_PLS_soe_stocks],
                  [R2OOS_PLS_non_soe_stocks]])
                     
r2oos_pls = pd.DataFrame(chart, columns=['PLS'],
                              index=["All", "Top 70%", "Bottom 30%", 
                                     "A.M.C.P.S. Top 70%", "A.M.C.P.S. Bottom 30%", "SOE", "Non-SOE"])

r2oos_pls


# In[15]:


r2oos_pls.to_csv(r'r2oos results/r2oos_pls.csv')


# ### 变量重要性
# 原文定义了变量重要性，即对于每个模型，给定预测因子的所有值都设置为零，并计算预测 R2OOS 的减少。然后，将预测 R2OOS 的绝对减少量归一化为总和 1，表示每个变量对模型的相对贡献。变量重要性是根据最后一个滚动窗口观察样本计算的，并不代表整个样本的平均值。

# In[6]:


# 为每个自变量生成一个单独的 DataFrame，其中该变量的所有值都设置为零
for j in features:
    globals()['df_' + str(j)] =  df.copy()
    globals()['df_' + str(j)][str(j)] = 0


# In[7]:


#例子chmom
df_chmom


# In[8]:


import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error

# 初始化字典和参数
dic = {}    
numpc = np.arange(1, 18, 1).tolist()
mse = np.full((len(numpc), 1), np.nan, dtype=np.float32)

for j in features:
    df_var = globals()['df_' + str(j)]
    
    # 创建年份列
    df_var["year"] = df_var["交易月份"].dt.year
    
    # 定义训练和验证集
    X_train = df_var[features].loc[(df_var["year"] >= 2016) & (df_var["year"] <= 2021)]
    y_train = df_var["mom1m"].loc[(df_var["year"] >= 2016) & (df_var["year"] <= 2021)]

    X_val = df_var[features].loc[(df_var["year"] >= 2021) & (df_var["year"] <= 2023)]
    y_val = df_var["mom1m"].loc[(df_var["year"] >= 2021) & (df_var["year"] <= 2023)]

    # 替换无穷大值为 0
    X_train.replace([np.inf, -np.inf], 0, inplace=True)
    X_val.replace([np.inf, -np.inf], 0, inplace=True)

    # 检查样本数量
    if X_train.shape[0] < 2 or X_val.shape[0] < 2:
        print(f"Not enough samples for feature {j}: X_train samples: {X_train.shape[0]}, X_val samples: {X_val.shape[0]}")
        dic['R2OOS_' + str(j)] = np.nan  # 或其他适当的值
        continue  # 跳过当前特征的后续处理

    for i in range(len(numpc)):
        pls_val = PLSRegression(n_components=numpc[i], scale=False)
        pls_val.fit(X_train, y_train)
        Yval_predict = pls_val.predict(X_val)
        Yval_predict = Yval_predict.ravel()
        mse[i, 0] = np.sqrt(mean_squared_error(y_val, Yval_predict))

    optim_numpc = numpc[np.argmin(mse)]
    pls = PLSRegression(n_components=optim_numpc, scale=False)
    pls.fit(np.concatenate((X_train, X_val)), np.concatenate((y_train, y_val)))
    preds = pls.predict(np.concatenate((X_train, X_val))) 
    preds = preds.ravel()
    
    R2OOS_var = 1 - sum(pow(np.concatenate((y_train, y_val)) - preds, 2)) / sum(pow(np.concatenate((y_train, y_val)), 2))
    dic['R2OOS_' + str(j)] = R2OOS_var

# 输出字典 dic
print(dic)


# In[9]:


dic


# In[16]:


import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error

features = df.columns[~df.columns.isin(['证券代码', "mom1m"])].tolist()
df["year"] = df["交易月份"].dt.year

# 定义训练和验证集
X_train = df[features].loc[(df["year"] >= 2016) & (df["year"] <= 2021)]
y_train = df["mom1m"].loc[(df["year"] >= 2016) & (df["year"] <= 2021)]

X_val = df[features].loc[(df["year"] >= 2021) & (df["year"] <= 2023)]
y_val = df["mom1m"].loc[(df["year"] >= 2021) & (df["year"] <= 2023)]

# 仅选择数值型特征
X_train = X_train.select_dtypes(include=[np.number])
X_val = X_val.select_dtypes(include=[np.number])

numpc = np.arange(1, 18, 1).tolist()
mse = np.full((len(numpc), 1), np.nan, dtype=np.float32)

# 替换无穷大值为 0
X_train.replace([np.inf, -np.inf], 0, inplace=True)
X_val.replace([np.inf, -np.inf], 0, inplace=True)

# 检查样本数量
if X_train.shape[0] < 2 or X_val.shape[0] < 2:
    print(f"Not enough samples: X_train samples: {X_train.shape[0]}, X_val samples: {X_val.shape[0]}")
else:
    for i in range(len(numpc)):
        pls_val = PLSRegression(n_components=numpc[i], scale=False)
        pls_val.fit(X_train, y_train)
        Yval_predict = pls_val.predict(X_val)
        Yval_predict = Yval_predict.ravel()
        mse[i, 0] = np.sqrt(mean_squared_error(y_val, Yval_predict))

    optim_numpc = numpc[np.argmin(mse)]
    pls = PLSRegression(n_components=optim_numpc, scale=False)
    pls.fit(np.concatenate((X_train, X_val)), np.concatenate((y_train, y_val)))
    preds = pls.predict(np.concatenate((X_train, X_val))) 
    preds = preds.ravel()

    R2OOS_all = 1 - sum(pow(np.concatenate((y_train, y_val)) - preds, 2)) / sum(pow(np.concatenate((y_train, y_val)), 2))
    print(f"R² OOS: {R2OOS_all}")


# In[18]:


pd.DataFrame(dic.items())
imp=pd.DataFrame(dic.items(), columns=['Feature', 'R2OOS'])
imp["Feature"] = imp["Feature"].str[6:]

# 计算预测 R2OOS 的减少 
imp["reduce_R2OOS"] = R2OOS_all -imp["R2OOS"]
imp["var_imp"] = imp["reduce_R2OOS"]/sum(imp["reduce_R2OOS"])
imp=imp.sort_values(by = ['var_imp'], ascending = False)
imp


# In[23]:


imp.to_csv('pls_variable_importance.csv')


# In[21]:


# 可视化变量重要性
fea_imp_graph = imp.sort_values(['var_imp', 'Feature'], ascending=[True, False]).iloc[-20:]
_ = fea_imp_graph.plot(kind='barh', x='Feature', y='var_imp', figsize=(20, 10))
plt.title('PLS')
plt.savefig('pls_variable_importance.png', bbox_inches='tight', dpi=600)
plt.show()

