#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import math
import sklearn
import sklearn.preprocessing
import datetime
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
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


# In[5]:


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


# # Gradient boosted regression trees（梯度提升树）
# 
# ### Tuning parameters:
# 
# - shrinkage/learning weight **𝜈**: **learning_rate**
# - debth of the simple trees **𝐿**: **max_depth**
# - the number of trees **𝐵** added to the ensemble model: **n_estimators** 
# 
# ## 所有股票

# In[6]:


def huber_loss_error(y_true, y_pred, delta=1.35):
    error = y_true - y_pred
    is_small_error = np.abs(error) <= delta
    squared_loss = 0.5 * np.square(error)
    linear_loss = delta * (np.abs(error) - 0.5 * delta)
    return np.sum(np.where(is_small_error, squared_loss, linear_loss))

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

# Model’s complexity: dictionary to save the number of predictors randomly considered as potential split variables over time
dic_max_depth_all = {}

for train_index, val_index, test_index in tscv.split(X, first_split_date=datetime.date(2005, 1, 31), second_split_date=datetime.date(2013, 1, 31)):

    X_train = X.loc[train_index].drop('交易月份', axis=1)
    y_train = y.loc[train_index]
    
    X_val = X.loc[val_index].drop('交易月份', axis=1)
    y_val = y.loc[val_index]

    X_test = X.loc[test_index].drop('交易月份', axis=1)
    y_test = y.loc[test_index]

    # 使用默认参数初始化模型
    GBR = GradientBoostingRegressor(loss='huber')

    GBR.fit(np.concatenate((X_train, X_val)), np.concatenate((y_train, y_val)))
    
    preds = GBR.predict(X_test)

    
    predictions.append(preds)
    dates.append(y_test.index)
    y_test_list.append(y_test)

    r2 = 1 - sum(pow(y_test - preds, 2)) / sum(pow(y_test, 2))
    dic_r2_all["r2." + str(y_test.index)] = r2


predictions_all = np.concatenate(predictions, axis=0)
y_test_list_all = np.concatenate(y_test_list, axis=0)
dates_all = np.concatenate(dates, axis=0)

R2OOS_GBR = 1 - sum(pow(y_test_list_all - predictions_all, 2)) / sum(pow(y_test_list_all, 2))
print("R2OOS gradient boosted regression tree: ", R2OOS_GBR)


# In[10]:


import warnings
import datetime
import numpy as np
from sklearn.model_selection import ParameterGrid
from xgboost import XGBRegressor

def huber_loss_error(y_true, y_pred, delta=1.35):
    error = y_true - y_pred
    is_small_error = np.abs(error) <= delta
    squared_loss = 0.5 * np.square(error)
    linear_loss = delta * (np.abs(error) - 0.5 * delta)
    return np.sum(np.where(is_small_error, squared_loss, linear_loss))

# Suppress warnings
warnings.simplefilter(action='ignore', category=Warning)

# Assuming TimeBasedCV is defined elsewhere in your code
tscv = TimeBasedCV(train_period=60, val_period=24, test_period=12, freq='months')

features = df.columns[~df.columns.isin(['证券代码', "mom1m"])].tolist()
X = df[features]
y = df["mom1m"]

predictions = []
y_test_list = []
dates = []
dic_r2_all = {}
dic_max_depth_all = {}

param_grid = {
    'max_depth': [1, 2, 3, 4, 5, 6, 7, 8],  
    'n_estimators': [100], 
    "learning_rate": [0.01, 0.1],  
    "colsample_bytree": [0.5, 0.75, 1.0],  
    "min_child_weight": [50, 100, 200],  
}

grid = list(ParameterGrid(param_grid))
huber_loss = np.full((len(grid), 1), np.nan, dtype=np.float32)

for fold, (train_index, val_index, test_index) in enumerate(tscv.split(X, first_split_date=datetime.date(2005, 1, 31), second_split_date=datetime.date(2013, 1, 31))):
    
    X_train = X.loc[train_index].drop('交易月份', axis=1)
    y_train = y.loc[train_index]
    
    X_val = X.loc[val_index].drop('交易月份', axis=1)
    y_val = y.loc[val_index]

    X_test = X.loc[test_index].drop('交易月份', axis=1)
    y_test = y.loc[test_index]

    # Hyperparameter tuning
    for i in range(len(grid)):
        XGB_val = XGBRegressor(objective='reg:squarederror',
                                max_depth=grid[i]["max_depth"], 
                                learning_rate=grid[i]["learning_rate"],
                                n_estimators=grid[i]["n_estimators"],
                                subsample=0.8,  # Optional: to prevent overfitting
                                colsample_bytree=grid[i]["colsample_bytree"],
                                min_child_weight=grid[i]["min_child_weight"])
        
        XGB_val.fit(X_train, y_train)
        Yval_predict = XGB_val.predict(X_val)
        huber_loss[i, 0] = huber_loss_error(y_val, Yval_predict, delta=1.35)

    # Optimal hyperparameters
    optim_param = grid[np.argmin(huber_loss)]

    # Fit final model
    XGB = XGBRegressor(objective='reg:squarederror',
                        max_depth=optim_param["max_depth"], 
                        learning_rate=optim_param["learning_rate"],
                        n_estimators=optim_param["n_estimators"],
                        subsample=0.8,
                        colsample_bytree=optim_param["colsample_bytree"],
                        min_child_weight=optim_param["min_child_weight"])

    XGB.fit(np.concatenate((X_train, X_val)), np.concatenate((y_train, y_val)))
    preds = XGB.predict(X_test)

    # Save predictions and calculate R²
    predictions.append(preds)
    dates.append(y_test.index)
    y_test_list.append(y_test)

    r2 = 1 - sum(pow(y_test - preds, 2)) / sum(pow(y_test, 2))
    dic_r2_all["r2." + str(y_test.index)] = r2
    dic_max_depth_all["feat." + str(y_test.index)] = optim_param["max_depth"]

# Concatenate results
predictions_all = np.concatenate(predictions, axis=0)
y_test_list_all = np.concatenate(y_test_list, axis=0) 
dates_all = np.concatenate(dates, axis=0)

R2OOS_XGB = 1 - sum(pow(y_test_list_all - predictions_all, 2)) / sum(pow(y_test_list_all, 2))
print("R² OOS XGBoost regression tree: ", R2OOS_XGB)


# # 大股票

# In[12]:


warnings.simplefilter(action='ignore', category=Warning)

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
dic_r2_large_stocks = {}
dic_max_depth_large_stocks = {}


param_grid = {
    'max_depth': [1, 2, 3, 4, 5, 6, 7, 8],  
    'n_estimators': [100], 
    "learning_rate": [0.01, 0.1],  
    "colsample_bytree": [0.5, 0.75, 1.0],  
    "min_child_weight": [50, 100, 200],  
}

grid = list(ParameterGrid(param_grid))
huber_loss = np.full((len(grid), 1), np.nan, dtype=np.float32)

for fold, (train_index, val_index, test_index) in enumerate(tscv.split(X, first_split_date=datetime.date(2005, 1, 31), second_split_date=datetime.date(2013, 1, 31))):
    
    X_train = X.loc[train_index].drop('交易月份', axis=1)
    y_train = y.loc[train_index]
    
    X_val = X.loc[val_index].drop('交易月份', axis=1)
    y_val = y.loc[val_index]

    X_test = X.loc[test_index].drop('交易月份', axis=1)
    y_test = y.loc[test_index]

    # Hyperparameter tuning
    for i in range(len(grid)):
        XGB_val = XGBRegressor(objective='reg:squarederror',
                                max_depth=grid[i]["max_depth"], 
                                learning_rate=grid[i]["learning_rate"],
                                n_estimators=grid[i]["n_estimators"],
                                subsample=0.8,  # Optional: to prevent overfitting
                                colsample_bytree=grid[i]["colsample_bytree"],
                                min_child_weight=grid[i]["min_child_weight"])
        
        XGB_val.fit(X_train, y_train)
        Yval_predict = XGB_val.predict(X_val)
        huber_loss[i, 0] = huber_loss_error(y_val, Yval_predict, delta=1.35)

    # Optimal hyperparameters
    optim_param = grid[np.argmin(huber_loss)]

    # Fit final model
    XGB = XGBRegressor(objective='reg:squarederror',
                        max_depth=optim_param["max_depth"], 
                        learning_rate=optim_param["learning_rate"],
                        n_estimators=optim_param["n_estimators"],
                        subsample=0.8,
                        colsample_bytree=optim_param["colsample_bytree"],
                        min_child_weight=optim_param["min_child_weight"])

    XGB.fit(np.concatenate((X_train, X_val)), np.concatenate((y_train, y_val)))
    preds = XGB.predict(X_test)

    # Save predictions and calculate R²
    predictions_large_stocks.append(preds)
    dates_large_stocks.append(y_test.index)
    y_test_list_large_stocks.append(y_test)

    r2 = 1 - sum(pow(y_test - preds, 2)) / sum(pow(y_test, 2))
    dic_r2_large_stocks["r2." + str(y_test.index)] = r2
    dic_max_depth_large_stocks["feat." + str(y_test.index)] = optim_param["max_depth"]

# Concatenate results
predictions_large_stocks= np.concatenate(predictions_large_stocks, axis=0)
y_test_list_large_stocks= np.concatenate(y_test_list_large_stocks, axis=0) 
dates_large_stocks= np.concatenate(dates_large_stocks, axis=0)

R2OOS_XGB_large_stocks = 1 - sum(pow(y_test_list_large_stocks - predictions_large_stocks, 2)) / sum(pow(y_test_list_large_stocks, 2))
print("R² OOS XGBoost regression tree Large Stocks: ", R2OOS_XGB_large_stocks)


# # 小股票

# In[ ]:


import warnings
import datetime
import numpy as np
from sklearn.model_selection import ParameterGrid
from xgboost import XGBRegressor

warnings.simplefilter(action='ignore', category=Warning)

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
dic_r2_small_stocks = {}
dic_max_depth_small_stocks = {}


param_grid = {
    'max_depth': [1, 2, 3, 4, 5, 6, 7, 8],  
    'n_estimators': [100], 
    "learning_rate": [0.01, 0.1],  
    "colsample_bytree": [0.5, 0.75, 1.0],  
    "min_child_weight": [50, 100, 200],  
}

grid = list(ParameterGrid(param_grid))
huber_loss = np.full((len(grid), 1), np.nan, dtype=np.float32)

for fold, (train_index, val_index, test_index) in enumerate(tscv.split(X, first_split_date=datetime.date(2005, 1, 31), second_split_date=datetime.date(2013, 1, 31))):
    
    X_train = X.loc[train_index].drop('交易月份', axis=1)
    y_train = y.loc[train_index]
    
    X_val = X.loc[val_index].drop('交易月份', axis=1)
    y_val = y.loc[val_index]

    X_test = X.loc[test_index].drop('交易月份', axis=1)
    y_test = y.loc[test_index]

    # Hyperparameter tuning
    for i in range(len(grid)):
        XGB_val = XGBRegressor(objective='reg:squarederror',
                                max_depth=grid[i]["max_depth"], 
                                learning_rate=grid[i]["learning_rate"],
                                n_estimators=grid[i]["n_estimators"],
                                subsample=0.8,  # Optional: to prevent overfitting
                                colsample_bytree=grid[i]["colsample_bytree"],
                                min_child_weight=grid[i]["min_child_weight"])
        
        XGB_val.fit(X_train, y_train)
        Yval_predict = XGB_val.predict(X_val)
        huber_loss[i, 0] = huber_loss_error(y_val, Yval_predict, delta=1.35)

    # Optimal hyperparameters
    optim_param = grid[np.argmin(huber_loss)]

    # Fit final model
    XGB = XGBRegressor(objective='reg:squarederror',
                        max_depth=optim_param["max_depth"], 
                        learning_rate=optim_param["learning_rate"],
                        n_estimators=optim_param["n_estimators"],
                        subsample=0.8,
                        colsample_bytree=optim_param["colsample_bytree"],
                        min_child_weight=optim_param["min_child_weight"])

    XGB.fit(np.concatenate((X_train, X_val)), np.concatenate((y_train, y_val)))
    preds = XGB.predict(X_test)

    # Save predictions and calculate R²
    predictions_small_stocks.append(preds)
    dates_small_stocks.append(y_test.index)
    y_test_list_small_stocks.append(y_test)

    r2 = 1 - sum(pow(y_test - preds, 2)) / sum(pow(y_test, 2))
    dic_r2_small_stocks["r2." + str(y_test.index)] = r2
    dic_max_depth_small_stocks["feat." + str(y_test.index)] = optim_param["max_depth"]

predictions_all_small_stocks= np.concatenate(predictions_small_stocks, axis=0)
y_test_list_all_small_stocks= np.concatenate(y_test_list_small_stocks, axis=0) 
dates_all_small_stocks= np.concatenate(dates_small_stocks, axis=0)

R2OOS_XGB_small_stocks = 1 - sum(pow(y_test_list_all_small_stocks - predictions_all_small_stocks, 2)) / sum(pow(y_test_list_all_small_stocks, 2))
print("R² OOS XGBoost regression tree Small Stocks: ", R2OOS_XGB_small_stocks)


# In[10]:


R2OOS_XGB_small_stocks = 1 - sum(pow(y_test_list_all_small_stocks - predictions_all_small_stocks, 2)) / sum(pow(y_test_list_all_small_stocks, 2))
print("R² OOS XGBoost regression tree Small Stocks: ", R2OOS_XGB_small_stocks)


# # 大股东

# In[ ]:


import warnings
import datetime
import numpy as np
from sklearn.model_selection import ParameterGrid
from xgboost import XGBRegressor

warnings.simplefilter(action='ignore', category=Warning)

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
dic_r2_large_shareholder = {}
dic_max_depth_large_shareholder = {}


param_grid = {
    'max_depth': [1, 2, 3, 4, 5, 6, 7, 8],  
    'n_estimators': [100], 
    "learning_rate": [0.01, 0.1],  
    "colsample_bytree": [0.5, 0.75, 1.0],  
    "min_child_weight": [50, 100, 200],  
}

grid = list(ParameterGrid(param_grid))
huber_loss = np.full((len(grid), 1), np.nan, dtype=np.float32)

for fold, (train_index, val_index, test_index) in enumerate(tscv.split(X, first_split_date=datetime.date(2005, 1, 31), second_split_date=datetime.date(2013, 1, 31))):
    
    X_train = X.loc[train_index].drop('交易月份', axis=1)
    y_train = y.loc[train_index]
    
    X_val = X.loc[val_index].drop('交易月份', axis=1)
    y_val = y.loc[val_index]

    X_test = X.loc[test_index].drop('交易月份', axis=1)
    y_test = y.loc[test_index]

    # Hyperparameter tuning
    for i in range(len(grid)):
        XGB_val = XGBRegressor(objective='reg:squarederror',
                                max_depth=grid[i]["max_depth"], 
                                learning_rate=grid[i]["learning_rate"],
                                n_estimators=grid[i]["n_estimators"],
                                subsample=0.8,  # Optional: to prevent overfitting
                                colsample_bytree=grid[i]["colsample_bytree"],
                                min_child_weight=grid[i]["min_child_weight"])
        
        XGB_val.fit(X_train, y_train)
        Yval_predict = XGB_val.predict(X_val)
        huber_loss[i, 0] = huber_loss_error(y_val, Yval_predict, delta=1.35)

    # Optimal hyperparameters
    optim_param = grid[np.argmin(huber_loss)]

    # Fit final model
    XGB = XGBRegressor(objective='reg:squarederror',
                        max_depth=optim_param["max_depth"], 
                        learning_rate=optim_param["learning_rate"],
                        n_estimators=optim_param["n_estimators"],
                        subsample=0.8,
                        colsample_bytree=optim_param["colsample_bytree"],
                        min_child_weight=optim_param["min_child_weight"])

    XGB.fit(np.concatenate((X_train, X_val)), np.concatenate((y_train, y_val)))
    preds = XGB.predict(X_test)

    # Save predictions and calculate R²
    predictions_large_shareholder.append(preds)
    dates_large_shareholder.append(y_test.index)
    y_test_list_large_shareholder.append(y_test)

    r2 = 1 - sum(pow(y_test - preds, 2)) / sum(pow(y_test, 2))
    dic_r2_large_shareholder["r2." + str(y_test.index)] = r2
    dic_max_depth_large_shareholder["feat." + str(y_test.index)] = optim_param["max_depth"]

predictions_all_large_shareholder= np.concatenate(predictions_large_shareholder, axis=0)
y_test_list_all_large_shareholder= np.concatenate(y_test_list_large_shareholder, axis=0) 
dates_all_large_shareholder= np.concatenate(dates_large_shareholder, axis=0)

R2OOS_XGB_large_shareholder = 1 - sum(pow(y_test_list_all_large_shareholder - predictions_all_large_shareholder, 2)) / sum(pow(y_test_list_all_large_shareholder, 2))
print("R² OOS XGBoost regression tree Large Shareholder: ", R2OOS_XGB_large_shareholder)


# In[12]:


R2OOS_XGB_large_shareholder = 1 - sum(pow(y_test_list_all_large_shareholder - predictions_all_large_shareholder, 2)) / sum(pow(y_test_list_all_large_shareholder, 2))
print("R² OOS XGBoost regression tree Large Shareholder: ", R2OOS_XGB_large_shareholder)


# # 小股东

# In[13]:


import warnings
import datetime
import numpy as np
from sklearn.model_selection import ParameterGrid
from xgboost import XGBRegressor

warnings.simplefilter(action='ignore', category=Warning)

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
dic_r2_small_shareholder = {}
dic_max_depth_small_shareholder = {}


param_grid = {
    'max_depth': [1, 2, 3, 4, 5, 6, 7, 8],  
    'n_estimators': [100], 
    "learning_rate": [0.01, 0.1],  
    "colsample_bytree": [0.5, 0.75, 1.0],  
    "min_child_weight": [50, 100, 200],  
}

grid = list(ParameterGrid(param_grid))
huber_loss = np.full((len(grid), 1), np.nan, dtype=np.float32)

for fold, (train_index, val_index, test_index) in enumerate(tscv.split(X, first_split_date=datetime.date(2005, 1, 31), second_split_date=datetime.date(2013, 1, 31))):
    
    X_train = X.loc[train_index].drop('交易月份', axis=1)
    y_train = y.loc[train_index]
    
    X_val = X.loc[val_index].drop('交易月份', axis=1)
    y_val = y.loc[val_index]

    X_test = X.loc[test_index].drop('交易月份', axis=1)
    y_test = y.loc[test_index]

    # Hyperparameter tuning
    for i in range(len(grid)):
        XGB_val = XGBRegressor(objective='reg:squarederror',
                                max_depth=grid[i]["max_depth"], 
                                learning_rate=grid[i]["learning_rate"],
                                n_estimators=grid[i]["n_estimators"],
                                subsample=0.8,  # Optional: to prevent overfitting
                                colsample_bytree=grid[i]["colsample_bytree"],
                                min_child_weight=grid[i]["min_child_weight"])
        
        XGB_val.fit(X_train, y_train)
        Yval_predict = XGB_val.predict(X_val)
        huber_loss[i, 0] = huber_loss_error(y_val, Yval_predict, delta=1.35)

    # Optimal hyperparameters
    optim_param = grid[np.argmin(huber_loss)]

    # Fit final model
    XGB = XGBRegressor(objective='reg:squarederror',
                        max_depth=optim_param["max_depth"], 
                        learning_rate=optim_param["learning_rate"],
                        n_estimators=optim_param["n_estimators"],
                        subsample=0.8,
                        colsample_bytree=optim_param["colsample_bytree"],
                        min_child_weight=optim_param["min_child_weight"])

    XGB.fit(np.concatenate((X_train, X_val)), np.concatenate((y_train, y_val)))
    preds = XGB.predict(X_test)

    # Save predictions and calculate R²
    predictions_small_shareholder.append(preds)
    dates_small_shareholder.append(y_test.index)
    y_test_list_small_shareholder.append(y_test)

    r2 = 1 - sum(pow(y_test - preds, 2)) / sum(pow(y_test, 2))
    dic_r2_small_shareholder["r2." + str(y_test.index)] = r2
    dic_max_depth_small_shareholder["feat." + str(y_test.index)] = optim_param["max_depth"]

predictions_all_small_shareholder= np.concatenate(predictions_small_shareholder, axis=0)
y_test_list_all_small_shareholder= np.concatenate(y_test_list_small_shareholder, axis=0) 
dates_all_small_shareholder= np.concatenate(dates_small_shareholder, axis=0)

R2OOS_XGB_small_shareholder = 1 - sum(pow(y_test_list_all_small_shareholder - predictions_all_small_shareholder, 2)) / sum(pow(y_test_list_all_small_shareholder, 2))
print("R² OOS XGBoost regression tree Small Shareholder: ", R2OOS_XGB_small_shareholder)


# # 国企

# In[16]:


import warnings
import datetime
import numpy as np
from sklearn.model_selection import ParameterGrid
from xgboost import XGBRegressor

warnings.simplefilter(action='ignore', category=Warning)

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
dic_max_depth_soe_stocks={}

param_grid = {
    'max_depth': [1, 2, 3, 4, 5, 6, 7, 8],  
    'n_estimators': [100], 
    "learning_rate": [0.01, 0.1],  
    "colsample_bytree": [0.5, 0.75, 1.0],  
    "min_child_weight": [50, 100, 200],  
}

grid = list(ParameterGrid(param_grid))
huber_loss = np.full((len(grid), 1), np.nan, dtype=np.float32)

for fold, (train_index, val_index, test_index) in enumerate(tscv.split(X, first_split_date=datetime.date(2005, 1, 31), second_split_date=datetime.date(2013, 1, 31))):
    
    X_train = X.loc[train_index].drop('交易月份', axis=1)
    y_train = y.loc[train_index]
    
    X_val = X.loc[val_index].drop('交易月份', axis=1)
    y_val = y.loc[val_index]

    X_test = X.loc[test_index].drop('交易月份', axis=1)
    y_test = y.loc[test_index]

    # Hyperparameter tuning
    for i in range(len(grid)):
        XGB_val = XGBRegressor(objective='reg:squarederror',
                                max_depth=grid[i]["max_depth"], 
                                learning_rate=grid[i]["learning_rate"],
                                n_estimators=grid[i]["n_estimators"],
                                subsample=0.8,  # Optional: to prevent overfitting
                                colsample_bytree=grid[i]["colsample_bytree"],
                                min_child_weight=grid[i]["min_child_weight"])
        
        XGB_val.fit(X_train, y_train)
        Yval_predict = XGB_val.predict(X_val)
        huber_loss[i, 0] = huber_loss_error(y_val, Yval_predict, delta=1.35)

    # Optimal hyperparameters
    optim_param = grid[np.argmin(huber_loss)]

    # Fit final model
    XGB = XGBRegressor(objective='reg:squarederror',
                        max_depth=optim_param["max_depth"], 
                        learning_rate=optim_param["learning_rate"],
                        n_estimators=optim_param["n_estimators"],
                        subsample=0.8,
                        colsample_bytree=optim_param["colsample_bytree"],
                        min_child_weight=optim_param["min_child_weight"])

    XGB.fit(np.concatenate((X_train, X_val)), np.concatenate((y_train, y_val)))
    preds = XGB.predict(X_test)

    # Save predictions and calculate R²
    predictions_soe_stocks.append(preds)
    dates_soe_stocks.append(y_test.index)
    y_test_list_soe_stocks.append(y_test)

    r2 = 1 - sum(pow(y_test - preds, 2)) / sum(pow(y_test, 2))
    dic_r2_all_soe_stocks["r2." + str(y_test.index)] = r2
    dic_max_depth_soe_stocks["feat." + str(y_test.index)] = optim_param["max_depth"]

predictions_all_soe_stocks= np.concatenate(predictions_soe_stocks, axis=0)
y_test_list_all_soe_stocks= np.concatenate(y_test_list_soe_stocks, axis=0) 
dates_all_soe_stocks= np.concatenate(dates_soe_stocks, axis=0)

R2OOS_XGB_soe_stocks = 1 - sum(pow(y_test_list_all_soe_stocks - predictions_all_soe_stocks, 2)) / sum(pow(y_test_list_all_soe_stocks, 2))
print("R² OOS XGBoost regression tree Soe Stocks: ", R2OOS_XGB_soe_stocks)


# # 非国企

# In[17]:


import warnings
import datetime
import numpy as np
from sklearn.model_selection import ParameterGrid
from xgboost import XGBRegressor

warnings.simplefilter(action='ignore', category=Warning)

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
dic_max_depth_non_soe_stocks={}

param_grid = {
    'max_depth': [1, 2, 3, 4, 5, 6, 7, 8],  
    'n_estimators': [100], 
    "learning_rate": [0.01, 0.1],  
    "colsample_bytree": [0.5, 0.75, 1.0],  
    "min_child_weight": [50, 100, 200],  
}

grid = list(ParameterGrid(param_grid))
huber_loss = np.full((len(grid), 1), np.nan, dtype=np.float32)

for fold, (train_index, val_index, test_index) in enumerate(tscv.split(X, first_split_date=datetime.date(2005, 1, 31), second_split_date=datetime.date(2013, 1, 31))):
    
    X_train = X.loc[train_index].drop('交易月份', axis=1)
    y_train = y.loc[train_index]
    
    X_val = X.loc[val_index].drop('交易月份', axis=1)
    y_val = y.loc[val_index]

    X_test = X.loc[test_index].drop('交易月份', axis=1)
    y_test = y.loc[test_index]

    # Hyperparameter tuning
    for i in range(len(grid)):
        XGB_val = XGBRegressor(objective='reg:squarederror',
                                max_depth=grid[i]["max_depth"], 
                                learning_rate=grid[i]["learning_rate"],
                                n_estimators=grid[i]["n_estimators"],
                                subsample=0.8,  # Optional: to prevent overfitting
                                colsample_bytree=grid[i]["colsample_bytree"],
                                min_child_weight=grid[i]["min_child_weight"])
        
        XGB_val.fit(X_train, y_train)
        Yval_predict = XGB_val.predict(X_val)
        huber_loss[i, 0] = huber_loss_error(y_val, Yval_predict, delta=1.35)

    # Optimal hyperparameters
    optim_param = grid[np.argmin(huber_loss)]

    # Fit final model
    XGB = XGBRegressor(objective='reg:squarederror',
                        max_depth=optim_param["max_depth"], 
                        learning_rate=optim_param["learning_rate"],
                        n_estimators=optim_param["n_estimators"],
                        subsample=0.8,
                        colsample_bytree=optim_param["colsample_bytree"],
                        min_child_weight=optim_param["min_child_weight"])

    XGB.fit(np.concatenate((X_train, X_val)), np.concatenate((y_train, y_val)))
    preds = XGB.predict(X_test)

    # Save predictions and calculate R²
    predictions_non_soe_stocks.append(preds)
    dates_non_soe_stocks.append(y_test.index)
    y_test_list_non_soe_stocks.append(y_test)

    r2 = 1 - sum(pow(y_test - preds, 2)) / sum(pow(y_test, 2))
    dic_r2_all_non_soe_stocks["r2." + str(y_test.index)] = r2
    dic_max_depth_non_soe_stocks["feat." + str(y_test.index)] = optim_param["max_depth"]

predictions_all_non_soe_stocks= np.concatenate(predictions_non_soe_stocks, axis=0)
y_test_list_all_non_soe_stocks= np.concatenate(y_test_list_non_soe_stocks, axis=0) 
dates_all_non_soe_stocks= np.concatenate(dates_non_soe_stocks, axis=0)

R2OOS_XGB_non_soe_stocks = 1 - sum(pow(y_test_list_all_non_soe_stocks - predictions_all_non_soe_stocks, 2)) / sum(pow(y_test_list_all_non_soe_stocks, 2))
print("R² OOS XGBoost regression tree non-Soe Stocks: ", R2OOS_XGB_non_soe_stocks)


# In[21]:


R2OOS_XGB = -0.028528697577702156
R2OOS_XGB_large_stocks = -0.003900853190877074


# In[22]:


#不同数据集下的R2oos
chart = np.array([[R2OOS_XGB],
                  [R2OOS_XGB_large_stocks],
                  [R2OOS_XGB_small_stocks],
                  [R2OOS_XGB_large_shareholder],
                  [R2OOS_XGB_small_shareholder],
                  [R2OOS_XGB_soe_stocks],
                  [R2OOS_XGB_non_soe_stocks]])
                     
r2oos_gbrt = pd.DataFrame(chart, columns=['GBRT(+H)'],
                              index=["All", "Top 70%", "Bottom 30%", 
                                     "A.M.C.P.S. Top 70%", "A.M.C.P.S. Bottom 30%", "SOE", "Non-SOE"])

r2oos_gbrt


# In[23]:


r2oos_gbrt.to_csv(r'r2oos results/r2oos_gbrt.csv')

