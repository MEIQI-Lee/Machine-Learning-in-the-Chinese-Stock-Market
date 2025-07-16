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
from sklearn.linear_model import ElasticNet
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
import warnings
warnings.simplefilter(action='ignore', category=Warning)
import seaborn as sns
sns.set()
pd.options.mode.chained_assignment = None  # default='warn'

#执行滚动窗口 train-val-test 拆分的Program
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


# # Elastic Net 弹性网模型
# 
# ### 公式:
# 
# $\phi(\theta ; \lambda, \rho)=\lambda(1-\rho) \sum_{j=1}^{P}\left|\theta_{j}\right|+\frac{1}{2} \lambda \rho \sum_{j=1}^{P} \theta_{j}^{2}
# $
# 
# - overall strength of the penalty, $\lambda \in(0,1)$
# - $\rho \in(0,1)$, regulating the weight of the lasso and ridge penalization, where $p=0$ corresponds to the lasso and $p=1$ to the ridge method. Here $\rho$ is set to $0.5 .$ 
# 
# - In sklearn the $\lambda$ parameter is called  **alpha** and $\rho $ is called **l1_ratio**
# ## 所有股票

# In[7]:


tscv = TimeBasedCV(train_period=60,
                   val_period=24,
                   test_period=12,
                   freq='months')

features = df.columns[~df.columns.isin(['证券代码',"mom1m"])].tolist()
X = df[features]
y = df["mom1m"]

#Empty containers to save results from each window
predictions = []
y_test_list =[]
dates = []
dic_r2_all = {}

# Model’s complexity: dictionary to save the number of characteristics over time
num_coef_time = {}

# List of values to use for the alpha hyperparameter
alphas = np.linspace(start=0.00001,stop=0.004,num=20)
# Empty container to save the objective loss function (mean squared errors) for each alpha
mse = np.full((len(alphas),1),np.nan, dtype = np.float32)


for train_index, val_index, test_index in tscv.split(X, first_split_date= datetime.date(2005,1,31), second_split_date= datetime.date(2013,1,31)):

    X_train   = X.loc[train_index].drop('交易月份', axis=1)
    y_train = y.loc[train_index]
    
    X_val   = X.loc[val_index].drop('交易月份', axis=1)
    y_val = y.loc[val_index]

    X_test    = X.loc[test_index].drop('交易月份', axis=1)
    y_test  = y.loc[test_index]
    
    #Loop over the list containing potential alpha values, fit on the training sample and use 
    #validation set to generate predictions
    for i in range(len(alphas)):
        model_val = ElasticNet(alpha=alphas[i], l1_ratio=0.5)
        model_val.fit(X_train,y_train)
        Yval_predict = model_val.predict(X_val)
        #calculate mean squared error for each potential value of the alpha hyperparameter
        mse[i,0] = np.sqrt(mean_squared_error(y_val,Yval_predict))
 
    #The optimal value of the alpha hyperparameter is the value that causes the lowest loss
    optim_alpha = alphas[np.argmin(mse)]
   
    #Fit again using the train and validation set and the optimal alpha parameter
    model = ElasticNet(alpha=optim_alpha, l1_ratio=0.5)
    model.fit(np.concatenate((X_train, X_val)), np.concatenate((y_train, y_val)))
    #Use test set to generate final predictions 
    preds = model.predict(X_test)

    #Save predictions, dates and the true values of the dependent variable to list  
    predictions.append(preds)
    dates.append(y_test.index)
    y_test_list.append(y_test)
    
    #Calculate OOS model performance the for current window
    r2 = 1-sum(pow(y_test-preds,2))/sum(pow(y_test,2))
    #Save OOS model performance and the respective month to dictionary
    dic_r2_all["r2." + str(y_test.index)] = r2
    # Save the number of characteristics to inspect  model's complexity over time 
    num_coef = len(model.coef_[np.nonzero(model.coef_ != 0)])
    num_coef_time["ncoef." + str(y_test.index)] = num_coef
   
        
#Concatenate to get results over the whole OOS test period (Jan 2010-Dec 2019)
predictions_all= np.concatenate(predictions, axis=0)
y_test_list_all= np.concatenate(y_test_list, axis=0) 
dates_all= np.concatenate(dates, axis=0)

#Calculate OOS model performance over the entire test period in line with Gu et al (2020)
R2OOS_ENet = 1-sum(pow(y_test_list_all-predictions_all,2))/sum(pow(y_test_list_all,2))
print("R2OOS Elastic net All Stocks: ", R2OOS_ENet)


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

alphas = np.linspace(start=0.00001,stop=0.004,num=20)
mse = np.full((len(alphas),1),np.nan, dtype = np.float32)


for train_index, val_index, test_index in tscv.split(X, first_split_date= datetime.date(2005,1,31), second_split_date= datetime.date(2013,1,31)):
    X_train   = X.loc[train_index].drop('交易月份', axis=1)
    y_train = y.loc[train_index]
    
    X_val   = X.loc[val_index].drop('交易月份', axis=1)
    y_val = y.loc[val_index]

    X_test    = X.loc[test_index].drop('交易月份', axis=1)
    y_test  = y.loc[test_index]

    for i in range(len(alphas)):
        model_val = ElasticNet(alpha=alphas[i], l1_ratio=0.5)
        model_val.fit(X_train,y_train)
        Yval_predict = model_val.predict(X_val)
        mse[i,0] = np.sqrt(mean_squared_error(y_val,Yval_predict))

    optim_alpha = alphas[np.argmin(mse)]
   
    model = ElasticNet(alpha=optim_alpha, l1_ratio=0.5)
    model.fit(np.concatenate((X_train, X_val)), np.concatenate((y_train, y_val))) 
    preds = model.predict(X_test)

    predictions_large_stocks.append(preds)
    dates_large_stocks.append(y_test.index)
    y_test_list_large_stocks.append(y_test)
    

    r2_large_stocks = r2_score(y_test, preds)
    dic_r2_all_large_stocks["r2." + str(y_test.index)] = r2
    
predictions_all_large_stocks= np.concatenate(predictions_large_stocks, axis=0)
y_test_list_all_large_stocks= np.concatenate(y_test_list_large_stocks, axis=0) 
dates_all_large_stocks= np.concatenate(dates_large_stocks, axis=0)

R2OOS_ENet_large_stocks = r2_score(y_test_list_all_large_stocks, predictions_all_large_stocks)
print("R2OOS Elastic net Large Stocks: ", R2OOS_ENet_large_stocks)


# # 小股票

# In[9]:


from sklearn.metrics import r2_score

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

alphas = np.linspace(start=0.00001,stop=0.004,num=20)
mse = np.full((len(alphas),1),np.nan, dtype = np.float32)


for train_index, val_index, test_index in tscv.split(X, first_split_date= datetime.date(2005,1,31), second_split_date= datetime.date(2013,1,31)):
    X_train   = X.loc[train_index].drop('交易月份', axis=1)
    y_train = y.loc[train_index]
    
    X_val   = X.loc[val_index].drop('交易月份', axis=1)
    y_val = y.loc[val_index]

    X_test    = X.loc[test_index].drop('交易月份', axis=1)
    y_test  = y.loc[test_index]

    for i in range(len(alphas)):
        model_val = ElasticNet(alpha=alphas[i], l1_ratio=0.5)
        model_val.fit(X_train,y_train)
        Yval_predict = model_val.predict(X_val)
        mse[i,0] = np.sqrt(mean_squared_error(y_val,Yval_predict))

    optim_alpha = alphas[np.argmin(mse)]
   
    model = ElasticNet(alpha=optim_alpha, l1_ratio=0.5)
    model.fit(np.concatenate((X_train, X_val)), np.concatenate((y_train, y_val))) 
    preds = model.predict(X_test)

    predictions_small_stocks.append(preds)
    dates_small_stocks.append(y_test.index)
    y_test_list_small_stocks.append(y_test)
    

    r2_small_stocks = r2_score(y_test, preds)
    dic_r2_all_small_stocks["r2." + str(y_test.index)] = r2
    
predictions_all_small_stocks= np.concatenate(predictions_small_stocks, axis=0)
y_test_list_all_small_stocks= np.concatenate(y_test_list_small_stocks, axis=0) 
dates_all_small_stocks= np.concatenate(dates_small_stocks, axis=0)

R2OOS_ENet_small_stocks = r2_score(y_test_list_all_small_stocks, predictions_all_small_stocks)
print("R2OOS Elastic net small Stocks: ", R2OOS_ENet_small_stocks)


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

alphas = np.linspace(start=0.00001,stop=0.004,num=20)
mse = np.full((len(alphas),1),np.nan, dtype = np.float32)


for train_index, val_index, test_index in tscv.split(X, first_split_date= datetime.date(2005,1,31), second_split_date= datetime.date(2013,1,31)):
    X_train   = X.loc[train_index].drop('交易月份', axis=1)
    y_train = y.loc[train_index]
    
    X_val   = X.loc[val_index].drop('交易月份', axis=1)
    y_val = y.loc[val_index]

    X_test    = X.loc[test_index].drop('交易月份', axis=1)
    y_test  = y.loc[test_index]

    for i in range(len(alphas)):
        model_val = ElasticNet(alpha=alphas[i], l1_ratio=0.5)
        model_val.fit(X_train,y_train)
        Yval_predict = model_val.predict(X_val)
        mse[i,0] = np.sqrt(mean_squared_error(y_val,Yval_predict))

    optim_alpha = alphas[np.argmin(mse)]
   
    model = ElasticNet(alpha=optim_alpha, l1_ratio=0.5)
    model.fit(np.concatenate((X_train, X_val)), np.concatenate((y_train, y_val))) 
    preds = model.predict(X_test)

    predictions_large_shareholder.append(preds)
    dates_large_shareholder.append(y_test.index)
    y_test_list_large_shareholder.append(y_test)
    

    r2_large_shareholder = r2_score(y_test, preds)
    dic_r2_all_large_shareholder["r2." + str(y_test.index)] = r2
    
predictions_all_large_shareholder= np.concatenate(predictions_large_shareholder, axis=0)
y_test_list_all_large_shareholder= np.concatenate(y_test_list_large_shareholder, axis=0) 
dates_all_large_shareholder= np.concatenate(dates_large_shareholder, axis=0)

R2OOS_ENet_large_shareholder = r2_score(y_test_list_all_large_shareholder, predictions_all_large_shareholder)
print("R2OOS Elastic net Large Shareholder: ", R2OOS_ENet_large_shareholder)


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

alphas = np.linspace(start=0.00001,stop=0.004,num=20)
mse = np.full((len(alphas),1),np.nan, dtype = np.float32)


for train_index, val_index, test_index in tscv.split(X, first_split_date= datetime.date(2005,1,31), second_split_date= datetime.date(2013,1,31)):
    X_train   = X.loc[train_index].drop('交易月份', axis=1)
    y_train = y.loc[train_index]
    
    X_val   = X.loc[val_index].drop('交易月份', axis=1)
    y_val = y.loc[val_index]

    X_test    = X.loc[test_index].drop('交易月份', axis=1)
    y_test  = y.loc[test_index]

    for i in range(len(alphas)):
        model_val = ElasticNet(alpha=alphas[i], l1_ratio=0.5)
        model_val.fit(X_train,y_train)
        Yval_predict = model_val.predict(X_val)
        mse[i,0] = np.sqrt(mean_squared_error(y_val,Yval_predict))

    optim_alpha = alphas[np.argmin(mse)]
   
    model = ElasticNet(alpha=optim_alpha, l1_ratio=0.5)
    model.fit(np.concatenate((X_train, X_val)), np.concatenate((y_train, y_val))) 
    preds = model.predict(X_test)

    predictions_small_shareholder.append(preds)
    dates_small_shareholder.append(y_test.index)
    y_test_list_small_shareholder.append(y_test)
    

    r2_small_shareholder = r2_score(y_test, preds)
    dic_r2_all_small_shareholder["r2." + str(y_test.index)] = r2
    
predictions_all_small_shareholder= np.concatenate(predictions_small_shareholder, axis=0)
y_test_list_all_small_shareholder= np.concatenate(y_test_list_small_shareholder, axis=0) 
dates_all_small_shareholder= np.concatenate(dates_small_shareholder, axis=0)

R2OOS_ENet_small_shareholder = r2_score(y_test_list_all_small_shareholder, predictions_all_small_shareholder)
print("R2OOS Elastic net Small Shareholder: ", R2OOS_ENet_small_shareholder)


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

alphas = np.linspace(start=0.00001,stop=0.004,num=20)
mse = np.full((len(alphas),1),np.nan, dtype = np.float32)


for train_index, val_index, test_index in tscv.split(X, first_split_date= datetime.date(2005,1,31), second_split_date= datetime.date(2013,1,31)):
    X_train   = X.loc[train_index].drop('交易月份', axis=1)
    y_train = y.loc[train_index]
    
    X_val   = X.loc[val_index].drop('交易月份', axis=1)
    y_val = y.loc[val_index]

    X_test    = X.loc[test_index].drop('交易月份', axis=1)
    y_test  = y.loc[test_index]

    for i in range(len(alphas)):
        model_val = ElasticNet(alpha=alphas[i], l1_ratio=0.5)
        model_val.fit(X_train,y_train)
        Yval_predict = model_val.predict(X_val)
        mse[i,0] = np.sqrt(mean_squared_error(y_val,Yval_predict))

    optim_alpha = alphas[np.argmin(mse)]
   
    model = ElasticNet(alpha=optim_alpha, l1_ratio=0.5)
    model.fit(np.concatenate((X_train, X_val)), np.concatenate((y_train, y_val))) 
    preds = model.predict(X_test)

    predictions_soe_stocks.append(preds)
    dates_soe_stocks.append(y_test.index)
    y_test_list_soe_stocks.append(y_test)
    

    r2_soe_stocks = r2_score(y_test, preds)
    dic_r2_all_soe_stocks["r2." + str(y_test.index)] = r2
    
predictions_all_soe_stocks= np.concatenate(predictions_soe_stocks, axis=0)
y_test_list_all_soe_stocks= np.concatenate(y_test_list_soe_stocks, axis=0) 
dates_all_soe_stocks= np.concatenate(dates_soe_stocks, axis=0)

R2OOS_ENet_soe_stocks = r2_score(y_test_list_all_soe_stocks, predictions_all_soe_stocks)
print("R2OOS Elastic net Soe Stocks: ", R2OOS_ENet_soe_stocks)


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

alphas = np.linspace(start=0.00001,stop=0.004,num=20)
mse = np.full((len(alphas),1),np.nan, dtype = np.float32)


for train_index, val_index, test_index in tscv.split(X, first_split_date= datetime.date(2005,1,31), second_split_date= datetime.date(2013,1,31)):
    X_train   = X.loc[train_index].drop('交易月份', axis=1)
    y_train = y.loc[train_index]
    
    X_val   = X.loc[val_index].drop('交易月份', axis=1)
    y_val = y.loc[val_index]

    X_test    = X.loc[test_index].drop('交易月份', axis=1)
    y_test  = y.loc[test_index]

    for i in range(len(alphas)):
        model_val = ElasticNet(alpha=alphas[i], l1_ratio=0.5)
        model_val.fit(X_train,y_train)
        Yval_predict = model_val.predict(X_val)
        mse[i,0] = np.sqrt(mean_squared_error(y_val,Yval_predict))

    optim_alpha = alphas[np.argmin(mse)]
   
    model = ElasticNet(alpha=optim_alpha, l1_ratio=0.5)
    model.fit(np.concatenate((X_train, X_val)), np.concatenate((y_train, y_val))) 
    preds = model.predict(X_test)

    predictions_non_soe_stocks.append(preds)
    dates_non_soe_stocks.append(y_test.index)
    y_test_list_non_soe_stocks.append(y_test)
    

    r2_non_soe_stocks = r2_score(y_test, preds)
    dic_r2_all_non_soe_stocks["r2." + str(y_test.index)] = r2
    
predictions_all_non_soe_stocks= np.concatenate(predictions_non_soe_stocks, axis=0)
y_test_list_all_non_soe_stocks= np.concatenate(y_test_list_non_soe_stocks, axis=0) 
dates_all_non_soe_stocks= np.concatenate(dates_non_soe_stocks, axis=0)

R2OOS_ENet_non_soe_stocks = r2_score(y_test_list_all_non_soe_stocks, predictions_all_non_soe_stocks)
print("R2OOS Elastic net non-Soe Stocks: ", R2OOS_ENet_non_soe_stocks)


# In[14]:


#不同数据集下的R2oos
chart = np.array([[R2OOS_ENet],
                  [R2OOS_ENet_large_stocks],
                  [R2OOS_ENet_small_stocks],
                  [R2OOS_ENet_large_shareholder],
                  [R2OOS_ENet_small_shareholder],
                  [R2OOS_ENet_soe_stocks],
                  [R2OOS_ENet_non_soe_stocks]])
                     
r2oos_enet = pd.DataFrame(chart, columns=['ENet(+H)'],
                              index=["All", "Top 70%", "Bottom 30%", 
                                     "A.M.C.P.S. Top 70%", "A.M.C.P.S. Bottom 30%", "SOE", "Non-SOE"])
                    
r2oos_enet


# In[15]:


r2oos_enet.to_csv(r'r2oos results/r2oos_enet.csv')

