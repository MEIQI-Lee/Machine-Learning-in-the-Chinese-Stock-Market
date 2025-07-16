#!/usr/bin/env python
# coding: utf-8

# ### NN1 - NN5

# In[1]:


import numpy as np
import pandas as pd
import math
import sklearn
import sklearn.preprocessing
import datetime
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error

import warnings
warnings.simplefilter(action='ignore', category=Warning)
import seaborn as sns
sns.set()
pd.options.mode.chained_assignment = None  

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


# 神经网络调参(以NN1为例)

# In[ ]:


import numpy as np
import datetime
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

tscv = TimeBasedCV(train_period=60, val_period=24, test_period=12, freq='months')

features = df.columns[~df.columns.isin(['证券代码', "mom1m"])].tolist()
X = df[features]
y = df["mom1m"]

best_params = {}
best_scores = []
test_scores = []

for fold, (train_index, val_index, test_index) in enumerate(tscv.split(X, first_split_date=datetime.date(2005, 1, 31), second_split_date=datetime.date(2013, 1, 31))):
    
    X_train = X.loc[train_index].drop('交易月份', axis=1)
    y_train = y.loc[train_index]
    
    X_val = X.loc[val_index].drop('交易月份', axis=1)
    y_val = y.loc[val_index]

    X_test = X.loc[test_index].drop('交易月份', axis=1)
    y_test = y.loc[test_index]

    estimator = MLPRegressor(hidden_layer_sizes=(16,), activation='relu', solver='adam', alpha=0.1, max_iter=2000, random_state=12)
    
    param_grid = {
        'learning_rate_init': [1e-3, 1e-2] 
    }
    
    grid = GridSearchCV(estimator, param_grid, scoring='neg_mean_squared_error', cv=5)
    
    grid.fit(X_train, y_train)
    
    best_params[fold] = grid.best_params_  
    best_scores.append(-grid.best_score_)  # 负均方误差转为正值
    
    y_pred = grid.predict(X_val)
    val_mse = mean_squared_error(y_val, y_pred)
    test_scores.append(val_mse)

    print(f"Best parameters for fold {fold}: {grid.best_params_}")
    print(f"Validation MSE for fold {fold}: {val_mse}")

print("Best parameters for all folds: ", best_params)
print("Best scores for all folds: ", best_scores)
print("Validation MSE for all folds: ", test_scores)


# 结论:learning_rate_init = 0.001

# # 所有股票
# 
# ## NN5

# In[5]:


def R_oos(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted).flatten()
    predicted = np.where(predicted < 0, 0, predicted)
    return 1 - (np.dot((actual - predicted), (actual - predicted))) / (np.dot(actual, actual))


# In[7]:


#如果模型在10个epoch内未能提高验证损失，早停机制就会被触发，从而提前结束训练。


# In[6]:


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks
from sklearn.metrics import mean_squared_error, r2_score
import datetime

# 设置随机种子
seed_value = 42
tf.random.set_seed(seed_value)
np.random.seed(seed_value)

tscv = TimeBasedCV(train_period=60, val_period=24, test_period=12, freq='months')
features = df.columns[~df.columns.isin(['证券代码', "mom1m"])].tolist()
X = df[features]
y = df["mom1m"]

best_val_mse = float('inf')
best_settings = {'l2_strength': None, 'learning_rate': None, 'model': None}

# 定义 L2 正则化强度
l2_strengths = [0.00001, 0.0001, 0.001]
learning_rates = [0.001]  

for fold, (train_index, val_index, test_index) in enumerate(tscv.split(X, first_split_date=datetime.date(2005, 1, 31), second_split_date=datetime.date(2013, 1, 31))):
    
    X_train = X.loc[train_index].drop('交易月份', axis=1)
    y_train = y.loc[train_index]
    
    X_val = X.loc[val_index].drop('交易月份', axis=1)
    y_val = y.loc[val_index]

    X_test = X.loc[test_index].drop('交易月份', axis=1)
    y_test = y.loc[test_index]

    for l2_strength in l2_strengths:
        for lr in learning_rates:
            print(f"Fold: {fold}, L2 Strength: {l2_strength}, Learning Rate: {lr}")

            model = models.Sequential([
                layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),
                layers.BatchNormalization(),
                layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),
                layers.BatchNormalization(),
                layers.Dense(8, activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),
                layers.BatchNormalization(),
                layers.Dense(4, activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),
                layers.BatchNormalization(),
                layers.Dense(2, activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),
                layers.BatchNormalization(),
                layers.Dense(1)  
            ])

            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                          loss='mean_squared_error',
                          metrics=['mean_squared_error'])

            early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

            model.fit(X_train, y_train, validation_data=(X_val, y_val),
                      epochs=100, callbacks=[early_stopping], batch_size=5000, verbose=1)

            val_mse = model.evaluate(X_val, y_val, verbose=0)[1]

            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_settings['l2_strength'] = l2_strength
                best_settings['learning_rate'] = lr
                best_settings['model'] = model

print(f"Best Validation MSE: {best_val_mse}")
print(f"Best Settings: L2 Strength = {best_settings['l2_strength']}, Learning Rate = {best_settings['learning_rate']}")

y_train_pred = best_settings['model'].predict(X_train, verbose=0)
y_val_pred = best_settings['model'].predict(X_val, verbose=0)
y_test_pred = best_settings['model'].predict(X_test, verbose=0)


r2_train = r2_score(y_train, y_train_pred)
r2_val = r2_score(y_val, y_val_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)
r2_oos_test = R_oos(y_test, y_test_pred)  

print(f"Training R-squared: {r2_train}")
print(f"Validation R-squared: {r2_val}")
print(f"Test MSE: {mse_test}")
print(f"Test R-squared: {r2_test}")
print(f"R2OOS NN5:: {r2_oos_test}")


# ## NN1

# In[13]:


l2_strength = 0.0001
learning_rate = 0.001  

best_val_mse = float('inf')
best_settings = {'model': None}

for fold, (train_index, val_index, test_index) in enumerate(tscv.split(X, first_split_date=datetime.date(2005, 1, 31), second_split_date=datetime.date(2013, 1, 31))):
    
    X_train = X.loc[train_index].drop('交易月份', axis=1)
    y_train = y.loc[train_index]
    
    X_val = X.loc[val_index].drop('交易月份', axis=1)
    y_val = y.loc[val_index]

    X_test = X.loc[test_index].drop('交易月份', axis=1)
    y_test = y.loc[test_index]

    # 修改为只有一层的神经网络
    model = models.Sequential([
        layers.Dense(1, activation='linear', kernel_regularizer=regularizers.l2(l2_strength), input_shape=(X_train.shape[1],))
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mean_squared_error',
                  metrics=['mean_squared_error'])

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # 训练模型
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=100, callbacks=[early_stopping], batch_size=5000, verbose=1)

    # 评估模型
    val_mse = model.evaluate(X_val, y_val, verbose=0)[1]

    # 保存最佳模型
    if val_mse < best_val_mse:
        best_val_mse = val_mse
        best_settings['model'] = model

# 预测
y_train_pred = best_settings['model'].predict(X_train, verbose=0)
y_val_pred = best_settings['model'].predict(X_val, verbose=0)
y_test_pred = best_settings['model'].predict(X_test, verbose=0)

# 计算指标
r2_train = r2_score(y_train, y_train_pred)
r2_val = r2_score(y_val, y_val_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)
r2_oos_test = R_oos(y_test, y_test_pred)  

# 打印结果
print(f"Training R-squared: {r2_train}")
print(f"Validation R-squared: {r2_val}")
print(f"Test MSE: {mse_test}")
print(f"Test R-squared: {r2_test}")
print(f"R2OOS NN1:{r2_oos_test}")


# ## NN2

# In[15]:


l2_strength = 0.0001
learning_rate = 0.001  

best_val_mse = float('inf')
best_settings = {'model': None}

for fold, (train_index, val_index, test_index) in enumerate(tscv.split(X, first_split_date=datetime.date(2005, 1, 31), second_split_date=datetime.date(2013, 1, 31))):
    
    X_train = X.loc[train_index].drop('交易月份', axis=1)
    y_train = y.loc[train_index]
    
    X_val = X.loc[val_index].drop('交易月份', axis=1)
    y_val = y.loc[val_index]

    X_test = X.loc[test_index].drop('交易月份', axis=1)
    y_test = y.loc[test_index]

    # 修改为两层的神经网络
    model = models.Sequential([
        layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(l2_strength), input_shape=(X_train.shape[1],)),  # 第一层
        layers.Dense(1, activation='linear', kernel_regularizer=regularizers.l2(l2_strength))  # 输出层
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mean_squared_error',
                  metrics=['mean_squared_error'])

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=100, callbacks=[early_stopping], batch_size=5000, verbose=1)

    
    val_mse = model.evaluate(X_val, y_val, verbose=0)[1]

    
    if val_mse < best_val_mse:
        best_val_mse = val_mse
        best_settings['model'] = model


y_train_pred = best_settings['model'].predict(X_train, verbose=0)
y_val_pred = best_settings['model'].predict(X_val, verbose=0)
y_test_pred = best_settings['model'].predict(X_test, verbose=0)


r2_train = r2_score(y_train, y_train_pred)
r2_val = r2_score(y_val, y_val_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)
r2_oos_test = R_oos(y_test, y_test_pred)  


print(f"Training R-squared: {r2_train}")
print(f"Validation R-squared: {r2_val}")
print(f"Test MSE: {mse_test}")
print(f"Test R-squared: {r2_test}")
print(f"R2OOS NN2: {r2_oos_test}")


# ## NN3

# In[16]:


l2_strength = 0.0001
learning_rate = 0.001  

best_val_mse = float('inf')
best_settings = {'model': None}

for fold, (train_index, val_index, test_index) in enumerate(tscv.split(X, first_split_date=datetime.date(2005, 1, 31), second_split_date=datetime.date(2013, 1, 31))):
    
    X_train = X.loc[train_index].drop('交易月份', axis=1)
    y_train = y.loc[train_index]
    
    X_val = X.loc[val_index].drop('交易月份', axis=1)
    y_val = y.loc[val_index]

    X_test = X.loc[test_index].drop('交易月份', axis=1)
    y_test = y.loc[test_index]

    # 修改为三层的神经网络
    model = models.Sequential([
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(l2_strength), input_shape=(X_train.shape[1],)),  # 第一层
        layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),  # 第二层
        layers.Dense(1, activation='linear', kernel_regularizer=regularizers.l2(l2_strength))  # 输出层
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mean_squared_error',
                  metrics=['mean_squared_error'])

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # 训练模型
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=100, callbacks=[early_stopping], batch_size=5000, verbose=1)

    # 评估模型
    val_mse = model.evaluate(X_val, y_val, verbose=0)[1]

    # 保存最佳模型
    if val_mse < best_val_mse:
        best_val_mse = val_mse
        best_settings['model'] = model

# 预测
y_train_pred = best_settings['model'].predict(X_train, verbose=0)
y_val_pred = best_settings['model'].predict(X_val, verbose=0)
y_test_pred = best_settings['model'].predict(X_test, verbose=0)

# 计算指标
r2_train = r2_score(y_train, y_train_pred)
r2_val = r2_score(y_val, y_val_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)
r2_oos_test = R_oos(y_test, y_test_pred)  


print(f"Training R-squared: {r2_train}")
print(f"Validation R-squared: {r2_val}")
print(f"Test MSE: {mse_test}")
print(f"Test R-squared: {r2_test}")
print(f"R2OOS NN3: {r2_oos_test}")


# ## NN4

# In[17]:


l2_strength = 0.0001
learning_rate = 0.001  

best_val_mse = float('inf')
best_settings = {'model': None}

for fold, (train_index, val_index, test_index) in enumerate(tscv.split(X, first_split_date=datetime.date(2005, 1, 31), second_split_date=datetime.date(2013, 1, 31))):
    
    X_train = X.loc[train_index].drop('交易月份', axis=1)
    y_train = y.loc[train_index]
    
    X_val = X.loc[val_index].drop('交易月份', axis=1)
    y_val = y.loc[val_index]

    X_test = X.loc[test_index].drop('交易月份', axis=1)
    y_test = y.loc[test_index]

    # 修改为四层的神经网络
    model = models.Sequential([
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2_strength), input_shape=(X_train.shape[1],)),  # 第一层
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),  # 第二层
        layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),  # 第三层
        layers.Dense(1, activation='linear', kernel_regularizer=regularizers.l2(l2_strength))  # 输出层
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mean_squared_error',
                  metrics=['mean_squared_error'])

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # 训练模型
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=100, callbacks=[early_stopping], batch_size=5000, verbose=1)

    # 评估模型
    val_mse = model.evaluate(X_val, y_val, verbose=0)[1]

    # 保存最佳模型
    if val_mse < best_val_mse:
        best_val_mse = val_mse
        best_settings['model'] = model

# 预测
y_train_pred = best_settings['model'].predict(X_train, verbose=0)
y_val_pred = best_settings['model'].predict(X_val, verbose=0)
y_test_pred = best_settings['model'].predict(X_test, verbose=0)

# 计算指标
r2_train = r2_score(y_train, y_train_pred)
r2_val = r2_score(y_val, y_val_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)
r2_oos_test = R_oos(y_test, y_test_pred)  

print(f"Training R-squared: {r2_train}")
print(f"Validation R-squared: {r2_val}")
print(f"Test MSE: {mse_test}")
print(f"Test R-squared: {r2_test}")
print(f"R2OOS NN4: {r2_oos_test}")


# In[45]:


R2OOS_NN1 = 0.002991318702697754
R2OOS_NN2 = -0.0271146297454834
R2OOS_NN3 = -0.16752839088439941
R2OOS_NN4 = 0.003899514675140381
R2OOS_NN5 = -0.04843342304229736


# In[46]:


chart_all = np.array([[R2OOS_NN1],
                      [R2OOS_NN2],
                      [R2OOS_NN3],
                      [R2OOS_NN4],
                      [R2OOS_NN5]])

r2oos_NN_all = pd.DataFrame(chart_all, columns=['All'], index=['NN1', 'NN2', 'NN3', 'NN4', 'NN5'])

r2oos_NN_all


# # 大股票

# In[24]:


features = df_large_stocks.columns[~df_large_stocks.columns.isin(['证券代码',"mom1m"])].tolist()
X = df_large_stocks[features]
y = df_large_stocks["mom1m"]


# ## NN1

# In[26]:


l2_strength = 0.0001
learning_rate = 0.001  

best_val_mse = float('inf')
best_settings = {'model': None}

for fold, (train_index, val_index, test_index) in enumerate(tscv.split(X, first_split_date=datetime.date(2005, 1, 31), second_split_date=datetime.date(2013, 1, 31))):
    
    X_train = X.loc[train_index].drop('交易月份', axis=1)
    y_train = y.loc[train_index]
    
    X_val = X.loc[val_index].drop('交易月份', axis=1)
    y_val = y.loc[val_index]

    X_test = X.loc[test_index].drop('交易月份', axis=1)
    y_test = y.loc[test_index]

    # 修改为只有一层的神经网络
    model = models.Sequential([
                layers.Dense(1, activation='linear', kernel_regularizer=regularizers.l2(l2_strength), input_shape=(X_train.shape[1],))
            ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mean_squared_error',
                  metrics=['mean_squared_error'])

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # 训练模型
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=100, callbacks=[early_stopping], batch_size=5000, verbose=1)

    # 评估模型
    val_mse = model.evaluate(X_val, y_val, verbose=0)[1]

    # 保存最佳模型
    if val_mse < best_val_mse:
        best_val_mse = val_mse
        best_settings['model'] = model

# 预测
y_train_pred = best_settings['model'].predict(X_train, verbose=0)
y_val_pred = best_settings['model'].predict(X_val, verbose=0)
y_test_pred = best_settings['model'].predict(X_test, verbose=0)

r2_train = r2_score(y_train, y_train_pred)
r2_val = r2_score(y_val, y_val_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)
r2_oos_large_stocks_NN1 = R_oos(y_test, y_test_pred)  

print(f"Training R-squared: {r2_train}")
print(f"Validation R-squared: {r2_val}")
print(f"Test MSE: {mse_test}")
print(f"Test R-squared: {r2_test}")
print(f"R2OOS Large Stocks NN1:{r2_oos_large_stocks_NN1}")


# ## NN2

# In[27]:


l2_strength = 0.0001
learning_rate = 0.001  

best_val_mse = float('inf')
best_settings = {'model': None}

for fold, (train_index, val_index, test_index) in enumerate(tscv.split(X, first_split_date=datetime.date(2005, 1, 31), second_split_date=datetime.date(2013, 1, 31))):
    
    X_train = X.loc[train_index].drop('交易月份', axis=1)
    y_train = y.loc[train_index]
    
    X_val = X.loc[val_index].drop('交易月份', axis=1)
    y_val = y.loc[val_index]

    X_test = X.loc[test_index].drop('交易月份', axis=1)
    y_test = y.loc[test_index]

    # 修改为两层的神经网络
    model = models.Sequential([
        layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(l2_strength), input_shape=(X_train.shape[1],)),  # 第一层
        layers.Dense(1, activation='linear', kernel_regularizer=regularizers.l2(l2_strength))  # 输出层
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mean_squared_error',
                  metrics=['mean_squared_error'])

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # 训练模型
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=100, callbacks=[early_stopping], batch_size=5000, verbose=1)

    # 评估模型
    val_mse = model.evaluate(X_val, y_val, verbose=0)[1]

    # 保存最佳模型
    if val_mse < best_val_mse:
        best_val_mse = val_mse
        best_settings['model'] = model

# 预测
y_train_pred = best_settings['model'].predict(X_train, verbose=0)
y_val_pred = best_settings['model'].predict(X_val, verbose=0)
y_test_pred = best_settings['model'].predict(X_test, verbose=0)

r2_train = r2_score(y_train, y_train_pred)
r2_val = r2_score(y_val, y_val_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)
r2_oos_large_stocks_NN2 = R_oos(y_test, y_test_pred)  

print(f"Training R-squared: {r2_train}")
print(f"Validation R-squared: {r2_val}")
print(f"Test MSE: {mse_test}")
print(f"Test R-squared: {r2_test}")
print(f"R2OOS Large Stocks NN2:{r2_oos_large_stocks_NN2}")


# ## NN3

# In[28]:


l2_strength = 0.0001
learning_rate = 0.001  

best_val_mse = float('inf')
best_settings = {'model': None}

for fold, (train_index, val_index, test_index) in enumerate(tscv.split(X, first_split_date=datetime.date(2005, 1, 31), second_split_date=datetime.date(2013, 1, 31))):
    
    X_train = X.loc[train_index].drop('交易月份', axis=1)
    y_train = y.loc[train_index]
    
    X_val = X.loc[val_index].drop('交易月份', axis=1)
    y_val = y.loc[val_index]

    X_test = X.loc[test_index].drop('交易月份', axis=1)
    y_test = y.loc[test_index]

    # 修改为三层的神经网络
    model = models.Sequential([
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(l2_strength), input_shape=(X_train.shape[1],)),  # 第一层
        layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),  # 第二层
        layers.Dense(1, activation='linear', kernel_regularizer=regularizers.l2(l2_strength))  # 输出层
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mean_squared_error',
                  metrics=['mean_squared_error'])

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # 训练模型
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=100, callbacks=[early_stopping], batch_size=5000, verbose=1)

    # 评估模型
    val_mse = model.evaluate(X_val, y_val, verbose=0)[1]

    # 保存最佳模型
    if val_mse < best_val_mse:
        best_val_mse = val_mse
        best_settings['model'] = model

# 预测
y_train_pred = best_settings['model'].predict(X_train, verbose=0)
y_val_pred = best_settings['model'].predict(X_val, verbose=0)
y_test_pred = best_settings['model'].predict(X_test, verbose=0)

r2_train = r2_score(y_train, y_train_pred)
r2_val = r2_score(y_val, y_val_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)
r2_oos_large_stocks_NN3 = R_oos(y_test, y_test_pred)  

print(f"Training R-squared: {r2_train}")
print(f"Validation R-squared: {r2_val}")
print(f"Test MSE: {mse_test}")
print(f"Test R-squared: {r2_test}")
print(f"R2OOS Large Stocks NN3:{r2_oos_large_stocks_NN3}")


# ## NN4

# In[29]:


l2_strength = 0.0001
learning_rate = 0.001  

best_val_mse = float('inf')
best_settings = {'model': None}

for fold, (train_index, val_index, test_index) in enumerate(tscv.split(X, first_split_date=datetime.date(2005, 1, 31), second_split_date=datetime.date(2013, 1, 31))):
    
    X_train = X.loc[train_index].drop('交易月份', axis=1)
    y_train = y.loc[train_index]
    
    X_val = X.loc[val_index].drop('交易月份', axis=1)
    y_val = y.loc[val_index]

    X_test = X.loc[test_index].drop('交易月份', axis=1)
    y_test = y.loc[test_index]

    # 修改为四层的神经网络
    model = models.Sequential([
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2_strength), input_shape=(X_train.shape[1],)),  # 第一层
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),  # 第二层
        layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),  # 第三层
        layers.Dense(1, activation='linear', kernel_regularizer=regularizers.l2(l2_strength))  # 输出层
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mean_squared_error',
                  metrics=['mean_squared_error'])

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # 训练模型
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=100, callbacks=[early_stopping], batch_size=5000, verbose=1)

    # 评估模型
    val_mse = model.evaluate(X_val, y_val, verbose=0)[1]

    # 保存最佳模型
    if val_mse < best_val_mse:
        best_val_mse = val_mse
        best_settings['model'] = model

# 预测
y_train_pred = best_settings['model'].predict(X_train, verbose=0)
y_val_pred = best_settings['model'].predict(X_val, verbose=0)
y_test_pred = best_settings['model'].predict(X_test, verbose=0)

r2_train = r2_score(y_train, y_train_pred)
r2_val = r2_score(y_val, y_val_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)
r2_oos_large_stocks_NN4 = R_oos(y_test, y_test_pred)  

print(f"Training R-squared: {r2_train}")
print(f"Validation R-squared: {r2_val}")
print(f"Test MSE: {mse_test}")
print(f"Test R-squared: {r2_test}")
print(f"R2OOS Large Stocks NN4:{r2_oos_large_stocks_NN4}")


# ## NN5

# In[30]:


l2_strength = 0.0001
learning_rate = 0.001  

best_val_mse = float('inf')
best_settings = {'model': None}

for fold, (train_index, val_index, test_index) in enumerate(tscv.split(X, first_split_date=datetime.date(2005, 1, 31), second_split_date=datetime.date(2013, 1, 31))):
    
    X_train = X.loc[train_index].drop('交易月份', axis=1)
    y_train = y.loc[train_index]
    
    X_val = X.loc[val_index].drop('交易月份', axis=1)
    y_val = y.loc[val_index]

    X_test = X.loc[test_index].drop('交易月份', axis=1)
    y_test = y.loc[test_index]

    # 修改为五层的神经网络
    model = models.Sequential([
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2_strength), input_shape=(X_train.shape[1],)),  # 第一层
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),  # 第二层
        layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),  # 第三层
        layers.Dense(8, activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),  # 第四层
        layers.Dense(1, activation='linear', kernel_regularizer=regularizers.l2(l2_strength))  # 输出层
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mean_squared_error',
                  metrics=['mean_squared_error'])

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # 训练模型
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=100, callbacks=[early_stopping], batch_size=5000, verbose=1)

    # 评估模型
    val_mse = model.evaluate(X_val, y_val, verbose=0)[1]

    # 保存最佳模型
    if val_mse < best_val_mse:
        best_val_mse = val_mse
        best_settings['model'] = model

# 预测
y_train_pred = best_settings['model'].predict(X_train, verbose=0)
y_val_pred = best_settings['model'].predict(X_val, verbose=0)
y_test_pred = best_settings['model'].predict(X_test, verbose=0)

r2_train = r2_score(y_train, y_train_pred)
r2_val = r2_score(y_val, y_val_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)
r2_oos_large_stocks_NN5 = R_oos(y_test, y_test_pred)  

print(f"Training R-squared: {r2_train}")
print(f"Validation R-squared: {r2_val}")
print(f"Test MSE: {mse_test}")
print(f"Test R-squared: {r2_test}")
print(f"R2OOS Large Stocks NN5:{r2_oos_large_stocks_NN5}")


# In[ ]:


r2_oos_large_stocks_NN1 = 0.006511807441711426
r2_oos_large_stocks_NN2 = -0.1205834150314331
r2_oos_large_stocks_NN3 = -0.21247875690460205
r2_oos_large_stocks_NN4 = -0.06101834774017334
r2_oos_large_stocks_NN5 = -0.10608279705047607

chart_large_stocks = np.array([[r2_oos_large_stocks_NN1],
                      [r2_oos_large_stocks_NN2],
                      [r2_oos_large_stocks_NN3],
                      [r2_oos_large_stocks_NN4],
                      [r2_oos_large_stocks_NN5]])

r2oos_NN_large_stocks = pd.DataFrame(chart_large_stocks, columns=['Top 70%'], index=['NN1', 'NN2', 'NN3', 'NN4', 'NN5'])

r2oos_NN_large_stocks


# In[53]:


merged_df = pd.concat([r2oos_NN_all, r2oos_NN_large_stocks], axis=1)


# In[54]:


merged_df = merged_df.T


# In[55]:


merged_df


# # 小股票

# In[5]:


features = df_small_stocks.columns[~df_small_stocks.columns.isin(['证券代码',"mom1m"])].tolist()
X = df_small_stocks[features]
y = df_small_stocks["mom1m"]


# ## NN1

# In[11]:


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks
from sklearn.metrics import mean_squared_error, r2_score
import datetime

tscv = TimeBasedCV(train_period=60, val_period=24, test_period=12, freq='months')
l2_strength = 0.0001
learning_rate = 0.001  

best_val_mse = float('inf')
best_settings = {'model': None}

for fold, (train_index, val_index, test_index) in enumerate(tscv.split(X, first_split_date=datetime.date(2005, 1, 31), second_split_date=datetime.date(2013, 1, 31))):
    
    X_train = X.loc[train_index].drop('交易月份', axis=1)
    y_train = y.loc[train_index]
    
    X_val = X.loc[val_index].drop('交易月份', axis=1)
    y_val = y.loc[val_index]

    X_test = X.loc[test_index].drop('交易月份', axis=1)
    y_test = y.loc[test_index]

    # 修改为只有一层的神经网络
    model = models.Sequential([
                layers.Dense(1, activation='linear', kernel_regularizer=regularizers.l2(l2_strength), input_shape=(X_train.shape[1],))
            ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mean_squared_error',
                  metrics=['mean_squared_error'])

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # 训练模型
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=100, callbacks=[early_stopping], batch_size=5000, verbose=1)

    # 评估模型
    val_mse = model.evaluate(X_val, y_val, verbose=0)[1]

    # 保存最佳模型
    if val_mse < best_val_mse:
        best_val_mse = val_mse
        best_settings['model'] = model

# 预测
y_train_pred = best_settings['model'].predict(X_train, verbose=0)
y_val_pred = best_settings['model'].predict(X_val, verbose=0)
y_test_pred = best_settings['model'].predict(X_test, verbose=0)

r2_train = r2_score(y_train, y_train_pred)
r2_val = r2_score(y_val, y_val_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)
r2_oos_small_stocks_NN1 = R_oos(y_test, y_test_pred)  

print(f"Training R-squared: {r2_train}")
print(f"Validation R-squared: {r2_val}")
print(f"Test MSE: {mse_test}")
print(f"Test R-squared: {r2_test}")
print(f"R2OOS Small Stocks NN1:{r2_oos_small_stocks_NN1}")


# ## NN2

# In[12]:


l2_strength = 0.0001
learning_rate = 0.001  

best_val_mse = float('inf')
best_settings = {'model': None}

for fold, (train_index, val_index, test_index) in enumerate(tscv.split(X, first_split_date=datetime.date(2005, 1, 31), second_split_date=datetime.date(2013, 1, 31))):
    
    X_train = X.loc[train_index].drop('交易月份', axis=1)
    y_train = y.loc[train_index]
    
    X_val = X.loc[val_index].drop('交易月份', axis=1)
    y_val = y.loc[val_index]

    X_test = X.loc[test_index].drop('交易月份', axis=1)
    y_test = y.loc[test_index]

    # 修改为两层的神经网络
    model = models.Sequential([
        layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(l2_strength), input_shape=(X_train.shape[1],)),  # 第一层
        layers.Dense(1, activation='linear', kernel_regularizer=regularizers.l2(l2_strength))  # 输出层
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mean_squared_error',
                  metrics=['mean_squared_error'])

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # 训练模型
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=100, callbacks=[early_stopping], batch_size=5000, verbose=1)

    # 评估模型
    val_mse = model.evaluate(X_val, y_val, verbose=0)[1]

    # 保存最佳模型
    if val_mse < best_val_mse:
        best_val_mse = val_mse
        best_settings['model'] = model

# 预测
y_train_pred = best_settings['model'].predict(X_train, verbose=0)
y_val_pred = best_settings['model'].predict(X_val, verbose=0)
y_test_pred = best_settings['model'].predict(X_test, verbose=0)

r2_train = r2_score(y_train, y_train_pred)
r2_val = r2_score(y_val, y_val_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)
r2_oos_small_stocks_NN2 = R_oos(y_test, y_test_pred)  

print(f"Training R-squared: {r2_train}")
print(f"Validation R-squared: {r2_val}")
print(f"Test MSE: {mse_test}")
print(f"Test R-squared: {r2_test}")
print(f"R2OOS Small Stocks NN2:{r2_oos_small_stocks_NN2}")


# ## NN3

# In[13]:


l2_strength = 0.0001
learning_rate = 0.001  

best_val_mse = float('inf')
best_settings = {'model': None}

for fold, (train_index, val_index, test_index) in enumerate(tscv.split(X, first_split_date=datetime.date(2005, 1, 31), second_split_date=datetime.date(2013, 1, 31))):
    
    X_train = X.loc[train_index].drop('交易月份', axis=1)
    y_train = y.loc[train_index]
    
    X_val = X.loc[val_index].drop('交易月份', axis=1)
    y_val = y.loc[val_index]

    X_test = X.loc[test_index].drop('交易月份', axis=1)
    y_test = y.loc[test_index]

    # 修改为三层的神经网络
    model = models.Sequential([
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(l2_strength), input_shape=(X_train.shape[1],)),  # 第一层
        layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),  # 第二层
        layers.Dense(1, activation='linear', kernel_regularizer=regularizers.l2(l2_strength))  # 输出层
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mean_squared_error',
                  metrics=['mean_squared_error'])

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # 训练模型
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=100, callbacks=[early_stopping], batch_size=5000, verbose=1)

    # 评估模型
    val_mse = model.evaluate(X_val, y_val, verbose=0)[1]

    # 保存最佳模型
    if val_mse < best_val_mse:
        best_val_mse = val_mse
        best_settings['model'] = model

# 预测
y_train_pred = best_settings['model'].predict(X_train, verbose=0)
y_val_pred = best_settings['model'].predict(X_val, verbose=0)
y_test_pred = best_settings['model'].predict(X_test, verbose=0)

r2_train = r2_score(y_train, y_train_pred)
r2_val = r2_score(y_val, y_val_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)
r2_oos_small_stocks_NN3 = R_oos(y_test, y_test_pred)  

print(f"Training R-squared: {r2_train}")
print(f"Validation R-squared: {r2_val}")
print(f"Test MSE: {mse_test}")
print(f"Test R-squared: {r2_test}")
print(f"R2OOS Small Stocks NN3:{r2_oos_small_stocks_NN3}")


# ## NN4

# In[14]:


l2_strength = 0.0001
learning_rate = 0.001  

best_val_mse = float('inf')
best_settings = {'model': None}

for fold, (train_index, val_index, test_index) in enumerate(tscv.split(X, first_split_date=datetime.date(2005, 1, 31), second_split_date=datetime.date(2013, 1, 31))):
    
    X_train = X.loc[train_index].drop('交易月份', axis=1)
    y_train = y.loc[train_index]
    
    X_val = X.loc[val_index].drop('交易月份', axis=1)
    y_val = y.loc[val_index]

    X_test = X.loc[test_index].drop('交易月份', axis=1)
    y_test = y.loc[test_index]

    # 修改为四层的神经网络
    model = models.Sequential([
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2_strength), input_shape=(X_train.shape[1],)),  # 第一层
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),  # 第二层
        layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),  # 第三层
        layers.Dense(1, activation='linear', kernel_regularizer=regularizers.l2(l2_strength))  # 输出层
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mean_squared_error',
                  metrics=['mean_squared_error'])

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # 训练模型
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=100, callbacks=[early_stopping], batch_size=5000, verbose=1)

    # 评估模型
    val_mse = model.evaluate(X_val, y_val, verbose=0)[1]

    # 保存最佳模型
    if val_mse < best_val_mse:
        best_val_mse = val_mse
        best_settings['model'] = model

# 预测
y_train_pred = best_settings['model'].predict(X_train, verbose=0)
y_val_pred = best_settings['model'].predict(X_val, verbose=0)
y_test_pred = best_settings['model'].predict(X_test, verbose=0)

r2_train = r2_score(y_train, y_train_pred)
r2_val = r2_score(y_val, y_val_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)
r2_oos_small_stocks_NN4 = R_oos(y_test, y_test_pred)  

print(f"Training R-squared: {r2_train}")
print(f"Validation R-squared: {r2_val}")
print(f"Test MSE: {mse_test}")
print(f"Test R-squared: {r2_test}")
print(f"R2OOS Small Stocks NN4:{r2_oos_small_stocks_NN4}")


# ## NN5

# In[15]:


l2_strength = 0.0001
learning_rate = 0.001  

best_val_mse = float('inf')
best_settings = {'model': None}

for fold, (train_index, val_index, test_index) in enumerate(tscv.split(X, first_split_date=datetime.date(2005, 1, 31), second_split_date=datetime.date(2013, 1, 31))):
    
    X_train = X.loc[train_index].drop('交易月份', axis=1)
    y_train = y.loc[train_index]
    
    X_val = X.loc[val_index].drop('交易月份', axis=1)
    y_val = y.loc[val_index]

    X_test = X.loc[test_index].drop('交易月份', axis=1)
    y_test = y.loc[test_index]

    # 修改为五层的神经网络
    model = models.Sequential([
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2_strength), input_shape=(X_train.shape[1],)),  # 第一层
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),  # 第二层
        layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),  # 第三层
        layers.Dense(8, activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),  # 第四层
        layers.Dense(1, activation='linear', kernel_regularizer=regularizers.l2(l2_strength))  # 输出层
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mean_squared_error',
                  metrics=['mean_squared_error'])

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # 训练模型
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=100, callbacks=[early_stopping], batch_size=5000, verbose=1)

    # 评估模型
    val_mse = model.evaluate(X_val, y_val, verbose=0)[1]

    # 保存最佳模型
    if val_mse < best_val_mse:
        best_val_mse = val_mse
        best_settings['model'] = model

# 预测
y_train_pred = best_settings['model'].predict(X_train, verbose=0)
y_val_pred = best_settings['model'].predict(X_val, verbose=0)
y_test_pred = best_settings['model'].predict(X_test, verbose=0)

r2_train = r2_score(y_train, y_train_pred)
r2_val = r2_score(y_val, y_val_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)
r2_oos_small_stocks_NN5 = R_oos(y_test, y_test_pred)  

print(f"Training R-squared: {r2_train}")
print(f"Validation R-squared: {r2_val}")
print(f"Test MSE: {mse_test}")
print(f"Test R-squared: {r2_test}")
print(f"R2OOS Small Stocks NN5:{r2_oos_small_stocks_NN5}")


# In[56]:


r2_oos_small_stocks_NN1 = 0.007626235485076904
r2_oos_small_stocks_NN2 = 0.0017200112342834473
r2_oos_small_stocks_NN3 = -0.05031764507293701
r2_oos_small_stocks_NN4 = -0.061913371086120605
r2_oos_small_stocks_NN5 = -0.2830575704574585

chart_small_stocks = np.array([[r2_oos_small_stocks_NN1],
                      [r2_oos_small_stocks_NN2],
                      [r2_oos_small_stocks_NN3],
                      [r2_oos_small_stocks_NN4],
                      [r2_oos_small_stocks_NN5]])

r2oos_NN_small_stocks = pd.DataFrame(chart_small_stocks, columns=['Bottom 30%'], index=['NN1', 'NN2', 'NN3', 'NN4', 'NN5'])

r2oos_NN_small_stocks


# In[58]:


merged_df = pd.concat([r2oos_NN_all,r2oos_NN_large_stocks, r2oos_NN_small_stocks], axis=1)
merged_df = merged_df.T
merged_df


# # 大股东

# In[59]:


features = df_large_shareholder.columns[~df_large_shareholder.columns.isin(['证券代码',"mom1m"])].tolist()
X = df_large_shareholder[features]
y = df_large_shareholder["mom1m"]


# NN1

# In[17]:


l2_strength = 0.0001
learning_rate = 0.001  

best_val_mse = float('inf')
best_settings = {'model': None}

for fold, (train_index, val_index, test_index) in enumerate(tscv.split(X, first_split_date=datetime.date(2005, 1, 31), second_split_date=datetime.date(2013, 1, 31))):
    
    X_train = X.loc[train_index].drop('交易月份', axis=1)
    y_train = y.loc[train_index]
    
    X_val = X.loc[val_index].drop('交易月份', axis=1)
    y_val = y.loc[val_index]

    X_test = X.loc[test_index].drop('交易月份', axis=1)
    y_test = y.loc[test_index]

    # 修改为只有一层的神经网络
    model = models.Sequential([
                layers.Dense(1, activation='linear', kernel_regularizer=regularizers.l2(l2_strength), input_shape=(X_train.shape[1],))
            ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mean_squared_error',
                  metrics=['mean_squared_error'])

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # 训练模型
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=100, callbacks=[early_stopping], batch_size=5000, verbose=1)

    # 评估模型
    val_mse = model.evaluate(X_val, y_val, verbose=0)[1]

    # 保存最佳模型
    if val_mse < best_val_mse:
        best_val_mse = val_mse
        best_settings['model'] = model

# 预测
y_train_pred = best_settings['model'].predict(X_train, verbose=0)
y_val_pred = best_settings['model'].predict(X_val, verbose=0)
y_test_pred = best_settings['model'].predict(X_test, verbose=0)

r2_train = r2_score(y_train, y_train_pred)
r2_val = r2_score(y_val, y_val_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)
r2_oos_large_shareholder_NN1 = R_oos(y_test, y_test_pred)  

print(f"Training R-squared: {r2_train}")
print(f"Validation R-squared: {r2_val}")
print(f"Test MSE: {mse_test}")
print(f"Test R-squared: {r2_test}")
print(f"R2OOS Large Shareholder NN1:{r2_oos_large_shareholder_NN1}")


# In[60]:


l2_strength = 0.0001
learning_rate = 0.001  

best_val_mse = float('inf')
best_settings = {'model': None}

for fold, (train_index, val_index, test_index) in enumerate(tscv.split(X, first_split_date=datetime.date(2005, 1, 31), second_split_date=datetime.date(2013, 1, 31))):
    
    X_train = X.loc[train_index].drop('交易月份', axis=1)
    y_train = y.loc[train_index]
    
    X_val = X.loc[val_index].drop('交易月份', axis=1)
    y_val = y.loc[val_index]

    X_test = X.loc[test_index].drop('交易月份', axis=1)
    y_test = y.loc[test_index]

    # 修改为两层的神经网络
    model = models.Sequential([
        layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(l2_strength), input_shape=(X_train.shape[1],)),  # 第一层
        layers.Dense(1, activation='linear', kernel_regularizer=regularizers.l2(l2_strength))  # 输出层
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mean_squared_error',
                  metrics=['mean_squared_error'])

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # 训练模型
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=100, callbacks=[early_stopping], batch_size=5000, verbose=1)

    # 评估模型
    val_mse = model.evaluate(X_val, y_val, verbose=0)[1]

    # 保存最佳模型
    if val_mse < best_val_mse:
        best_val_mse = val_mse
        best_settings['model'] = model

# 预测
y_train_pred = best_settings['model'].predict(X_train, verbose=0)
y_val_pred = best_settings['model'].predict(X_val, verbose=0)
y_test_pred = best_settings['model'].predict(X_test, verbose=0)

r2_train = r2_score(y_train, y_train_pred)
r2_val = r2_score(y_val, y_val_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)
r2_oos_large_shareholder_NN2 = R_oos(y_test, y_test_pred)  

print(f"Training R-squared: {r2_train}")
print(f"Validation R-squared: {r2_val}")
print(f"Test MSE: {mse_test}")
print(f"Test R-squared: {r2_test}")
print(f"R2OOS Large Shareholder NN2:{r2_oos_large_shareholder_NN2}")


# In[19]:


l2_strength = 0.0001
learning_rate = 0.001  

best_val_mse = float('inf')
best_settings = {'model': None}

for fold, (train_index, val_index, test_index) in enumerate(tscv.split(X, first_split_date=datetime.date(2005, 1, 31), second_split_date=datetime.date(2013, 1, 31))):
    
    X_train = X.loc[train_index].drop('交易月份', axis=1)
    y_train = y.loc[train_index]
    
    X_val = X.loc[val_index].drop('交易月份', axis=1)
    y_val = y.loc[val_index]

    X_test = X.loc[test_index].drop('交易月份', axis=1)
    y_test = y.loc[test_index]

    # 修改为三层的神经网络
    model = models.Sequential([
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(l2_strength), input_shape=(X_train.shape[1],)),  # 第一层
        layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),  # 第二层
        layers.Dense(1, activation='linear', kernel_regularizer=regularizers.l2(l2_strength))  # 输出层
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mean_squared_error',
                  metrics=['mean_squared_error'])

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # 训练模型
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=100, callbacks=[early_stopping], batch_size=5000, verbose=1)

    # 评估模型
    val_mse = model.evaluate(X_val, y_val, verbose=0)[1]

    # 保存最佳模型
    if val_mse < best_val_mse:
        best_val_mse = val_mse
        best_settings['model'] = model

# 预测
y_train_pred = best_settings['model'].predict(X_train, verbose=0)
y_val_pred = best_settings['model'].predict(X_val, verbose=0)
y_test_pred = best_settings['model'].predict(X_test, verbose=0)

r2_train = r2_score(y_train, y_train_pred)
r2_val = r2_score(y_val, y_val_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)
r2_oos_large_shareholder_NN3 = R_oos(y_test, y_test_pred)  

print(f"Training R-squared: {r2_train}")
print(f"Validation R-squared: {r2_val}")
print(f"Test MSE: {mse_test}")
print(f"Test R-squared: {r2_test}")
print(f"R2OOS Large Shareholder NN3:{r2_oos_large_shareholder_NN3}")


# In[20]:


l2_strength = 0.0001
learning_rate = 0.001  

best_val_mse = float('inf')
best_settings = {'model': None}

for fold, (train_index, val_index, test_index) in enumerate(tscv.split(X, first_split_date=datetime.date(2005, 1, 31), second_split_date=datetime.date(2013, 1, 31))):
    
    X_train = X.loc[train_index].drop('交易月份', axis=1)
    y_train = y.loc[train_index]
    
    X_val = X.loc[val_index].drop('交易月份', axis=1)
    y_val = y.loc[val_index]

    X_test = X.loc[test_index].drop('交易月份', axis=1)
    y_test = y.loc[test_index]

    # 修改为四层的神经网络
    model = models.Sequential([
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2_strength), input_shape=(X_train.shape[1],)),  # 第一层
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),  # 第二层
        layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),  # 第三层
        layers.Dense(1, activation='linear', kernel_regularizer=regularizers.l2(l2_strength))  # 输出层
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mean_squared_error',
                  metrics=['mean_squared_error'])

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # 训练模型
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=100, callbacks=[early_stopping], batch_size=5000, verbose=1)

    # 评估模型
    val_mse = model.evaluate(X_val, y_val, verbose=0)[1]

    # 保存最佳模型
    if val_mse < best_val_mse:
        best_val_mse = val_mse
        best_settings['model'] = model

# 预测
y_train_pred = best_settings['model'].predict(X_train, verbose=0)
y_val_pred = best_settings['model'].predict(X_val, verbose=0)
y_test_pred = best_settings['model'].predict(X_test, verbose=0)

r2_train = r2_score(y_train, y_train_pred)
r2_val = r2_score(y_val, y_val_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)
r2_oos_large_shareholder_NN4 = R_oos(y_test, y_test_pred)  

print(f"Training R-squared: {r2_train}")
print(f"Validation R-squared: {r2_val}")
print(f"Test MSE: {mse_test}")
print(f"Test R-squared: {r2_test}")
print(f"R2OOS Large Shareholder NN4:{r2_oos_large_shareholder_NN4}")


# In[21]:


l2_strength = 0.0001
learning_rate = 0.001  

best_val_mse = float('inf')
best_settings = {'model': None}

for fold, (train_index, val_index, test_index) in enumerate(tscv.split(X, first_split_date=datetime.date(2005, 1, 31), second_split_date=datetime.date(2013, 1, 31))):
    
    X_train = X.loc[train_index].drop('交易月份', axis=1)
    y_train = y.loc[train_index]
    
    X_val = X.loc[val_index].drop('交易月份', axis=1)
    y_val = y.loc[val_index]

    X_test = X.loc[test_index].drop('交易月份', axis=1)
    y_test = y.loc[test_index]

    # 修改为五层的神经网络
    model = models.Sequential([
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2_strength), input_shape=(X_train.shape[1],)),  # 第一层
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),  # 第二层
        layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),  # 第三层
        layers.Dense(8, activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),  # 第四层
        layers.Dense(1, activation='linear', kernel_regularizer=regularizers.l2(l2_strength))  # 输出层
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mean_squared_error',
                  metrics=['mean_squared_error'])

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # 训练模型
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=100, callbacks=[early_stopping], batch_size=5000, verbose=1)

    # 评估模型
    val_mse = model.evaluate(X_val, y_val, verbose=0)[1]

    # 保存最佳模型
    if val_mse < best_val_mse:
        best_val_mse = val_mse
        best_settings['model'] = model

# 预测
y_train_pred = best_settings['model'].predict(X_train, verbose=0)
y_val_pred = best_settings['model'].predict(X_val, verbose=0)
y_test_pred = best_settings['model'].predict(X_test, verbose=0)

r2_train = r2_score(y_train, y_train_pred)
r2_val = r2_score(y_val, y_val_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)
r2_oos_large_shareholder_NN5 = R_oos(y_test, y_test_pred)  

print(f"Training R-squared: {r2_train}")
print(f"Validation R-squared: {r2_val}")
print(f"Test MSE: {mse_test}")
print(f"Test R-squared: {r2_test}")
print(f"R2OOS Large Shareholder NN5:{r2_oos_large_shareholder_NN5}")


# In[62]:


r2_oos_large_shareholder_NN1 = 0.003509819507598877
r2_oos_large_shareholder_NN2 = -0.035060763359069824
r2_oos_large_shareholder_NN3 = -0.13361477851867676
r2_oos_large_shareholder_NN4 = -0.06039154529571533
r2_oos_large_shareholder_NN5 = -0.08671283721923828

chart_large_shareholder = np.array([[r2_oos_large_shareholder_NN1],
                      [r2_oos_large_shareholder_NN2],
                      [r2_oos_large_shareholder_NN3],
                      [r2_oos_large_shareholder_NN4],
                      [r2_oos_large_shareholder_NN5]])

r2oos_NN_large_shareholder = pd.DataFrame(chart_large_shareholder, columns=['A.M.C.P.S. Top 70%'], index=['NN1', 'NN2', 'NN3', 'NN4', 'NN5'])

r2oos_NN_large_shareholder


# # 小股东

# In[7]:


features = df_small_shareholder.columns[~df_small_shareholder.columns.isin(['证券代码',"mom1m"])].tolist()
X = df_small_shareholder[features]
y = df_small_shareholder["mom1m"]


# In[13]:


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks
from sklearn.metrics import mean_squared_error, r2_score
import datetime

# 设置随机种子
seed_value = 42
tf.random.set_seed(seed_value)
np.random.seed(seed_value)

tscv = TimeBasedCV(train_period=60, val_period=24, test_period=12, freq='months')
l2_strength = 0.0001
learning_rate = 0.001  

best_val_mse = float('inf')
best_settings = {'model': None}

for fold, (train_index, val_index, test_index) in enumerate(tscv.split(X, first_split_date=datetime.date(2005, 1, 31), second_split_date=datetime.date(2013, 1, 31))):
    
    X_train = X.loc[train_index].drop('交易月份', axis=1)
    y_train = y.loc[train_index]
    
    X_val = X.loc[val_index].drop('交易月份', axis=1)
    y_val = y.loc[val_index]

    X_test = X.loc[test_index].drop('交易月份', axis=1)
    y_test = y.loc[test_index]

    # 修改为只有一层的神经网络
    model = models.Sequential([
                layers.Dense(1, activation='linear', kernel_regularizer=regularizers.l2(l2_strength), input_shape=(X_train.shape[1],))
            ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mean_squared_error',
                  metrics=['mean_squared_error'])

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # 训练模型
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=100, callbacks=[early_stopping], batch_size=5000, verbose=1)

    # 评估模型
    val_mse = model.evaluate(X_val, y_val, verbose=0)[1]

    # 保存最佳模型
    if val_mse < best_val_mse:
        best_val_mse = val_mse
        best_settings['model'] = model

# 预测
y_train_pred = best_settings['model'].predict(X_train, verbose=0)
y_val_pred = best_settings['model'].predict(X_val, verbose=0)
y_test_pred = best_settings['model'].predict(X_test, verbose=0)

r2_train = r2_score(y_train, y_train_pred)
r2_val = r2_score(y_val, y_val_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)
r2_oos_small_shareholder_NN1 = R_oos(y_test, y_test_pred)  

print(f"Training R-squared: {r2_train}")
print(f"Validation R-squared: {r2_val}")
print(f"Test MSE: {mse_test}")
print(f"Test R-squared: {r2_test}")
print(f"R2OOS Small Shareholder NN1:{r2_oos_small_shareholder_NN1}")


# In[8]:


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks
from sklearn.metrics import mean_squared_error, r2_score
import datetime

# 设置随机种子
seed_value = 42
tf.random.set_seed(seed_value)
np.random.seed(seed_value)

tscv = TimeBasedCV(train_period=60, val_period=24, test_period=12, freq='months')

l2_strength = 0.0001
learning_rate = 0.001  

best_val_mse = float('inf')
best_settings = {'model': None}

for fold, (train_index, val_index, test_index) in enumerate(tscv.split(X, first_split_date=datetime.date(2005, 1, 31), second_split_date=datetime.date(2013, 1, 31))):
    
    X_train = X.loc[train_index].drop('交易月份', axis=1)
    y_train = y.loc[train_index]
    
    X_val = X.loc[val_index].drop('交易月份', axis=1)
    y_val = y.loc[val_index]

    X_test = X.loc[test_index].drop('交易月份', axis=1)
    y_test = y.loc[test_index]

    # 修改为两层的神经网络
    model = models.Sequential([
        layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(l2_strength), input_shape=(X_train.shape[1],)),  # 第一层
        layers.Dense(1, activation='linear', kernel_regularizer=regularizers.l2(l2_strength))  # 输出层
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mean_squared_error',
                  metrics=['mean_squared_error'])

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # 训练模型
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=100, callbacks=[early_stopping], batch_size=5000, verbose=1)

    # 评估模型
    val_mse = model.evaluate(X_val, y_val, verbose=0)[1]

    # 保存最佳模型
    if val_mse < best_val_mse:
        best_val_mse = val_mse
        best_settings['model'] = model

# 预测
y_train_pred = best_settings['model'].predict(X_train, verbose=0)
y_val_pred = best_settings['model'].predict(X_val, verbose=0)
y_test_pred = best_settings['model'].predict(X_test, verbose=0)

r2_train = r2_score(y_train, y_train_pred)
r2_val = r2_score(y_val, y_val_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)
r2_oos_small_shareholder_NN2 = R_oos(y_test, y_test_pred)  

print(f"Training R-squared: {r2_train}")
print(f"Validation R-squared: {r2_val}")
print(f"Test MSE: {mse_test}")
print(f"Test R-squared: {r2_test}")
print(f"R2OOS Small Shareholder NN2:{r2_oos_small_shareholder_NN2}")


# In[9]:


l2_strength = 0.0001
learning_rate = 0.001  

best_val_mse = float('inf')
best_settings = {'model': None}

for fold, (train_index, val_index, test_index) in enumerate(tscv.split(X, first_split_date=datetime.date(2005, 1, 31), second_split_date=datetime.date(2013, 1, 31))):
    
    X_train = X.loc[train_index].drop('交易月份', axis=1)
    y_train = y.loc[train_index]
    
    X_val = X.loc[val_index].drop('交易月份', axis=1)
    y_val = y.loc[val_index]

    X_test = X.loc[test_index].drop('交易月份', axis=1)
    y_test = y.loc[test_index]

    # 修改为三层的神经网络
    model = models.Sequential([
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(l2_strength), input_shape=(X_train.shape[1],)),  # 第一层
        layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),  # 第二层
        layers.Dense(1, activation='linear', kernel_regularizer=regularizers.l2(l2_strength))  # 输出层
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mean_squared_error',
                  metrics=['mean_squared_error'])

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # 训练模型
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=100, callbacks=[early_stopping], batch_size=5000, verbose=1)

    # 评估模型
    val_mse = model.evaluate(X_val, y_val, verbose=0)[1]

    # 保存最佳模型
    if val_mse < best_val_mse:
        best_val_mse = val_mse
        best_settings['model'] = model

# 预测
y_train_pred = best_settings['model'].predict(X_train, verbose=0)
y_val_pred = best_settings['model'].predict(X_val, verbose=0)
y_test_pred = best_settings['model'].predict(X_test, verbose=0)

r2_train = r2_score(y_train, y_train_pred)
r2_val = r2_score(y_val, y_val_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)
r2_oos_small_shareholder_NN3 = R_oos(y_test, y_test_pred)  

print(f"Training R-squared: {r2_train}")
print(f"Validation R-squared: {r2_val}")
print(f"Test MSE: {mse_test}")
print(f"Test R-squared: {r2_test}")
print(f"R2OOS Small Shareholder NN3:{r2_oos_small_shareholder_NN3}")


# In[10]:


l2_strength = 0.0001
learning_rate = 0.001  

best_val_mse = float('inf')
best_settings = {'model': None}

for fold, (train_index, val_index, test_index) in enumerate(tscv.split(X, first_split_date=datetime.date(2005, 1, 31), second_split_date=datetime.date(2013, 1, 31))):
    
    X_train = X.loc[train_index].drop('交易月份', axis=1)
    y_train = y.loc[train_index]
    
    X_val = X.loc[val_index].drop('交易月份', axis=1)
    y_val = y.loc[val_index]

    X_test = X.loc[test_index].drop('交易月份', axis=1)
    y_test = y.loc[test_index]

    # 修改为四层的神经网络
    model = models.Sequential([
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2_strength), input_shape=(X_train.shape[1],)),  # 第一层
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),  # 第二层
        layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),  # 第三层
        layers.Dense(1, activation='linear', kernel_regularizer=regularizers.l2(l2_strength))  # 输出层
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mean_squared_error',
                  metrics=['mean_squared_error'])

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # 训练模型
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=100, callbacks=[early_stopping], batch_size=5000, verbose=1)

    # 评估模型
    val_mse = model.evaluate(X_val, y_val, verbose=0)[1]

    # 保存最佳模型
    if val_mse < best_val_mse:
        best_val_mse = val_mse
        best_settings['model'] = model

# 预测
y_train_pred = best_settings['model'].predict(X_train, verbose=0)
y_val_pred = best_settings['model'].predict(X_val, verbose=0)
y_test_pred = best_settings['model'].predict(X_test, verbose=0)

r2_train = r2_score(y_train, y_train_pred)
r2_val = r2_score(y_val, y_val_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)
r2_oos_small_shareholder_NN4 = R_oos(y_test, y_test_pred)  

print(f"Training R-squared: {r2_train}")
print(f"Validation R-squared: {r2_val}")
print(f"Test MSE: {mse_test}")
print(f"Test R-squared: {r2_test}")
print(f"R2OOS Small Shareholder NN4:{r2_oos_small_shareholder_NN4}")


# In[11]:


l2_strength = 0.0001
learning_rate = 0.001  

best_val_mse = float('inf')
best_settings = {'model': None}

for fold, (train_index, val_index, test_index) in enumerate(tscv.split(X, first_split_date=datetime.date(2005, 1, 31), second_split_date=datetime.date(2013, 1, 31))):
    
    X_train = X.loc[train_index].drop('交易月份', axis=1)
    y_train = y.loc[train_index]
    
    X_val = X.loc[val_index].drop('交易月份', axis=1)
    y_val = y.loc[val_index]

    X_test = X.loc[test_index].drop('交易月份', axis=1)
    y_test = y.loc[test_index]

    # 修改为五层的神经网络
    model = models.Sequential([
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2_strength), input_shape=(X_train.shape[1],)),  # 第一层
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),  # 第二层
        layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),  # 第三层
        layers.Dense(8, activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),  # 第四层
        layers.Dense(1, activation='linear', kernel_regularizer=regularizers.l2(l2_strength))  # 输出层
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mean_squared_error',
                  metrics=['mean_squared_error'])

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # 训练模型
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=100, callbacks=[early_stopping], batch_size=5000, verbose=1)

    # 评估模型
    val_mse = model.evaluate(X_val, y_val, verbose=0)[1]

    # 保存最佳模型
    if val_mse < best_val_mse:
        best_val_mse = val_mse
        best_settings['model'] = model

# 预测
y_train_pred = best_settings['model'].predict(X_train, verbose=0)
y_val_pred = best_settings['model'].predict(X_val, verbose=0)
y_test_pred = best_settings['model'].predict(X_test, verbose=0)

r2_train = r2_score(y_train, y_train_pred)
r2_val = r2_score(y_val, y_val_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)
r2_oos_small_shareholder_NN5 = R_oos(y_test, y_test_pred)  

print(f"Training R-squared: {r2_train}")
print(f"Validation R-squared: {r2_val}")
print(f"Test MSE: {mse_test}")
print(f"Test R-squared: {r2_test}")
print(f"R2OOS Small Shareholder NN5:{r2_oos_small_shareholder_NN5}")


# In[61]:


r2_oos_small_shareholder_NN1 = 0.011348605155944824
r2_oos_small_shareholder_NN2 = 0.012341618537902832
r2_oos_small_shareholder_NN3 = 0.03934425115585327
r2_oos_small_shareholder_NN4 = -0.07373440265655518
r2_oos_small_shareholder_NN5 = -0.13797211647033691

chart_small_shareholder = np.array([[r2_oos_small_shareholder_NN1],
                      [r2_oos_small_shareholder_NN2],
                      [r2_oos_small_shareholder_NN3],
                      [r2_oos_small_shareholder_NN4],
                      [r2_oos_small_shareholder_NN5]])

r2oos_NN_small_shareholder = pd.DataFrame(chart_small_shareholder, columns=['A.M.C.P.S. Bottom 30%'], index=['NN1', 'NN2', 'NN3', 'NN4', 'NN5'])

r2oos_NN_small_shareholder


# In[63]:


merged_df = pd.concat([r2oos_NN_all,
                       r2oos_NN_large_stocks, 
                       r2oos_NN_small_stocks,
                       r2oos_NN_large_shareholder,
                       r2oos_NN_small_shareholder], axis=1)
merged_df = merged_df.T
merged_df


# # 国企

# In[12]:


features = df_soe_stocks.columns[~df_soe_stocks.columns.isin(['证券代码',"mom1m"])].tolist()
X = df_soe_stocks[features]
y = df_soe_stocks["mom1m"]


# In[13]:


for fold, (train_index, val_index, test_index) in enumerate(tscv.split(X, first_split_date=datetime.date(2005, 1, 31), second_split_date=datetime.date(2013, 1, 31))):
    
    X_train = X.loc[train_index].drop('交易月份', axis=1)
    y_train = y.loc[train_index]
    
    X_val = X.loc[val_index].drop('交易月份', axis=1)
    y_val = y.loc[val_index]

    X_test = X.loc[test_index].drop('交易月份', axis=1)
    y_test = y.loc[test_index]

    # 修改为只有一层的神经网络
    model = models.Sequential([
                layers.Dense(1, activation='linear', kernel_regularizer=regularizers.l2(l2_strength), input_shape=(X_train.shape[1],))
            ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mean_squared_error',
                  metrics=['mean_squared_error'])

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # 训练模型
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=100, callbacks=[early_stopping], batch_size=5000, verbose=1)

    # 评估模型
    val_mse = model.evaluate(X_val, y_val, verbose=0)[1]

    # 保存最佳模型
    if val_mse < best_val_mse:
        best_val_mse = val_mse
        best_settings['model'] = model

# 预测
y_train_pred = best_settings['model'].predict(X_train, verbose=0)
y_val_pred = best_settings['model'].predict(X_val, verbose=0)
y_test_pred = best_settings['model'].predict(X_test, verbose=0)

r2_train = r2_score(y_train, y_train_pred)
r2_val = r2_score(y_val, y_val_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)
r2_oos_soe_stocks_NN1 = R_oos(y_test, y_test_pred)  

print(f"Training R-squared: {r2_train}")
print(f"Validation R-squared: {r2_val}")
print(f"Test MSE: {mse_test}")
print(f"Test R-squared: {r2_test}")
print(f"R2OOS Soe Stocks NN1:{r2_oos_soe_stocks_NN1}")


# In[14]:


l2_strength = 0.0001
learning_rate = 0.001  

best_val_mse = float('inf')
best_settings = {'model': None}

for fold, (train_index, val_index, test_index) in enumerate(tscv.split(X, first_split_date=datetime.date(2005, 1, 31), second_split_date=datetime.date(2013, 1, 31))):
    
    X_train = X.loc[train_index].drop('交易月份', axis=1)
    y_train = y.loc[train_index]
    
    X_val = X.loc[val_index].drop('交易月份', axis=1)
    y_val = y.loc[val_index]

    X_test = X.loc[test_index].drop('交易月份', axis=1)
    y_test = y.loc[test_index]

    # 修改为两层的神经网络
    model = models.Sequential([
        layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(l2_strength), input_shape=(X_train.shape[1],)),  # 第一层
        layers.Dense(1, activation='linear', kernel_regularizer=regularizers.l2(l2_strength))  # 输出层
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mean_squared_error',
                  metrics=['mean_squared_error'])

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # 训练模型
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=100, callbacks=[early_stopping], batch_size=5000, verbose=1)

    # 评估模型
    val_mse = model.evaluate(X_val, y_val, verbose=0)[1]

    # 保存最佳模型
    if val_mse < best_val_mse:
        best_val_mse = val_mse
        best_settings['model'] = model

# 预测
y_train_pred = best_settings['model'].predict(X_train, verbose=0)
y_val_pred = best_settings['model'].predict(X_val, verbose=0)
y_test_pred = best_settings['model'].predict(X_test, verbose=0)

r2_train = r2_score(y_train, y_train_pred)
r2_val = r2_score(y_val, y_val_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)
r2_oos_soe_stocks_NN2 = R_oos(y_test, y_test_pred)  

print(f"Training R-squared: {r2_train}")
print(f"Validation R-squared: {r2_val}")
print(f"Test MSE: {mse_test}")
print(f"Test R-squared: {r2_test}")
print(f"R2OOS Soe Stocks NN2:{r2_oos_soe_stocks_NN2}")


# In[15]:


l2_strength = 0.0001
learning_rate = 0.001  

best_val_mse = float('inf')
best_settings = {'model': None}

for fold, (train_index, val_index, test_index) in enumerate(tscv.split(X, first_split_date=datetime.date(2005, 1, 31), second_split_date=datetime.date(2013, 1, 31))):
    
    X_train = X.loc[train_index].drop('交易月份', axis=1)
    y_train = y.loc[train_index]
    
    X_val = X.loc[val_index].drop('交易月份', axis=1)
    y_val = y.loc[val_index]

    X_test = X.loc[test_index].drop('交易月份', axis=1)
    y_test = y.loc[test_index]

    # 修改为三层的神经网络
    model = models.Sequential([
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(l2_strength), input_shape=(X_train.shape[1],)),  # 第一层
        layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),  # 第二层
        layers.Dense(1, activation='linear', kernel_regularizer=regularizers.l2(l2_strength))  # 输出层
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mean_squared_error',
                  metrics=['mean_squared_error'])

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # 训练模型
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=100, callbacks=[early_stopping], batch_size=5000, verbose=1)

    # 评估模型
    val_mse = model.evaluate(X_val, y_val, verbose=0)[1]

    # 保存最佳模型
    if val_mse < best_val_mse:
        best_val_mse = val_mse
        best_settings['model'] = model

# 预测
y_train_pred = best_settings['model'].predict(X_train, verbose=0)
y_val_pred = best_settings['model'].predict(X_val, verbose=0)
y_test_pred = best_settings['model'].predict(X_test, verbose=0)

r2_train = r2_score(y_train, y_train_pred)
r2_val = r2_score(y_val, y_val_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)
r2_oos_soe_stocks_NN3 = R_oos(y_test, y_test_pred)  

print(f"Training R-squared: {r2_train}")
print(f"Validation R-squared: {r2_val}")
print(f"Test MSE: {mse_test}")
print(f"Test R-squared: {r2_test}")
print(f"R2OOS Soe Stocks NN3:{r2_oos_soe_stocks_NN3}")


# In[16]:


l2_strength = 0.0001
learning_rate = 0.001  

best_val_mse = float('inf')
best_settings = {'model': None}

for fold, (train_index, val_index, test_index) in enumerate(tscv.split(X, first_split_date=datetime.date(2005, 1, 31), second_split_date=datetime.date(2013, 1, 31))):
    
    X_train = X.loc[train_index].drop('交易月份', axis=1)
    y_train = y.loc[train_index]
    
    X_val = X.loc[val_index].drop('交易月份', axis=1)
    y_val = y.loc[val_index]

    X_test = X.loc[test_index].drop('交易月份', axis=1)
    y_test = y.loc[test_index]

    # 修改为四层的神经网络
    model = models.Sequential([
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2_strength), input_shape=(X_train.shape[1],)),  # 第一层
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),  # 第二层
        layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),  # 第三层
        layers.Dense(1, activation='linear', kernel_regularizer=regularizers.l2(l2_strength))  # 输出层
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mean_squared_error',
                  metrics=['mean_squared_error'])

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # 训练模型
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=100, callbacks=[early_stopping], batch_size=5000, verbose=1)

    # 评估模型
    val_mse = model.evaluate(X_val, y_val, verbose=0)[1]

    # 保存最佳模型
    if val_mse < best_val_mse:
        best_val_mse = val_mse
        best_settings['model'] = model

# 预测
y_train_pred = best_settings['model'].predict(X_train, verbose=0)
y_val_pred = best_settings['model'].predict(X_val, verbose=0)
y_test_pred = best_settings['model'].predict(X_test, verbose=0)

r2_train = r2_score(y_train, y_train_pred)
r2_val = r2_score(y_val, y_val_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)
r2_oos_soe_stocks_NN4 = R_oos(y_test, y_test_pred)  

print(f"Training R-squared: {r2_train}")
print(f"Validation R-squared: {r2_val}")
print(f"Test MSE: {mse_test}")
print(f"Test R-squared: {r2_test}")
print(f"R2OOS Soe Stocks NN4:{r2_oos_soe_stocks_NN4}")


# In[17]:


l2_strength = 0.0001
learning_rate = 0.001  

best_val_mse = float('inf')
best_settings = {'model': None}

for fold, (train_index, val_index, test_index) in enumerate(tscv.split(X, first_split_date=datetime.date(2005, 1, 31), second_split_date=datetime.date(2013, 1, 31))):
    
    X_train = X.loc[train_index].drop('交易月份', axis=1)
    y_train = y.loc[train_index]
    
    X_val = X.loc[val_index].drop('交易月份', axis=1)
    y_val = y.loc[val_index]

    X_test = X.loc[test_index].drop('交易月份', axis=1)
    y_test = y.loc[test_index]

    # 修改为五层的神经网络
    model = models.Sequential([
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2_strength), input_shape=(X_train.shape[1],)),  # 第一层
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),  # 第二层
        layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),  # 第三层
        layers.Dense(8, activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),  # 第四层
        layers.Dense(1, activation='linear', kernel_regularizer=regularizers.l2(l2_strength))  # 输出层
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mean_squared_error',
                  metrics=['mean_squared_error'])

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # 训练模型
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=100, callbacks=[early_stopping], batch_size=5000, verbose=1)

    # 评估模型
    val_mse = model.evaluate(X_val, y_val, verbose=0)[1]

    # 保存最佳模型
    if val_mse < best_val_mse:
        best_val_mse = val_mse
        best_settings['model'] = model

# 预测
y_train_pred = best_settings['model'].predict(X_train, verbose=0)
y_val_pred = best_settings['model'].predict(X_val, verbose=0)
y_test_pred = best_settings['model'].predict(X_test, verbose=0)

r2_train = r2_score(y_train, y_train_pred)
r2_val = r2_score(y_val, y_val_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)
r2_oos_soe_stocks_NN5 = R_oos(y_test, y_test_pred)  

print(f"Training R-squared: {r2_train}")
print(f"Validation R-squared: {r2_val}")
print(f"Test MSE: {mse_test}")
print(f"Test R-squared: {r2_test}")
print(f"R2OOS Soe Stocks NN5:{r2_oos_soe_stocks_NN5}")


# In[64]:


r2_oos_soe_stocks_NN1 = -0.4697035551071167
r2_oos_soe_stocks_NN2 = 0.11208224296569824
r2_oos_soe_stocks_NN3 = 0.15095829963684082
r2_oos_soe_stocks_NN4 = 0.15028947591781616
r2_oos_soe_stocks_NN5 = 0.1426929235458374

chart_soe_stocks = np.array([[r2_oos_soe_stocks_NN1],
                      [r2_oos_soe_stocks_NN2],
                      [r2_oos_soe_stocks_NN3],
                      [r2_oos_soe_stocks_NN4],
                      [r2_oos_soe_stocks_NN5]])

r2oos_NN_soe_stocks = pd.DataFrame(chart_soe_stocks, columns=['SOE'], index=['NN1', 'NN2', 'NN3', 'NN4', 'NN5'])

r2oos_NN_soe_stocks


# # 非国企

# In[18]:


features = df_non_soe_stocks.columns[~df_non_soe_stocks.columns.isin(['证券代码',"mom1m"])].tolist()
X = df_non_soe_stocks[features]
y = df_non_soe_stocks["mom1m"]


# In[19]:


for fold, (train_index, val_index, test_index) in enumerate(tscv.split(X, first_split_date=datetime.date(2005, 1, 31), second_split_date=datetime.date(2013, 1, 31))):
    
    X_train = X.loc[train_index].drop('交易月份', axis=1)
    y_train = y.loc[train_index]
    
    X_val = X.loc[val_index].drop('交易月份', axis=1)
    y_val = y.loc[val_index]

    X_test = X.loc[test_index].drop('交易月份', axis=1)
    y_test = y.loc[test_index]

    # 修改为只有一层的神经网络
    model = models.Sequential([
                layers.Dense(1, activation='linear', kernel_regularizer=regularizers.l2(l2_strength), input_shape=(X_train.shape[1],))
            ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mean_squared_error',
                  metrics=['mean_squared_error'])

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # 训练模型
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=100, callbacks=[early_stopping], batch_size=5000, verbose=1)

    # 评估模型
    val_mse = model.evaluate(X_val, y_val, verbose=0)[1]

    # 保存最佳模型
    if val_mse < best_val_mse:
        best_val_mse = val_mse
        best_settings['model'] = model

# 预测
y_train_pred = best_settings['model'].predict(X_train, verbose=0)
y_val_pred = best_settings['model'].predict(X_val, verbose=0)
y_test_pred = best_settings['model'].predict(X_test, verbose=0)

r2_train = r2_score(y_train, y_train_pred)
r2_val = r2_score(y_val, y_val_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)
r2_oos_non_soe_stocks_NN1 = R_oos(y_test, y_test_pred)  

print(f"Training R-squared: {r2_train}")
print(f"Validation R-squared: {r2_val}")
print(f"Test MSE: {mse_test}")
print(f"Test R-squared: {r2_test}")
print(f"R2OOS non-Soe Stocks NN1:{r2_oos_non_soe_stocks_NN1}")


# In[20]:


l2_strength = 0.0001
learning_rate = 0.001  

best_val_mse = float('inf')
best_settings = {'model': None}

for fold, (train_index, val_index, test_index) in enumerate(tscv.split(X, first_split_date=datetime.date(2005, 1, 31), second_split_date=datetime.date(2013, 1, 31))):
    
    X_train = X.loc[train_index].drop('交易月份', axis=1)
    y_train = y.loc[train_index]
    
    X_val = X.loc[val_index].drop('交易月份', axis=1)
    y_val = y.loc[val_index]

    X_test = X.loc[test_index].drop('交易月份', axis=1)
    y_test = y.loc[test_index]

    # 修改为两层的神经网络
    model = models.Sequential([
        layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(l2_strength), input_shape=(X_train.shape[1],)),  # 第一层
        layers.Dense(1, activation='linear', kernel_regularizer=regularizers.l2(l2_strength))  # 输出层
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mean_squared_error',
                  metrics=['mean_squared_error'])

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # 训练模型
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=100, callbacks=[early_stopping], batch_size=5000, verbose=1)

    # 评估模型
    val_mse = model.evaluate(X_val, y_val, verbose=0)[1]

    # 保存最佳模型
    if val_mse < best_val_mse:
        best_val_mse = val_mse
        best_settings['model'] = model

# 预测
y_train_pred = best_settings['model'].predict(X_train, verbose=0)
y_val_pred = best_settings['model'].predict(X_val, verbose=0)
y_test_pred = best_settings['model'].predict(X_test, verbose=0)

r2_train = r2_score(y_train, y_train_pred)
r2_val = r2_score(y_val, y_val_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)
r2_oos_non_soe_stocks_NN2 = R_oos(y_test, y_test_pred)  

print(f"Training R-squared: {r2_train}")
print(f"Validation R-squared: {r2_val}")
print(f"Test MSE: {mse_test}")
print(f"Test R-squared: {r2_test}")
print(f"R2OOS non-Soe Stocks NN2:{r2_oos_non_soe_stocks_NN2}")


# In[21]:


l2_strength = 0.0001
learning_rate = 0.001  

best_val_mse = float('inf')
best_settings = {'model': None}

for fold, (train_index, val_index, test_index) in enumerate(tscv.split(X, first_split_date=datetime.date(2005, 1, 31), second_split_date=datetime.date(2013, 1, 31))):
    
    X_train = X.loc[train_index].drop('交易月份', axis=1)
    y_train = y.loc[train_index]
    
    X_val = X.loc[val_index].drop('交易月份', axis=1)
    y_val = y.loc[val_index]

    X_test = X.loc[test_index].drop('交易月份', axis=1)
    y_test = y.loc[test_index]

    # 修改为三层的神经网络
    model = models.Sequential([
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(l2_strength), input_shape=(X_train.shape[1],)),  # 第一层
        layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),  # 第二层
        layers.Dense(1, activation='linear', kernel_regularizer=regularizers.l2(l2_strength))  # 输出层
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mean_squared_error',
                  metrics=['mean_squared_error'])

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # 训练模型
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=100, callbacks=[early_stopping], batch_size=5000, verbose=1)

    # 评估模型
    val_mse = model.evaluate(X_val, y_val, verbose=0)[1]

    # 保存最佳模型
    if val_mse < best_val_mse:
        best_val_mse = val_mse
        best_settings['model'] = model

# 预测
y_train_pred = best_settings['model'].predict(X_train, verbose=0)
y_val_pred = best_settings['model'].predict(X_val, verbose=0)
y_test_pred = best_settings['model'].predict(X_test, verbose=0)

r2_train = r2_score(y_train, y_train_pred)
r2_val = r2_score(y_val, y_val_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)
r2_oos_non_soe_stocks_NN3 = R_oos(y_test, y_test_pred)  

print(f"Training R-squared: {r2_train}")
print(f"Validation R-squared: {r2_val}")
print(f"Test MSE: {mse_test}")
print(f"Test R-squared: {r2_test}")
print(f"R2OOS non-Soe Stocks NN3:{r2_oos_non_soe_stocks_NN3}")


# In[22]:


l2_strength = 0.0001
learning_rate = 0.001  

best_val_mse = float('inf')
best_settings = {'model': None}

for fold, (train_index, val_index, test_index) in enumerate(tscv.split(X, first_split_date=datetime.date(2005, 1, 31), second_split_date=datetime.date(2013, 1, 31))):
    
    X_train = X.loc[train_index].drop('交易月份', axis=1)
    y_train = y.loc[train_index]
    
    X_val = X.loc[val_index].drop('交易月份', axis=1)
    y_val = y.loc[val_index]

    X_test = X.loc[test_index].drop('交易月份', axis=1)
    y_test = y.loc[test_index]

    # 修改为四层的神经网络
    model = models.Sequential([
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2_strength), input_shape=(X_train.shape[1],)),  # 第一层
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),  # 第二层
        layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),  # 第三层
        layers.Dense(1, activation='linear', kernel_regularizer=regularizers.l2(l2_strength))  # 输出层
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mean_squared_error',
                  metrics=['mean_squared_error'])

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # 训练模型
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=100, callbacks=[early_stopping], batch_size=5000, verbose=1)

    # 评估模型
    val_mse = model.evaluate(X_val, y_val, verbose=0)[1]

    # 保存最佳模型
    if val_mse < best_val_mse:
        best_val_mse = val_mse
        best_settings['model'] = model

# 预测
y_train_pred = best_settings['model'].predict(X_train, verbose=0)
y_val_pred = best_settings['model'].predict(X_val, verbose=0)
y_test_pred = best_settings['model'].predict(X_test, verbose=0)

r2_train = r2_score(y_train, y_train_pred)
r2_val = r2_score(y_val, y_val_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)
r2_oos_non_soe_stocks_NN4 = R_oos(y_test, y_test_pred)  

print(f"Training R-squared: {r2_train}")
print(f"Validation R-squared: {r2_val}")
print(f"Test MSE: {mse_test}")
print(f"Test R-squared: {r2_test}")
print(f"R2OOS non-Soe Stocks NN4:{r2_oos_non_soe_stocks_NN4}")


# In[23]:


l2_strength = 0.0001
learning_rate = 0.001  

best_val_mse = float('inf')
best_settings = {'model': None}

for fold, (train_index, val_index, test_index) in enumerate(tscv.split(X, first_split_date=datetime.date(2005, 1, 31), second_split_date=datetime.date(2013, 1, 31))):
    
    X_train = X.loc[train_index].drop('交易月份', axis=1)
    y_train = y.loc[train_index]
    
    X_val = X.loc[val_index].drop('交易月份', axis=1)
    y_val = y.loc[val_index]

    X_test = X.loc[test_index].drop('交易月份', axis=1)
    y_test = y.loc[test_index]

    # 修改为五层的神经网络
    model = models.Sequential([
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2_strength), input_shape=(X_train.shape[1],)),  # 第一层
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),  # 第二层
        layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),  # 第三层
        layers.Dense(8, activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),  # 第四层
        layers.Dense(1, activation='linear', kernel_regularizer=regularizers.l2(l2_strength))  # 输出层
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mean_squared_error',
                  metrics=['mean_squared_error'])

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # 训练模型
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=100, callbacks=[early_stopping], batch_size=5000, verbose=1)

    # 评估模型
    val_mse = model.evaluate(X_val, y_val, verbose=0)[1]

    # 保存最佳模型
    if val_mse < best_val_mse:
        best_val_mse = val_mse
        best_settings['model'] = model

# 预测
y_train_pred = best_settings['model'].predict(X_train, verbose=0)
y_val_pred = best_settings['model'].predict(X_val, verbose=0)
y_test_pred = best_settings['model'].predict(X_test, verbose=0)

r2_train = r2_score(y_train, y_train_pred)
r2_val = r2_score(y_val, y_val_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)
r2_oos_non_soe_stocks_NN5 = R_oos(y_test, y_test_pred)  

print(f"Training R-squared: {r2_train}")
print(f"Validation R-squared: {r2_val}")
print(f"Test MSE: {mse_test}")
print(f"Test R-squared: {r2_test}")
print(f"R2OOS non-Soe Stocks NN5:{r2_oos_non_soe_stocks_NN5}")


# In[65]:


r2_oos_non_soe_stocks_NN1 = 0.004043102264404297
r2_oos_non_soe_stocks_NN2 = -0.03620803356170654
r2_oos_non_soe_stocks_NN3 = -0.10835087299346924
r2_oos_non_soe_stocks_NN4 = -0.027768850326538086
r2_oos_non_soe_stocks_NN5 = -0.10466206073760986

chart_non_soe_stocks = np.array([[r2_oos_non_soe_stocks_NN1],
                      [r2_oos_non_soe_stocks_NN2],
                      [r2_oos_non_soe_stocks_NN3],
                      [r2_oos_non_soe_stocks_NN4],
                      [r2_oos_non_soe_stocks_NN5]])

r2oos_NN_non_soe_stocks = pd.DataFrame(chart_non_soe_stocks, columns=['Non-SOE'], index=['NN1', 'NN2', 'NN3', 'NN4', 'NN5'])

r2oos_NN_non_soe_stocks


# In[66]:


merged_df = pd.concat([r2oos_NN_all,
                       r2oos_NN_large_stocks, 
                       r2oos_NN_small_stocks,
                       r2oos_NN_large_shareholder,
                       r2oos_NN_small_shareholder,
                       r2oos_NN_soe_stocks,
                       r2oos_NN_non_soe_stocks], 
                       axis=1)
merged_df = merged_df.T
merged_df


# In[68]:


merged_df.to_csv(r'r2oos results/r2oos_NN1-NN5.csv')


# In[ ]:




