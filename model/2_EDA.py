#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import math
import sklearn
import sklearn.preprocessing
import datetime
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
import seaborn as sns 
sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')
from collections import Counter
import warnings
warnings.simplefilter(action='ignore', category=Warning)
import seaborn as sns
sns.set()
pd.options.mode.chained_assignment = None  # default='warn'

from statsmodels.distributions.empirical_distribution import ECDF
from sklearn.metrics import mean_squared_error, r2_score


# In[2]:


df = pd.read_csv(r'../Monthly_features/cleaned_merged_Monthly.csv')


# In[3]:


print("Number of months: ", df["交易月份"].nunique())
print("Start: ", df["交易月份"].min())
print("End: ", df["交易月份"].max())


# In[4]:


print("Number of unique stocks: ", df["证券代码"].nunique())


# ### 特征的描述性统计

# In[5]:


features = df.columns[~df.columns.isin(['证券代码',"交易月份"])].tolist()


# In[6]:


descriptive_statistics = df[features].describe()
descriptive_statistics


# In[7]:


descriptive_statistics.to_excel('descriptive_statistics.xlsx')


# ### 每个特征的分布

# In[8]:


fig, ax = plt.subplots()
fig.set_figheight(30)
fig.set_figwidth(30)
df[features].hist(layout=(-1, 3), bins=np.linspace(-1,1,50), ax=ax);


# ### 单月累积回报

# In[9]:


plt.rcParams['figure.figsize'] = 15, 6

df_1 = df[["mom1m"]].rename(columns={'mom1m': 'Stock return'})

sns.histplot(data=df_1, x="Stock return", binwidth=0.01, 
             binrange=(df_1["Stock return"].min() + 0.00000000001, 
                       df_1["Stock return"].max() - 0.01))

plt.xlim(-1, 1)

plt.xlabel('Stock return')
plt.ylabel('Frequency')
plt.title('Cumulative returns in a single month.png')
plt.savefig('Cumulative returns in a single month.png')
plt.show()


# ## 月度变量相关性热力图

# In[10]:


features = df.columns[~df.columns.isin(['证券代码',"交易月份"])].tolist()
plt.figure(figsize = (18,18))
sns.heatmap(data=df[features].corr(), cmap='YlGnBu', annot=True, fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap')
plt.savefig('Correlation Heatmap.png')
plt.show()
plt.gcf().clear()


# ## 主要相关特征（Pearson 相关系数）

# In[11]:


c = df[features].corr().abs()
s = c.unstack()
so = s.sort_values(kind="quicksort").reset_index()
so.columns = ['Variable 1','Variable 2', 'corr']
so = so.sort_values(by = ['corr', 'Variable 1'], ascending = False)
so = so[so['corr']!=1]
so = so.iloc[::2].reset_index(drop=True)
so


# In[12]:


correlation = df[features].corr()["mom1m"].abs().sort_values(ascending = False)
corr_df = pd.DataFrame(correlation)
corr_df


# In[ ]:




