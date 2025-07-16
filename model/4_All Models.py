#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import math
import datetime
import os
import matplotlib.pyplot as plt
import warnings
import collections
import statsmodels.api as sm
from scipy.stats import t
from sklearn.utils import check_array
from functools import reduce
import seaborn as sns
import warnings
import collections
from matplotlib.pylab import rcParams
warnings.simplefilter(action='ignore', category=Warning)
import seaborn as sns
sns.set()
pd.options.mode.chained_assignment = None  # default='warn'
sns.set()


# In[14]:


r2oos_ols = pd.read_csv(r'r2oos results/r2oos_ols.csv',index_col = "Unnamed: 0")
r2oos_nn = pd.read_csv(r'r2oos results/r2oos_NN1-NN5.csv',index_col = "Unnamed: 0")
r2oos_enet = pd.read_csv(r'r2oos results/r2oos_enet.csv',index_col = "Unnamed: 0")
r2oos_pls = pd.read_csv(r'r2oos results/r2oos_pls.csv',index_col = "Unnamed: 0")
r2oos_rf = pd.read_csv(r'r2oos results/r2oos_rf.csv',index_col = "Unnamed: 0")
r2oos_gbrt = pd.read_csv(r'r2oos results/r2oos_gbrt.csv',index_col = "Unnamed: 0")
r2oos_lasso = pd.read_csv(r'r2oos results/r2oos_lasso.csv',index_col = "Unnamed: 0")


# In[19]:


dfs = [r2oos_ols, r2oos_pls, r2oos_lasso, r2oos_enet, r2oos_gbrt, r2oos_rf, r2oos_nn]
df = pd.concat(dfs, axis=1)
df = df[['OLS(+H)', 'PLS', 'LASSO(+H)','ENet(+H)', 'GBRT', 'RF', 'NN1', 'NN2', 'NN3', 'NN4', 'NN5']]
df


# In[24]:


rcParams['figure.figsize'] = 16, 4
df_ = df.reset_index().melt(id_vars="index", var_name="Model", value_name="R2OOS")

df_.rename(columns={'index': 'Sample'}, inplace=True)

sns.barplot(x='Model', y='R2OOS', hue='Sample', data=df_)
plt.savefig('R2OOS.png', bbox_inches='tight', dpi=1200)
plt.show()


# In[23]:


df.to_csv(r'r2oos results/r2oos_all.csv')

