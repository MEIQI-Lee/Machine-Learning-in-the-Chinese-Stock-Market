#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import math
import sklearn
import sklearn.preprocessing
import datetime
pd.options.mode.chained_assignment = None  


# 导入月度数据

# In[2]:


df1 = pd.read_csv(r"../Monthly_features/merged_Monthly.csv")
df1["交易月份"] = pd.to_datetime(df1["交易月份"]) 
df1 = df1[df1['交易月份'] != '2024-11-30']
df1


# 导入季度数据

# In[4]:


df2 = pd.read_csv(r"../Quarterly_features/merged_Quarterly.csv")
df2["会计期间"] = pd.to_datetime(df2["会计期间"])    
df2


# 导入半年度数据

# In[5]:


df3 = pd.read_csv(r"../Semi-annual_features/merged_semi-annual.csv")
df3["会计期间"] = pd.to_datetime(df3["会计期间"])    
df3


# 导入年度数据

# In[6]:


df5 = pd.read_csv(r"../Annual_features/merged_annual.csv")
df5["会计期间"] = pd.to_datetime(df5["会计期间"])    
df5


# ### 处理缺失值
# 缺失值被替换为每只股票每月的横截面中位数

# - 检查月度缺失值最多的变量，共1719162条数据：

# In[7]:


df1.isna().sum().sort_values(ascending = False)


# - 检查季度缺失值最多的变量，共571131条数据

# In[8]:


df2.isna().sum().sort_values(ascending = False)


# - 检查半年度缺失值最多的变量，共282681条数据

# In[9]:


df3.isna().sum().sort_values(ascending = False)


# - 检查年度缺失值最多的变量，共138456条数据

# In[12]:


df5.isna().sum().sort_values(ascending = False)


# - 月度数据：在每个月用横截面中位数替换缺失的值

# In[13]:


df1 = df1.fillna(df1.groupby('交易月份').transform('median'))


# In[14]:


df1.isna().sum().sort_values(ascending = False)


# In[15]:


#线性插值
df1 = df1.interpolate(method='linear')


# In[16]:


df1.isna().sum().sort_values(ascending = False)


# In[17]:


df1 = df1.fillna(df1.groupby('交易月份').transform('median'))


# In[18]:


df1.isna().sum().sort_values(ascending = False)


# In[19]:


df1


# In[20]:


print(len(df1))


# ## Explore the data before treating 
# 

# In[21]:


df1 = df1.sort_values(by=['交易月份', "证券代码"])


# In[22]:


#from 2000.01 to 2024.10 
print(df1['交易月份'].nunique())


# In[23]:


#共有5769只股票
print(df1['证券代码'].nunique())


# In[24]:


#Inspect variable types
df1.info(verbose=True)


# In[25]:


#Save dataset:
df1.to_csv(r'../Monthly_features/cleaned_merged_Monthly.csv', index = False)


# In[26]:


#大小股票,小股(每月市值最低的30%股票)和大股(每月市值最高的70%股票)
#mve : 月底市值的自然对数

# 筛掉 mve 为空值的证券代码
df2 = df2.dropna(subset=['mve'])

# 按 mve 列倒序排序
df2_sorted = df2.sort_values(by='mve', ascending=False)

# 计算小股和大股的分界线
cutoff_index = int(len(df2_sorted) * 0.3)

# 获取小股和大股
small_stocks = df2_sorted.tail(cutoff_index)  # 每月市值最低的30%
large_stocks = df2_sorted.head(len(df2_sorted) - cutoff_index)  # 每月市值最高的70%

# 输出结果
print("小股:")
print(small_stocks[['证券代码', 'mve']])
stk1 = small_stocks[['证券代码', 'mve']]
stk1.to_csv('small_stocks.csv', index=False)
print("\n大股:")
print(large_stocks[['证券代码', 'mve']])
stk2 = large_stocks[['证券代码', 'mve']]
stk2.to_csv('large_stocks.csv', index=False)


# In[27]:


'''
大小股东
我们从CSMAR收集了所有上市公司已发行A股的股东数量，每季度报告一次，以及相应的市值。
然后，我们计算每个股东的平均市值，即 A.M.C.P.S. = Market Cap / Number of Shareholders = 市值 / 股东数量
并将所有股票按照最高70%的门槛分成两组
'''


# In[28]:


df4 = pd.read_excel('HLD_Shrpro(Merge Query).xlsx')

df4.rename(columns={
    'HLD_Shrpro.Stkcd': '证券代码',
    'HLD_Shrpro.Reptdt': '会计期间',
    'HLD_Shrpro.S0101a': 'A股股东总数',
    'FI_T10.F100801A': '市值A'
}, inplace=True)

df4['证券代码'] = pd.to_numeric(df4['证券代码'], errors='coerce')
# 删除 '证券代码' 中的 NaN 值的行
df4.dropna(subset=['证券代码'], inplace=True)
# 将 '证券代码' 转换为整数类型
df4['证券代码'] = df4['证券代码'].astype(int)

df4 = df4.dropna(subset=['市值A', 'A股股东总数'])
df4 = df4[df4['A股股东总数'] != 0]

# 计算amcps
df4['amcps'] = df4['市值A'] / df4['A股股东总数']

# 直接计算amcps的平均值
df4_ave = df4.groupby(['证券代码'])['amcps'].mean().reset_index()
df4_ave.rename(columns={'amcps': 'amcps_ave'}, inplace=True)

# 按照amcps_ave降序排序
df4_ave_sorted = df4_ave.sort_values(by='amcps_ave', ascending=False)

# 计算小股东和大股东的分界线
cutoff_index = int(len(df4_ave_sorted) * 0.3)

# 获取小股和大股
small_shareholder = df4_ave_sorted.tail(cutoff_index)  # 每月市值最低的30%
large_shareholder = df4_ave_sorted.head(len(df4_ave_sorted) - cutoff_index)  # 每月市值最高的70%

# 输出结果
print("小股东:")
print(small_shareholder[['证券代码', 'amcps_ave']])
holder1 = small_shareholder[['证券代码', 'amcps_ave']]
holder1.to_csv('small_shareholder.csv', index=False)

print("\n大股东:")
print(large_shareholder[['证券代码', 'amcps_ave']])
holder2 = large_shareholder[['证券代码', 'amcps_ave']]
holder2.to_csv('large_shareholder.csv', index=False)


# In[29]:


#国企与非国企
#soe年频率。公司是国有的，则为1，否则为0
# 筛选国企和非国企的股票
soe_stocks = df5[df5['soe'] == 1][['证券代码', 'soe']]
non_soe_stocks = df5[df5['soe'] != 1][['证券代码', 'soe']]

soe_stocks.to_csv('soe_stocks.csv', index=False)
non_soe_stocks.to_csv('non_soe_stocks.csv', index=False)


# In[30]:


soe_stocks


# In[31]:


non_soe_stocks


# In[ ]:




