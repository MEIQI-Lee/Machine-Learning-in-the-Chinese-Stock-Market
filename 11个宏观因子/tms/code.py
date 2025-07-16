#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
df = pd.read_excel('BND_TreasYield.xlsx')
df


# In[6]:


import numpy as np
import pandas as pd

# 生成从200001到202410的月份列表
mon_lst = []
for y in range(2000, 2025):
    for m in range(1, 13):
        yearmon = int('{:d}{:02d}'.format(y, m))
        if yearmon >= 200001 and yearmon <= 202410:
            mon_lst.append(yearmon)

# 读取数据
d = pd.read_excel('BND_TreasYield.xlsx').iloc[2:]
d['Yearmon'] = d['Trddt'].astype(str).replace('\-', '', regex=True).astype(int) // 100
d['Yeartomatu'] = d['Yeartomatu'].astype(str)

# 处理1年期收益率
d1 = d[d['Yeartomatu'] == '1'][['Yearmon', 'Yield']]
tmp1 = pd.DataFrame(d1.groupby(['Yearmon']).apply(lambda s: s['Yield'].sum() / s['Yield'].count()), columns=['r1']).reset_index()

# 创建R2 DataFrame
R2 = pd.DataFrame(mon_lst, columns=['Yearmon'])
tmp1 = pd.merge(R2, tmp1, on=['Yearmon'], how='left')

# 处理10年期收益率
d10 = d[d['Yeartomatu'] == '10'][['Yearmon', 'Yield']]
tmp10 = pd.DataFrame(d10.groupby(['Yearmon']).apply(lambda s: s['Yield'].sum() / s['Yield'].count()), columns=['r10']).reset_index()
tmp10 = pd.merge(R2, tmp10, on=['Yearmon'], how='left')

# 合并1年期和10年期收益率
tmp1 = pd.merge(tmp1, tmp10, on=['Yearmon'], how='left')
print(tmp1)

# 计算10年期减去1年期的收益率
tmp1['r10_minus_r1'] = tmp1['r10'] - tmp1['r1']

# 输出结果到CSV文件
tt = tmp1[['Yearmon', 'r10_minus_r1']]
tt = tt.rename(columns={'r10_minus_r1': 'M_tms'}) 
tt.to_csv('M_tms.csv', encoding='utf_8_sig', index=False)
print(tt)


# In[ ]:




