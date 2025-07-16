#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

# 生成从200001到202410的月份列表
mon_lst = []
for y in range(2000, 2025):
    for m in range(1, 13):
        yearmon = int('{:d}{:02d}'.format(y, m))
        if yearmon >= 200001 and yearmon <= 202410:
            mon_lst.append(yearmon)

R2 = pd.DataFrame(mon_lst, columns=['Yearmon'])
R2['Year'] = R2['Yearmon'] // 100

# 读取数据
d = pd.read_excel(r"CME_Mstock2.xlsx").iloc[2:]

# 处理日期
d['Yearmon'] = d['Staper'].astype(str).replace('\-', '', regex=True).astype(int)

# 重命名列
d = d.rename(columns={
    'Esm0207': 'market_value',
    'Esm0217': 'PE',
    'Esm0210': 'turnover'
})

# 按年份分组并求和
dt = d[['Yearmon', 'Stocksgn', 'market_value', 'turnover']].groupby(['Yearmon']).sum().reset_index()
dt = dt[['Yearmon', 'market_value', 'turnover']]
dt['market_value'] = dt['market_value'].replace(0, np.nan)

# 获取年末数据
dt_end = dt[dt['Yearmon'] % 100 == 12][['Yearmon', 'market_value']]
dt_end['Yearmon'] = dt_end['Yearmon'] // 100
dt_end = dt_end.rename(columns={
    'Yearmon': 'Year',
    'market_value': 'year_market_value',
})

# 合并数据
dt = pd.merge(R2, dt, on=['Yearmon'], how='left')
dt = pd.merge(dt, dt_end, on=['Year'], how='left')

# 计算 M_ntis
dt['M_ntis'] = (dt['market_value'] - dt['market_value'].shift(12)) / dt['year_market_value']

# 计算 avg_PE
def avg_PE(s):
    pe_sh, pe_sz = list(s['PE'])
    m_sh, m_sz = list(s['market_value'])
    return (m_sh + m_sz) / (m_sh / pe_sh + m_sz / pe_sz)

tmp = pd.DataFrame(d[['Yearmon', 'Stocksgn', 'PE', 'market_value']].groupby(['Yearmon']).apply(avg_PE).reset_index())
tmp.columns = ['Yearmon', 'avg_PE']

# 合并 avg_PE 和计算 M_ep
dt = pd.merge(dt, tmp, on=['Yearmon'], how='left')
dt['M_ep'] = 1 / dt['avg_PE']
dt['M_mtr'] = dt['turnover'] / dt['market_value']

# 输出结果到 CSV 文件
factors = ['M_ntis', 'M_ep', 'M_mtr']
for f in factors:
    dt[['Yearmon', f]].to_csv(f'{f}.csv', encoding='utf_8_sig', index=False)


# In[ ]:




