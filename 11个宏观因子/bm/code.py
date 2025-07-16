#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd

# 生成从200001到202409的年月列表
mon_lst = []
for y in range(2000, 2025):
    for m in range(1, 13):
        yearmon = int('{:d}{:02d}'.format(y, m))
        if yearmon >= 200001 and yearmon <= 202409:
            mon_lst.append(yearmon)

# 生成季度的列表
season_lst = []
for y in range(2000, 2025):
    for m in range(3, 13, 3):
        yearmon = int('{:d}{:02d}'.format(y, m))
        if yearmon >= 200001 and yearmon <= 202409:
            season_lst.append(yearmon)

# 读取数据
d = pd.read_excel(r"FI_T10.xlsx").iloc[2:]
d['date'] = d['Accper'].astype(str).replace('\-', '', regex=True)
d['Season'] = d['date'].astype(int) // 100
d = d[(d['Season'] % 100) % 3 == 0]

# 数据处理
d['Stkcd'] = d['Stkcd'].astype(int)
d['Season'] = d['Season'].astype(int)
d = d.rename(columns={'F100801A': 'market', 'F101001A': 'book'})
d = d[['Stkcd', 'Season', 'market', 'book']]
d['book'] = d['book'] * d['market']

# 创建R2和R3 DataFrame
R2 = pd.DataFrame(mon_lst, columns=['Yearmon'])
R3 = pd.DataFrame(season_lst, columns=['Season'])
R2['num'] = R2['Yearmon'] // 100 * 10 + ((R2['Yearmon'] % 100) - 1) // 3
R3['num'] = R3['Season'] // 100 * 10 + (R3['Season'] % 100) // 3 - 1

# 合并数据
R4 = pd.merge(R2, R3, on=['num'], how='left')
tmp = pd.merge(R3, pd.DataFrame(d[['Season', 'market', 'book']].groupby(['Season']).sum().reset_index()), how='left', on=['Season'])

# 计算M_bm
tmp['M_bm'] = tmp['book'] / tmp['market']
tmp = pd.merge(R4[['Yearmon', 'Season']], tmp[['Season', 'M_bm']], how='left', on=['Season'])

# 输出结果到CSV文件
tmp[['Yearmon', 'M_bm']].to_csv('M_bm.csv', encoding='utf_8_sig', index=False)

# 打印最终结果
print(tmp[['Yearmon', 'M_bm']])


# In[ ]:




