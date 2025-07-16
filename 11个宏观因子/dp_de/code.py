#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
import akshare as ak
import time

# 获取所有股票信息
all_stock = ak.stock_a_indicator_lg(symbol='all')
all_stock.to_csv("stk.csv", encoding='utf_8_sig', index=False)
total = len(list(all_stock['code']))
d = pd.DataFrame()

# 遍历股票代码，从第1000个开始
for i, stkcd in enumerate(list(all_stock['code'])[1000:]):
    try:
        # 获取股票的财务指标
        tmp = ak.stock_a_indicator_lg(symbol=str(stkcd))[['trade_date', 'dv_ratio', 'dv_ttm', 'total_mv']]
        tmp['Stkcd'] = str(stkcd)
        d = pd.concat([d, tmp])
        
        # 每处理10个代码打印进度
        if i % 10 == 0:
            print('---- finish {:d}/{:d} ----'.format(i + 1000, total))
        
        # 每处理1000个代码保存一次数据
        if i % 1000 == 0:
            d.to_csv("dividend__{:02d}.csv".format(1 + i // 1000), encoding='utf_8_sig', index=False)
            d = pd.DataFrame()

        # 请求之间加延迟
        time.sleep(1)

    except Exception as e:
        print(f"获取股票代码 {stkcd} 的数据时出错: {e}")

# 保存最后的数据
if not d.empty:
    d.to_csv("dividend.csv", encoding='utf_8_sig', index=False)


# In[9]:


import numpy as np
import pandas as pd
import os

d = pd.concat([pd.read_csv('dividend/' + file) for file in os.listdir('dividend')])
d


# In[19]:


import numpy as np
import pandas as pd
import os

d = pd.concat([pd.read_csv('dividend/' + file) for file in os.listdir('dividend')])

# 生成从200001到202410的月份列表
mon_lst = []
for y in range(2000, 2025):
    for m in range(1, 13):
        yearmon = int('{:d}{:02d}'.format(y, m))
        if yearmon >= 200001 and yearmon <= 202410:
            mon_lst.append(yearmon)

R2 = pd.DataFrame(mon_lst, columns=['Yearmon']) 
d['Yearmon'] = d['trade_date'].astype(str).replace('\-', '', regex=True).astype(int) // 100
d['dv_ratio'] = d['dv_ratio'].fillna(0)
d['dv_ttm'] = d['dv_ttm'].fillna(0)
d['dv'] = d[['dv_ratio', 'dv_ttm']].max(axis=1)
d['dv'] = d['dv'] * d['total_mv'] / 100
d = d[['Stkcd', 'Yearmon', 'dv', 'total_mv']]
d1 = pd.DataFrame(d.groupby(['Stkcd', 'Yearmon']).mean().reset_index())
d1 = pd.merge(R2, d1, on=['Yearmon'], how='left')
print(d1)

d2 = pd.DataFrame(d1.groupby(['Yearmon']).sum().reset_index())
d2['M_dp'] = d2['dv'] / d2['total_mv']
print(d2)

ep = pd.read_csv('M_ep.csv')

d2 = pd.merge(d2, ep, on=['Yearmon'], how='left')
d2['M_de'] = d2['M_dp'] / d2['M_ep']
d2['M_dp'] = np.log(d2['M_dp'])
d2['M_de'] = np.log(d2['M_de'])

factors = ['M_dp', 'M_de']
for f in factors:
    d2[['Yearmon', f]].to_csv(f'{f}.csv', encoding='utf_8_sig', index=False)


# In[ ]:




