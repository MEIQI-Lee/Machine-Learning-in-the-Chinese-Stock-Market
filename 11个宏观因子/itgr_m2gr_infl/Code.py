#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
d = pd.read_excel(r'CME_Mpi1.xlsx')
d['Yearmon'] = d['Staper'].astype('str').replace('\-', '', regex=True)
#print(d.dtypes)
d = d.loc[(d['Fresgn'] == 'M') & (d['Datasgn'] == 'PYM') & (d['Areasgn'] =='1')]
d = d.rename(columns={'Epim0101':'CPI'})
d = d[['Yearmon','CPI']][d['Yearmon'] >= '200001']
print(d)
d.to_csv('M_infl.csv',encoding='utf_8_sig',index = False)

d = pd.read_excel(r'CME_Mfinamkt1.xlsx')
d['Yearmon'] = d['Staper'].astype('str').replace('\-', '', regex=True)
#print(d.dtypes)
d = d.loc[(d['Fresgn'] == 'M') & (d['Datasgn'] == 'B')]
d = d.rename(columns={'Ezm0109':'M2'})
d = d[['Yearmon','M2']][d['Yearmon'] >= '200001']
print(d)
d.to_csv('M_m2gr.csv',encoding='utf_8_sig',index = False)

d = pd.read_excel(r'CME_Mftrd1.xlsx')
d['Yearmon'] = d['Staper'].astype('str').replace('\-', '', regex=True)
print(d.dtypes)
d = d.loc[(d['Fresgn'] == 'M') & (d['Datasgn'] == 'A')]
d = d.rename(columns={'Eftm0101':'trade'})
d['itgr'] = (d['trade'] - d['trade'].shift(12)) / d['trade'].shift(12)
d = d[['Yearmon','itgr']][d['Yearmon'] >= '200001']
print(d)
d.to_csv('M_itgr.csv',encoding='utf_8_sig',index = False)


# In[ ]:




