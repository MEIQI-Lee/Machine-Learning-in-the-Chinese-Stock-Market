#!/usr/bin/env python
# coding: utf-8

# In[3]:


#用于执行基于时间的滚动窗口训练-验证-测试划分
import pandas as pd
import datetime
from datetime import datetime as dt
from dateutil.relativedelta import *

class TimeBasedCV(object):
    
    def __init__(self, train_period=30, val_period=7, test_period=7, freq='days'):
        self.train_period = train_period
        self.val_period = val_period
        self.test_period = test_period
        self.freq = freq

        
        
    def split(self, data, first_split_date, second_split_date, date_column='交易月份', gap=0):
        
        try:
            data[date_column]
        except:
            raise KeyError(date_column)
                    
        train_indices_list = []
        val_indices_list = []
        test_indices_list = []

       
        
        start_train = first_split_date - eval('relativedelta('+self.freq+'=self.train_period)')
        end_train = start_train + eval('relativedelta('+self.freq+'=self.train_period)')
        start_val = end_train + eval('relativedelta('+self.freq+'=gap)')
        end_val = start_val + eval('relativedelta('+self.freq+'=self.val_period)')
        start_test = end_val + eval('relativedelta('+self.freq+'=gap)')
        end_test = start_test + eval('relativedelta('+self.freq+'=self.test_period)')

        while end_test < data[date_column].max().date():
            # train indices:
            cur_train_indices = list(data[(data[date_column].dt.date>=start_train) & 
                                     (data[date_column].dt.date<end_train)].index)
            # validation indices:
            cur_val_indices = list(data[(data[date_column].dt.date>=start_val) & 
                                     (data[date_column].dt.date<end_val)].index)

            # test indices:
            cur_test_indices = list(data[(data[date_column].dt.date>=start_test) &
                                    (data[date_column].dt.date<end_test)].index)
            
            print("Train period:",start_train,"-" , end_train, ",val period:",start_val,"-" , end_val, ", Test period", start_test, "-", end_test,
                  "# train records", len(cur_train_indices),",# val records", len(cur_val_indices) , ", # test records", len(cur_test_indices))

            train_indices_list.append(cur_train_indices)
            val_indices_list.append(cur_val_indices)
            test_indices_list.append(cur_test_indices)

            # update dates:
            start_train = start_train + eval('relativedelta('+self.freq+'=self.val_period)')/2
            end_train = start_train + eval('relativedelta('+self.freq+'=self.train_period)')
            start_val = end_train + eval('relativedelta('+self.freq+'=gap)')
            end_val = start_val + eval('relativedelta('+self.freq+'=self.val_period)')            
            start_test = end_val + eval('relativedelta('+self.freq+'=gap)')
            end_test = start_test + eval('relativedelta('+self.freq+'=self.test_period)')
  
        index_output = [(train,val,test) for train,val,test in zip(train_indices_list,val_indices_list,test_indices_list)]

        self.n_splits = len(index_output)
        
        return index_output
    
    
    def get_n_splits(self):
        return self.n_splits 

