# M_dividend

## 因子


- **M_dp** 200001-202410

- - 股息是在A股市场支付的12个月的移动股息
  - 股息价格比（Dividend Price Ratio）是中国A股市场总股息的对数与加权平均股价的对数之差

- **M_de** 200701-202410

- 派息率（Dividend Payout Ratio）是中国A股市场上所有上市股票的股息对数与收益对数之差



## 源数据

- 股息数据 dividend

  A 股个股指标: 市盈率, 市净率, 股息率，使用AKShare下载

  移动股息使用**股息率ttm**乘以**市值**计算
  
  
  
- 宏观因子M_ep（市盈率的倒数，盈利/市值）

  M_dp=股息/市值

  M_de=股息/盈利=M_dp/M_ep

  之后再取对数

  


- 下载数据至dividend`文件夹



