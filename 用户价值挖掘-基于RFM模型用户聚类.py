"""
案例背景：
    A公司是一家大型礼品公司，主要经营在线零售的业务，其客户来源于世界各地、进年来，随着电商行业的发展，竞争变得越来越激烈，A公司为了能够在竞争中取得优势，
争取主动，就必须维护好现有的客户，做好客户运营。”酒香不怕巷子深的时代已经一去不复返了“
    不过，怎样维护客户关系，也不可能对所有的客户都一视同仁，对于A公司来说，其客户种类众多，行为不一，他们对A公司的贡献成都也是大相径庭。因此，传统的手工
维护的方式会存在很大的困难,A公司迫切需要一种更加精准高效的方式，来挖掘客户价值，并根据价值对客户做分类管理。

任务目标：
    A公司收集了2010.12.01-2011.12.9期间所有客户的购买记录，假设当前时间为2012.01.01，我们现在的任务就是从该数据集中，挖掘所有客户的价值。并根据
客户的价值来进行分组管理，采取不同的策略。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="darkgrid", font_scale=1.2)
plt.rcParams["font.family"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False

"""
数据集描述：
'InvoiceNo'：订单编号，如果以C开头表示订单被取消
'StockCode'：商品编号
'Description'：商品描述信息
'Quantity'：商品购买数量
'InvoiceDate'：订单日期
'UnitPrice'：商品价格
'CustomerID'：客户ID
 'Country'：客户来自哪个国家
"""
data = pd.read_csv("./data/data.csv")
print(data.shape)
