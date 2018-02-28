# -*- coding: utf-8 -*-
"""
Created on Wed Jul 05 15:20:13 2017

@author: Liunux
"""

import pandas as pd
import numpy as np

data=pd.read_csv("D:/_Liunux/study/DataFountain/vip_datafountain/data/user_action_good_train_1w.txt",sep="\t",
                 names=["uid","spu_id","action_type","start_date","brand_id","cat_id"],index_col=["uid","spu_id"])
#data.append([1,1,'','','',''],ignore_index=False)

'''查看索引、数据类型和内存信息  
print data.info()
'''

data2=pd.read_csv("D:/_Liunux/study/DataFountain/vip_datafountain/data/108_train.txt",sep='\t',
                       names=["u_spu_num" ,"u_brand_num" ,"u_cat_num","u_cat_spu","u_brand_spu",
                       "u_spu_freq","u_spu_date","u_spu_click_freq","u_spu_click_date","u_first_date" ,"u_last_date","u_spu_ratio","u_ratio","action_type"])    

x_train=data2[["u_brand_num" ,"u_cat_num","u_spu_ratio","u_ratio"]]
#x_train.reset_index(inplace=True)

for i in range(3):
    print x_train.iloc[:,i].name

from scipy import stats
for i in range(3): 
    print x_train.iloc[:,i].name
    print stats.shapiro(x_train.iloc[:,i])

x_train.loc[(x_train["u_ratio"] >= x_train["u_ratio"].mean()),['u_ratio']] = 1
x_train.loc[(x_train["u_ratio"] < x_train["u_ratio"].mean()),['u_ratio']] = 0

            
x_mean=data2[["u_spu_num","u_brand_num" ,"u_cat_num"]].mean()
ser=pd.Series([1,2,3,4])

'''离散化-每列的值小于均值则为0，大于为1'''
def SplitByMean(x):
    df_x=pd.Series(x)
    for iter in df_x.index:
        if df_x[iter]<df_x.mean():
            df_x[iter]=0
        else:
             df_x[iter]=1
    return df_x

data2=data2.apply(SplitByMean,axis=0)


mean.loc[:,"u_spu_num"]=1

data2.loc[(data2["u_cat_num"] > data2["u_cat_num"].mean()),'u_cat_num'] = 1


data = pd.DataFrame()
data['A'] = [12, 232, 21, 353, 2, 80]
data['B'] = [90, 789, -23, 32, -3232, 89]
data.loc[(data['A'] > 0) & (data['B'] < 0), 'A'] = 100


data2.replace(data2["u_spu_num"],x_mean["u_spu_num"])
replace_mean.head()



ser2=pd.Series(data2["u_spu_num"])

for iter in data2.index:
    if ser2[:iter]<ser2.mean():
        print 1
    
'''1.布尔索引(Boolean Indexing)'''
data_t1=data.loc[(data["action_type"]==1)&(data["start_date"]>230),["action_type","start_date","brand_id","cat_id"]]

'''2.apply使用'''             
#help查询pandas.DataFrame.apply
def num_missing(x):
    return sum(x.isnull())

data.apply(num_missing,axis=0)  #应用到每列
data.apply(num_missing,axis=1)  #应用到每行
                    
'''3.替换空值'''
#‘fillna()’ 可以一次解决这个问题。它被用来把缺失值替换为所在列的平均值/众数/中位数。
from scipy.stats import mode
mode(data["action_type"])
#记住，众数可以是个数组，因为高频的值可能不只一个。我们通常默认使用第一个：
mode(data["action_type"]).mode[0]

data["action_type"].fillna(mode(data["action_type"]).mode[0],inplace=True)
#再次检查缺失值以确认:
print data.apply(num_missing, axis=0)

#--u_brand_num
u_brand_group=data.groupby(["uid","brand_id"])
u_brand_num=u_brand_group["action_type"].sum().to_frame()
u_brand_num_1=u_brand_num.loc[(u_brand_num["action_type"]>0),["action_type"]]
#--u_cat_num
u_cat_group=data.groupby(["uid","cat_id"])
u_cat_num=u_brand_group["action_type"].sum()
print 
