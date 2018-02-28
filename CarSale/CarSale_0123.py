# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 14:30:32 2018

@author: Liunux
"""

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import metrics  
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing

'''读取数据'''
train_data=pd.read_csv("train_data.csv")

'''特征工程'''
train_data["sale_date"]=train_data["sale_date"].astype("str")
train_data["month"]=train_data["sale_date"].apply(lambda x:x[-2:])

from sklearn.feature_selection  import SelectKBest,f_classif,chi2
features1=["sale_date","class_id","brand_id","compartment","type_id","department_id","displacement","driven_type_id","newenergy_type_id","emission_standards_id","if_MPV_id","if_luxurious_id","cylinder_number","car_length","car_width","car_height","total_quality","equipment_quality","wheelbase","front_track","rear_track","level_id_format","TR_format","gearbox_type_AMT","gearbox_type_AT","gearbox_type_AT;DCT","gearbox_type_CVT","gearbox_type_DCT","gearbox_type_MT","gearbox_type_MT;AT","if_charging_L","if_charging_T","price_level_format","price_format","fuel_type_id_format","power_fomat","engine_torque_format","rated_passenger_format","month"]
features2=["class_id","if_MPV_id","level_id_format","power_fomat","rated_passenger_format"]
selector = SelectKBest(f_classif, k=5)
corr=selector.fit(train_data[features2],train_data["sale_quantity"])
print(np.array(corr.scores_),'\n',corr.get_support())
#f_classif "power_fomat">"class_id">"if_MPV_id">"rated_passenger_format">"level_id_format"

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
rfe=RFE(lr,5)
x=train_data[features1]
y=train_data["sale_quantity"]
rfe_result=rfe.fit(x,y)
print(rfe.support_) 
print(rfe.ranking_)
# [False False False False False False False  True False False  True  True
#   True False False False False False False False False False False False
#  False False False False False False False False False False  True False
#  False False False]
# [32 33 21  8  5 13  2  1 14 11  1  1  1 31 25 26 30 29 28 24 23  3 18 19 10
#  34 15  7  4 35 16  6 12 20  1 22 27  9 17]

"class_id","if_MPV_id","level_id_format","power_fomat","rated_passenger_format"
'''类型数值化'''
features=["brand_id"]
le=preprocessing.LabelEncoder()
for feature in features:  
    train_data[feature]=le.fit_transform(train_data[feature])

train_data.info()
train_data.columns

'''归一化'''
train_data.head()


from sklearn.feature_selection import SelectKBest, f_classif

##计算变量间相关性
corr_mat=train_data.corr(method='pearson',min_periods=1)
sort_quan=corr_mat.sort_values(by='sale_quantity',ascending=False,axis=1)
print (sort_quan["sale_quantity"].sort_values(ascending=False))


'''画时间-销量图'''
import matplotlib.pyplot as plt
import datetime
import matplotlib.dates as mdate
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = [u'SimHei']  #FangSong/黑体 FangSong/KaiTi
mpl.rcParams['axes.unicode_minus'] = False

#date_quantity=data.groupby(["sale_date","class_id"],as_index=False)["sale_quantity"].sum()
#one_calss=train_data.loc[train_data["class_id"]==745137]
date_quantity=one_calss.groupby(["sale_date"],as_index=False)["sale_quantity"].sum()
date_quantity["sale_date"]=date_quantity["sale_date"].astype(str)

def str_to_datetime(x):
    return datetime.datetime.strptime(x,'%Y%m')
date_quantity["sale_date"]=date_quantity["sale_date"].apply(str_to_datetime)

fig1 = plt.figure(figsize=(15,5))
ax1 = fig1.add_subplot(1,1,1)
#设置时间标签显示格式
ax1.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m'))
#设置范围，显示月份
plt.xticks(pd.date_range(date_quantity["sale_date"].min(),date_quantity["sale_date"].max(),freq='M'),rotation=90)
plt.title("时间-销量")

plt.plot(date_quantity["sale_date"],date_quantity["sale_quantity"])
plt.savefig("date_quantity.jpg")

#每个品牌画线，太多了
class_unique=date_quantity["class_id"].unique()
#class_unique=[125403,136916,178529,194450,198427]
for i in class_unique:
    cur_data=date_quantity.loc[data["class_id"]==i]
    plt.plot(cur_data["sale_date"],cur_data["sale_quantity"])

'''历史均值为预测值'''
sale_quantity_mean=train_data.groupby(["class_id"],as_index=False)["sale_quantity"].mean()
result_mean=pd.merge(result,sale_quantity_mean,on="class_id",how="left")
result_mean[["predict_date","class_id","sale_quantity"]].to_csv("result_mean.csv",index=False,header=None)

'''线性回归对每个ID预测'''
train_0105=train_data[["sale_date","class_id","sale_quantity"]]
train_xy=train_0105[["sale_date","class_id","sale_quantity"]].loc[train_0105["sale_date"]<201710]
test_xy=train_0105[["sale_date","class_id","sale_quantity"]].loc[train_0105["sale_date"]==201710]

lr_model=linear_model.LinearRegression()#LinearRegression()
test_y_list=pd.DataFrame()
#x=103507
for x in test_xy["class_id"].unique():
    print (x)
    cur_train_xy=train_0105.loc[train_0105["class_id"]==x]
    cur_train_x=cur_train_xy[["sale_date"]]
    cur_train_y=cur_train_xy["sale_quantity"]
    lr_model.fit(cur_train_x, cur_train_y)
    pre_y=lr_model.predict(201711)
    test_y_list=test_y_list.append([[201711,x,float(pre_y)]],ignore_index=True)
    #test_y_list.append(str(i)+','+str(pre_y))

test_y_list.columns=["sale_date","class_id","sale_quantity"]
result_0105=pd.merge(result,test_y_list,on="class_id",how="left")
result_0105.to_csv("result_0105.csv",index=None,header=None)

train_0105[train_0105["class_id"]==103507].groupby("sale_date")["sale_quantity"].sum()
plt.scatter(cur_train_x,cur_train_y)