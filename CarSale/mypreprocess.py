# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 11:30:49 2018

@author: Liunux
"""

import pandas as pd
import numpy as np

'''读取数据'''
data=pd.read_csv("yancheng_train_20171226.csv",low_memory=False)
#tmp=data  #引用传递，改变train_data也会改变data
#tmp=data.iloc[:,0:32] #改变train_data不改变data

result=pd.read_csv("yancheng_testA_20171225.csv")

'''查看数据'''
data.describe().transpose()
data["class_id"].unique().shape  #140

'''自定义函数检查异常值'''
def wrong_num_detect(array):   
    wrong_num=0
    wrong={}
    for i,x in enumerate(array):
        if x.isdigit():
            continue
        else:
            wrong_num+=1
            wrong[i]=x
    print (len(wrong))
    return wrong       

def wrong_float_detect(array):   
    wrong_num=0
    wrong={}
    for i,x in enumerate(array):
        try:
            float(x)
        except ValueError:
            wrong_num+=1
            wrong[i]=x
    print (len(wrong))
    return wrong
    
def wrong_str_detect(array,instr):   
    wrong_num=0
    wrong={}
    for i,x in enumerate(array):
        if x.find(instr)==-1:
            continue
        else:
            wrong_num+=1
            wrong[i]=x
    print (len(wrong))
    return wrong

'''异常值处理'''
train_data=data.iloc[:,0:32] #改变train_data不改变data

train_data["level_id"].unique()
#'-'
wrong_num_detect(train_data["level_id"])
train_data["level_id_format"]=train_data["level_id"]   
#替换成-1
train_data["level_id_format"].loc[train_data["level_id_format"]=='-']=-1
train_data["level_id_format"]=train_data["level_id_format"].astype("int")

train_data["TR"].unique()
#'8;7', '5;4'
train_data["TR_format"]=train_data["TR"]
#替换成-1
train_data["TR_format"].loc[(train_data["TR_format"]=='8;7')|(train_data["TR_format"]=='5;4')]=-1
tmp1=wrong_num_detect(train_data["TR"])

tmp2=wrong_str_detect(train_data["gearbox_type"],';')
train_data["gearbox_type"].unique()
#'AT;DCT', 'MT;AT'
tmp1.keys()==tmp2.keys() #判断两个字典的键值是否一致
train_data.drop(["TR_format"],axis=1,inplace=True)
train_data["TR"]=train_data["TR"].astype("str")
train_data["gearbox_type"]=train_data["gearbox_type"].astype(str)
##one-hot编码
gearbox_type_format=pd.get_dummies(train_data["gearbox_type"],prefix='gearbox_type')
train_data=train_data.join(gearbox_type_format)

train_data["if_charging"].unique()
train_data["if_charging"]=train_data["if_charging"].astype("str")
#one-hot编码
if_charging_format=pd.get_dummies(train_data["if_charging"],prefix="if_charging")
train_data=train_data.join(if_charging_format)

train_data.dtypes

train_data["price_level"].unique()
price_level_map={'8-10W':2,'10-15W':3,'5WL':0,'15-20W':4,'5-8W':1,'25-35W':6,'35-50W':7,'20-25W':5,'50-75W':8}
#train_data.drop(["price_level_format"],axis=1,inplace=True)
train_data["price_level_format"]=train_data["price_level"].map(price_level_map)

price_tmp=wrong_float_detect(train_data["price"])
#'-' 8780
train_data["price_format"]=train_data["price"]
#替换成-1
train_data["price_format"].loc[train_data["price_format"]=='-']=-1
train_data["price_format"]=train_data["price_format"].astype("float")

train_data["fuel_type_id"].unique()
wrong_num_detect(data["fuel_type_id"])
train_data["fuel_type_id_format"]=train_data["fuel_type_id"]
train_data["fuel_type_id_format"].loc[train_data["fuel_type_id_format"]=='-']=-1
train_data["fuel_type_id_format"]=train_data["fuel_type_id_format"].astype("int64")

train_data["power"].unique()
#均值替换
power_tmp=train_data["power"].loc[train_data["power"]!='81/70']
power_mean=power_tmp.astype("float").mean()
train_data["power_fomat"]=train_data["power"]
train_data["power_fomat"]=train_data["power_fomat"].loc[train_data["power_fomat"]=='81/70']=str(power_mean)
train_data["power_fomat"]=train_data["power_fomat"].astype("float")
wrong_float_detect(train_data["power_fomat"])
#{17932: '81/70', 18554: '81/70', 18600: '81/70'}

train_data["engine_torque"].unique()
wrong_float_detect(train_data["engine_torque"])
#用-1代替
train_data["engine_torque_format"]=train_data["engine_torque"]
train_data["engine_torque_format"].loc[(train_data["engine_torque_format"]=='-')|(train_data["engine_torque_format"]=='155/140')]=-1
train_data["engine_torque_format"]=train_data["engine_torque_format"].astype("float")
#'-',155/140

train_data["rated_passenger"].unique()
rated_passenger_map={'5':1, '7-8':4, '7':3.5, '6-7':2.5, '6-8':3.8, '4':0, '4-5':0.8, '5-7':2, '5-8':3, '9':5}
train_data["rated_passenger_format"]=train_data["rated_passenger"].map(rated_passenger_map)

'''预处理后数据 '''
train_data.drop(["sale_date_format","sale_date_format2","level_id","TR","gearbox_type","if_charging","price_level","price","fuel_type_id","power","engine_torque","rated_passenger"],
                axis=1,inplace=True)

train_data.to_csv("train_data.csv",index=False)
