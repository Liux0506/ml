# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 15:27:06 2017

@author: Liunux
"""

import numpy as np  
#import scipy as sp 
import pandas as pd 
from sklearn import tree  
from sklearn.metrics import precision_recall_curve  
from sklearn.metrics import classification_report  
from sklearn.cross_validation import train_test_split  
  
  
#数据读入
train_data=pd.read_csv("108_train.txt",sep='\t',
                       names=["u_spu_num" ,"u_brand_num" ,"u_cat_num","u_cat_spu","u_brand_spu",
                       "u_spu_freq","u_spu_date","u_spu_click_freq","u_spu_click_date","u_first_date" ,"u_last_date","u_spu_ratio","u_ratio","action_type"])    

x_train=train_data[["u_spu_num" ,"u_brand_num" ,"u_cat_num","u_cat_spu","u_brand_spu",
                    "u_spu_freq","u_spu_date","u_spu_click_freq","u_spu_click_date","u_first_date" ,"u_last_date","u_spu_ratio","u_ratio"]]
y_train=train_data[["action_type"]]

test_data=pd.read_csv("100_test.txt",sep='\t',names=["u_spu_num","u_brand_num","u_cat_num","u_cat_spu","u_brand_spu","u_spu_freq","u_spu_date","u_spu_click_freq","u_spu_click_date","u_first_date","u_last_date","u_spu_ratio","u_ratio","spu_id","uid"])

x_test=train_data[["u_spu_num" ,"u_brand_num" ,"u_cat_num","u_cat_spu","u_brand_spu",
                    "u_spu_freq","u_spu_date","u_spu_click_freq","u_spu_click_date","u_first_date" ,"u_last_date","u_spu_ratio","u_ratio"]]

print train_data.iloc[:,:-1]

'''''使用信息熵作为划分标准，对决策树进行训练 '''  
clf = tree.DecisionTreeClassifier(criterion='entropy')  
  
clf.fit(x_train, y_train)  

'''''测试结果的打印'''  
answer = clf.predict(x_test)  
print(answer)
