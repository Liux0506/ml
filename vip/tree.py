# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 00:16:15 2017

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
data   = []  
labels = []  
#with open("108_train.txt") as ifile:  
#        for line in ifile:  
#            tokens = line.strip().split('\t')  
#            data.append([float(tk) for tk in tokens[:-1]])  
#            labels.append(tokens[-1])  
#x = np.array(data)  
#labels = np.array(labels)  
#y = labels
#y = np.zeros(labels.shape)  # 创建长度等于labels的0矩阵

train_data=pd.read_csv("C:\\Users\\Liunux\\Desktop\\108_train.txt",sep='\t',
                       names=["u_spu_num" ,"u_brand_num" ,"u_cat_num","u_cat_spu","u_brand_spu",
                       "u_spu_freq","u_spu_date","u_spu_click_freq","u_spu_click_date","u_first_date" ,"u_last_date","u_spu_ratio","u_ratio","action_type"])    

x=train_data[["u_spu_num" ,"u_brand_num" ,"u_cat_num","u_cat_spu","u_brand_spu",
                       "u_spu_freq","u_spu_date","u_spu_click_freq","u_spu_click_date","u_first_date" ,"u_last_date","u_spu_ratio","u_ratio"]]
y=train_data[["action_type"]]

test_data=pd.read_csv("100_test.txt",sep='\t',names=["u_spu_num","u_brand_num","u_cat_num","u_cat_spu","u_brand_spu","u_spu_freq","u_spu_date","u_spu_click_freq","u_spu_click_date","u_first_date","u_last_date","u_spu_ratio","u_ratio","spu_id","u_id"],index_col=["spu_id","u_id"])

''''' 标签转换为0/1 '''  
#y[labels=='fat']=1  
  
''''' 拆分训练数据与测试数据 '''  
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)  
  
''''' 使用信息熵作为划分标准，对决策树进行训练 '''  
clf = tree.DecisionTreeClassifier(criterion='entropy')  
print(clf)  
clf.fit(x_train, y_train)  
  
''''' 把决策树结构写入文件 '''  
#with open("tree.dot", 'w') as f:  
#    f = tree.export_graphviz(clf, out_file=f)  
      
''''' 系数反映每个特征的影响力。越大表示该特征在分类中起到的作用越大 '''  
#print(clf.feature_importances_)  
feature_importances=clf.feature_importances_

'''''测试结果的打印'''  
answer1 = clf.predict(x_test)  
answer2 = clf.predict_proba(x)
print pd.DataFrame(answer2).head()
#print(x_train)  
#print(answer)  
#print(y_train)  
#print(np.mean( answer == y_train))  
  
'''''准确率与召回率'''  
#y_train=y_train.astype('float64')
#answer=answer.astype('float64')
#y_train.dtype
#print x_train[:,0]-y_train
precision, recall, thresholds = precision_recall_curve(y_test,answer1)  

#results=pd.DataFrame(answer1)
#results.to_csv('results.csv')  
#print answer

print(classification_report(y_test, answer1, target_names = ['0', '1']))



