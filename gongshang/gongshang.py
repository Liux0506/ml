# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 14:36:57 2017

@author: Liunux
"""

import pandas as pd

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble  import ExtraTreesClassifier
from sklearn.cross_validation import train_test_split  
#1、读入文件
#filepath="/home/liuxun/gongshang/lx_features_1023.txt"
filepath="D:/_Liunux/study/python/Project/gongshang/1k_lx_features_1023.txt"


df=pd.read_csv(filepath,sep='\t',encoding='utf8',
             names=["pripid","enttype","industryco","regcap1","town","industry",
             "district","ontime","is_move","regorg","business_type","regstate",
             "regcap","industryphy","national_tax_status","land_tax_status",
             "social_status","fund_status","record_total","record_category_total",
             "brand_total","out_investment_count","out_investment_total",
             "investment_count","investment_total","investment_alter_count",
             "investment_alter_total","transfer_count","brand_cancelled_count",
             "arrears_social_security","arrears_accumulation_fund",
             "en_search_new_result","social_security_pay_an_state",
             "busst","empnum","empnumdis","assgro","assgrodis","liagro",
             "liagrodis","vendinc","vendincdis","maibusinc","maibusincdis",
             "progro","progrodis","netinc","netincdis","ratgro","ratgrodis",
             "totequ","totequdis","is_active"])

#输出空值比例，不包括为0的
#从mysql到导出的文件空值为'\N' 需先替换进行统计空值，sed -i 's/\\N//g' lx_features_1023.txt 
print("####各字段空值比例####")
print(df.count()/df.shape[0])
#df['out_investment_count'].value_counts()

#2、空值处理
data=df.fillna(0)

#3、输出基本统计
print (data.describe())

#4、字符类型数值化
data['industry'] = data['industry'].astype("str")
data['district'] = data['district'].astype("str")
data['ontime'] = data['ontime'].astype("str")
data['is_move'] = data['is_move'].astype("str")
data['regcap'] = data['regcap'].astype("str")
data['industryphy'] = data['industryphy'].astype("str")

le=preprocessing.LabelEncoder()
data['industry']=le.fit_transform(data['industry'])
data['district']=le.fit_transform(data['district'])
data['ontime']=le.fit_transform(data['ontime'])
data['is_move']=le.fit_transform(data['is_move'])
data['regcap']=le.fit_transform(data['regcap'])
data['industryphy']=le.fit_transform(data['industryphy'])
#df[df['ontime']=='1-3年']['ontime'].value_counts() #974
###编码反推类型
#tmp=le.fit_transform(df['industry'])
#le.classes_   #输出编码类型
#le.inverse_transform(tmp)

data_train=data.iloc[:,0:52]


'''输出为0的比例     
data_train=data.iloc[:,0:52]      
for i in data_train.columns:
    data_train[i] = data_train[i].astype("float")
    print (i,data_train[data_train[i]==0][i].count())
'''

##构建训练集
'''
train_x=data[["enttype","industryco","regcap1","town","industry",
             "district","ontime","is_move","regorg","business_type","regstate",
             "regcap","industryphy","national_tax_status","land_tax_status",
             "social_status","fund_status","record_total","record_category_total",
             "brand_total","out_investment_count","out_investment_total",
             "investment_count","investment_total","investment_alter_count",
             "investment_alter_total","transfer_count","brand_cancelled_count",
             "arrears_social_security","arrears_accumulation_fund",
             "en_search_new_result","social_security_pay_an_state",
             "busst","empnum","empnumdis","assgro","assgrodis","liagro",
             "liagrodis","vendinc","vendincdis","maibusinc","maibusincdis",
             "progro","progrodis","netinc","netincdis","ratgro","ratgrodis",
             "totequ","totequdis"]]

for i in train_x.columns:
    train_x[i] = train_x[i].astype("float")
    print (i,train_x[train_x[i]==0][i].count())
'''
train_x=data.iloc[:,1:52]
train_y=data["is_active"]

#industryco,town,busst有非法字符不能转换为float进行归一化
#去掉上述三列再进行归一化
train_x=train_x.drop(['industryco','town','busst'],axis=1)
train_x.dtypes
train_x_scale=preprocessing.scale(train_x)

#特征选择
model = ExtraTreesClassifier()
model.fit(train_x_scale, train_y)
# display the relative importance of each attribute
feature_importances=model.feature_importances_

df_features=pd.DataFrame(feature_importances,index=["enttype","regcap1","industry","district","ontime","is_move","regorg","business_type","regstate","regcap","industryphy","national_tax_status","land_tax_status","social_status","fund_status","record_total","record_category_total","brand_total","out_investment_count","out_investment_total","investment_count","investment_total","investment_alter_count","investment_alter_total","transfer_count","brand_cancelled_count","arrears_social_security","arrears_accumulation_fund","en_search_new_result","social_security_pay_an_state","empnum","empnumdis","assgro","assgrodis","liagro","liagrodis","vendinc","vendincdis","maibusinc","maibusincdis","progro","progrodis","netinc","netincdis","ratgro","ratgrodis","totequ","totequdis"],columns=["importance"])

print (df_features)

#训练集验证集拆分
X_train,X_test, y_train,y_test=train_test_split(train_x_scale,train_y,test_size=1/3,random_state=7)

model_lr=LogisticRegression()

model_lr.fit(X_train,y_train)

y_pre=model_lr.predict(X_test)
model_lr.score(y_pre,y_test)

from sklearn import svm
model_svm=svm.SVC()
model_svm.fit(X_train,y_train)
model_svm.score(y_pre,y_train)

