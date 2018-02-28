# -*- coding: utf-8 -*-

import pandas as pd
import time
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression  
from sklearn import metrics  
#from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
def main():
    #第一步：加载训练集和测试集
    #加载带标记数据
    label_ds=pd.read_csv(r"/home/liuxun/vip/data/train_features_0712.txt",sep='\t',encoding='utf8',
                         names=["u_id","u_buy_num","u_click_num","u_buy_date","u_click_date","u_num_ratio","u_date_ratio","u_buy_freq","u_click_freq",\
                                "spu_id","spu_buy_num","spu_click_num","spu_buy_date","spu_click_date","spu_num_ratio","spu_date_ratio","spu_buy_freq","spu_click_freq",\
                                "brand_id","brand_buy_num","brand_click_num","brand_buy_date","brand_click_date","brand_num_ratio","brand_date_ratio","brand_buy_freq","brand_click_freq",\
                                "cat_id","cat_buy_num","cat_click_num","cat_buy_date","cat_click_date","cat_num_ratio","cat_date_ratio","cat_buy_freq","cat_click_freq",\
                                "action_type"]) 
  
    print "训练集，有", label_ds.shape[0], "行", label_ds.shape[1], "列" 
    #加载未标记数据
    unlabel_ds=pd.read_csv(r"/home/liuxun/vip/data/test_features_0712.txt",sep='\t',encoding='utf8',
                         names=["id","u_id","spu_id","brand_id","cat_id",\
                                "u_buy_num","u_click_num","u_buy_date","u_click_date","u_num_ratio","u_date_ratio","u_buy_freq","u_click_freq",\
                                "spu_buy_num","spu_click_num","spu_buy_date","spu_click_date","spu_num_ratio","spu_date_ratio","spu_buy_freq","spu_click_freq",\
                                "brand_buy_num","brand_click_num","brand_buy_date","brand_click_date","brand_num_ratio","brand_date_ratio","brand_buy_freq","brand_click_freq",\
                                "cat_buy_num","cat_click_num","cat_buy_date","cat_click_date","cat_num_ratio","cat_date_ratio","cat_buy_freq","cat_click_freq"]) 
    
    
    #特征分析
    #des_f=unlabel_ds.loc[(unlabel_ds["spu_num_ratio"]>0)]
    #print des_f.shape
    #特征组合1:195万，'u_num_ratio','spu_num_ratio','brand_num_ratio','cat_num_ratio'
    
    
    #模型训练
    ds_0=label_ds[label_ds['action_type']==0]#标记为0的样本
    ds_0_train=ds_0.sample(frac=0.03)#抽0.01出来训练
    ds_1=label_ds[label_ds['action_type']==1]#标记为1的样本
    ds_train=ds_1.append(ds_0_train)
    label_X=ds_train[['u_num_ratio','spu_num_ratio','brand_num_ratio','cat_num_ratio']]
    label_X_scale=preprocessing.scale(label_X)#归一化
    label_y = ds_train['action_type']#类别  ds=label_ds[label_ds['action_type']==0]
    model = DecisionTreeRegressor()  #LogisticRegression()#tree.DecisionTreeClassifier()
    model.fit(label_X_scale, label_y)   
    #第五步：模型验证和选择
    test_df=ds_train.sample(frac=0.2)#抽0.2验证
    test_X=test_df[['u_num_ratio','spu_num_ratio','brand_num_ratio','cat_num_ratio']]
    test_X_scale=preprocessing.scale(test_X)#归一化
    test_y=test_df['action_type']#类别
    predicted = model.predict(test_X_scale)
    predicted = predicted.astype(int)
    f1_score = metrics.f1_score(test_y, predicted) #模型评估  
    print f1_score
    #第六步：模型预测
    unlabe_X = unlabel_ds[['u_num_ratio','spu_num_ratio','brand_num_ratio','cat_num_ratio']]
    unlabe_X_scale=preprocessing.scale(unlabe_X)#归一化
    #unlabel_y=model.predict_proba(unlabe_X_scale)[:,1]#预测返回概率值，通过概率值阈值选择正例样本 
    unlabel_y=model.predict(unlabe_X_scale)
    out_y=pd.DataFrame(unlabel_y,columns=['prob']) #返回判定正例的比例 
    out_y["prob"]=out_y["prob"].apply(lambda x: '{0:.3f}'.format(x))
    out_1=out_y[out_y["prob"]>'0.5'] #看大于0.5的个数
    print out_1.shape
    #out_y['prob'].value_counts() #看值分布
    out_y.to_csv('result/outvip_0714.txt',index=False,header=None)#输出预测数据 
    
#执行
if __name__ == '__main__':  
    start = time.clock()  
    main()
    end = time.clock()  
    print('finish all in %s' % str(end - start)) 
