# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 14:36:50 2017

@author: Liunux
"""

import pandas as pd
import time
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression  
from sklearn import metrics  
#from sklearn import tree

def main():
    #第一步：加载训练集和测试集
    #加载带标记数据
    '''
    train=pd.read_csv(r"D:/_Liunux/study/DataFountain/vip_datafountain/script/1w_train_0714.txt",sep='\t',encoding='utf8',
                         names=["u_id","u_buy_num","u_click_num","u_buy_date","u_click_date","u_num_ratio","u_date_ratio","u_buy_freq","u_click_freq","u_last_date",\
                                "spu_id","spu_buy_num","spu_click_num","spu_buy_date","spu_click_date","spu_num_ratio","spu_date_ratio","spu_buy_freq","spu_click_freq","spu_last_date",\
                                "brand_id","brand_buy_num","brand_click_num","brand_buy_date","brand_click_date","brand_num_ratio","brand_date_ratio","brand_buy_freq","brand_click_freq","brand_last_date",\
                                "cat_id","cat_buy_num","cat_click_num","cat_buy_date","cat_click_date","cat_num_ratio","cat_date_ratio","cat_buy_freq","cat_click_freq","cat_last_date",\
                                "action_type"],chunksize=1000) 
    '''
    train_0=pd.read_csv(r"/home/liuxun/vip/data/train_features_0712_n.txt",sep='\t',encoding='utf8',
                         names=["u_id","u_buy_num","u_click_num","u_buy_date","u_click_date","u_num_ratio","u_date_ratio","u_buy_freq","u_click_freq",\
                                "spu_id","spu_buy_num","spu_click_num","spu_buy_date","spu_click_date","spu_num_ratio","spu_date_ratio","spu_buy_freq","spu_click_freq",\
                                "brand_id","brand_buy_num","brand_click_num","brand_buy_date","brand_click_date","brand_num_ratio","brand_date_ratio","brand_buy_freq","brand_click_freq",\
                                "cat_id","cat_buy_num","cat_click_num","cat_buy_date","cat_click_date","cat_num_ratio","cat_date_ratio","cat_buy_freq","cat_click_freq",\
                                "action_type"],chunksize=560000) 
    
    train_1=pd.read_csv(r"/home/liuxun/vip/data/train_features_0712_p.txt",sep='\t',encoding='utf8',
                         names=["u_id","u_buy_num","u_click_num","u_buy_date","u_click_date","u_num_ratio","u_date_ratio","u_buy_freq","u_click_freq",\
                                "spu_id","spu_buy_num","spu_click_num","spu_buy_date","spu_click_date","spu_num_ratio","spu_date_ratio","spu_buy_freq","spu_click_freq",\
                                "brand_id","brand_buy_num","brand_click_num","brand_buy_date","brand_click_date","brand_num_ratio","brand_date_ratio","brand_buy_freq","brand_click_freq",\
                                "cat_id","cat_buy_num","cat_click_num","cat_buy_date","cat_click_date","cat_num_ratio","cat_date_ratio","cat_buy_freq","cat_click_freq",\
                                "action_type"])
    	#加载未标记数据
    unlabel=pd.read_csv(r"/home/liuxun/vip/data/test_features_0712.txt",sep='\t',encoding='utf8',
                        names=["id","u_id","spu_id","brand_id","cat_id",\
                                "u_buy_num","u_click_num","u_buy_date","u_click_date","u_num_ratio","u_date_ratio","u_buy_freq","u_click_freq",\
                                "spu_buy_num","spu_click_num","spu_buy_date","spu_click_date","spu_num_ratio","spu_date_ratio","spu_buy_freq","spu_click_freq",\
                                "brand_buy_num","brand_click_num","brand_buy_date","brand_click_date","brand_num_ratio","brand_date_ratio","brand_buy_freq","brand_click_freq",\
                                "cat_buy_num","cat_click_num","cat_buy_date","cat_click_date","cat_num_ratio","cat_date_ratio","cat_buy_freq","cat_click_freq"])
    
    test_df_t1=train_0.sample(frac=0.01)
    test_df_t=train_1.append(test_df_t1)#抽0.2验证
    test_df=test_df_t.sample(frac=0.2)
    test_X=test_df[['u_num_ratio','spu_num_ratio','brand_num_ratio','cat_num_ratio']]
    test_X_scale=preprocessing.scale(test_X)#归一化
    test_y=test_df['action_type']#类别
    max_f1_score=0
    i=0
    for chunk in train_0:
	i += 1
	print "第",i,"次循环"
        train=train_1.append(chunk)
	#print train.shape
        #标记
        #print "训练集，有", train.shape[0], "行", train.shape[1], "列" 
        x_train = train[['u_num_ratio','spu_num_ratio','brand_num_ratio','cat_num_ratio']]
        x_train_scale=preprocessing.scale(x_train)#归一化
        y_train = train['action_type']#类别  ds=train[train['action_type']==0]
        model =LogisticRegression()#tree.DecisionTreeClassifier()
        model.fit(x_train_scale, y_train)   
        #第五步：模型验证和选择
        predicted = model.predict(test_X_scale)  
        f1_score = metrics.f1_score(test_y, predicted) #模型评估
	print f1_score,max_f1_score
        if f1_score>max_f1_score:
            #第六步：模型预测
            max_f1_score=f1_score
            unlabe_X = unlabel[['u_num_ratio','spu_num_ratio','brand_num_ratio','cat_num_ratio']]
            unlabe_X_scale=preprocessing.scale(unlabe_X)#归一化
            unlabel_y=model.predict_proba(unlabe_X_scale)[:,1]#预测返回概率值，通过概率值阈值选择正例样本 
            out_y=pd.DataFrame(unlabel_y,columns=['prob']) #返回判定正例的比例 
            out_y["prob"]=out_y["prob"].apply(lambda x: '{0:.3f}'.format(x))
            out_1=out_y[out_y["prob"]>'0.5'] #看大于0.5的个数
            print "大于0.5的个数为：",out_1.shape
            #out_y['prob'].value_counts() #看值分布
            out_y.to_csv('result/0714.txt',index=False,header=None)#输出预测数据 
 
#执行
if __name__ == '__main__':  
    start = time.clock()  
    main()
    end = time.clock()  
    print('finish all in %s' % str(end - start)) 
