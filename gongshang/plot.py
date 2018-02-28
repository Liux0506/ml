# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 14:24:35 2017

@author: Liunux
"""

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)

filepath="D:/_Liunux/study/python/Project/gongshang/hangye.txt"
data=pd.read_csv(filepath,sep=',',encoding='utf-8')
plot_data=data[data['月份']>=201401]


#plot_data["月份"]=plot_data["月份"].astype("str")

fig,ax = plt.subplots(figsize=(30, 10))

plt.xlabel('月份') 
plt.ylabel('新增数量')

x_labels=['201709','201708','201707','201706','201705','201704','201703','201702','201701','201612','201611','201610','201609','201608','201607','201606','201605','201604','201603','201602','201601','201512','201511','201510','201509','201508','201507','201506','201505','201504','201503','201502','201501','201412','201411','201410','201409','201408','201407','201406','201405','201404','201403','201402','201401']

    #print link
#plt_data["月份"]=plt_data["月份"].astype('str')
#绘制出每条路的折线图
for id in plot_data['行业']:
    #print (id)
    plt_data=plot_data.loc[(plot_data["行业"]==id)]
    #print link
    x=range(45)
    y=plt_data["新增数量"]
    plt.plot(x,y,label=id)
    plt.xticks(x,x_labels,rotation='vertical')
    ax.legend(loc='upper right')
#保存图片到本地 
fig.savefig("hangye.jpg")

plot_data['月份'][1][0:4]

import time

d=plot_data["月份"].astype("str")
for i in range(len(d)):
    d[i]=d[i][0:4]