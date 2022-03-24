import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rc('font', family='SimHei', size=13) # 显示中文
import os,gc,re,warnings,sys
warnings.filterwarnings("ignore") # 忽略报警

## 读取数据
trn_click = pd.read_csv("train_click_log.csv")
item_df = pd.read_csv("articles.csv")
item_df = item_df.rename(columns={'article_id': 'click_article_id'}) # 把 article_id 名称换成 click_article_id
item_emb_df = pd.read_csv('articles_emb.csv')
tst_click = pd.read_csv("testA_click_log.csv")
#print(trn_click.head()) # 读取前五行
#print(Data.describe())
#print(Data_train.describe())

## 计算每个用户的点击时间戳顺序，添加 rank 列
trn_click['rank'] = trn_click.groupby(['user_id'])['click_timestamp'].rank(ascending=False).astype(int)
#print(trn_click['rank'])
tst_click['rank'] = tst_click.groupby(['user_id'])['click_timestamp'].rank(ascending=False).astype(int)

## 计算每个用户的点击次数，添加 click_cnts 列
trn_click['click_cnts'] = trn_click.groupby(['user_id'])['click_timestamp'].transform('count')
tst_click['click_cnts'] = tst_click.groupby(['user_id'])['click_timestamp'].transform('count')

## 将 item_df 拼接上 trn_click
trn_click = trn_click.merge(item_df, how='left', on=['click_article_id'])
#print(trn_click.info()) #没有缺失值
#print(trn_click.describe())

#print(trn_click.user_id.nunique()) # 共有 20万用户
#print(trn_click.group.by('user_id')['click_article_id'].count().min()) # 每个用户至少点击了两篇文章


## 画图
plt.figure()
plt.figure(figsize=(15,20))
i=1

for col in ['click_article_id', 'click_timestamp', 'click_environment', 'click_deviceGroup', 'click_os', 'click_country', 'click_region', 'click_referrer_type', 'rank', 'click_cnts']:
    plot_envs = plt.subplot(5,2,i) # 画两列五行的子图
    i += 1
    v = trn_click[col].value_counts().reset_index()[:10] # 对col中的取值计算个数，取前十个
    fig = sns.barplot(x=v['index'], y=v[col]) # 画柱状图，横轴为 v[index]，纵轴为 v[col]，即 v[index] 出现的个数
    for item in fig.get_xticklabels():
        item.set_rotation(90) #旋转 x 轴上值书写的方向
    plt.title(col)
plt.tight_layout() # 调整子图相对位置
#plt.show() #在不同时间戳上的点击量比较均匀，点击环境基本上是 4，点击设备组大部分为 1，大部分在国家 1 点击，点击文章 id 比较均匀

tst_click = tst_click.merge(item_df, how='left', on=['click_article_id']) # 对测试集做同样处理
# print(tst_click.describe()) # 测试集的用户 id 和训练集不同，训练集为 0-199999，测试集为 200000-249999

## 分析 item
print(item_df.head()) #查看 item_df 的首尾行