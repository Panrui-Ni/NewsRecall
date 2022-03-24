import time, math, os
from tqdm import tqdm
import gc
import pickle
import random
from datetime import datetime
from operator import itemgetter
import numpy as np
import pandas as pd
import warnings
from collections import defaultdict
import collections
warnings.filterwarnings("ignore")



## 得到用户-点击文章-点击时间字典
def get_user_item_time(click_df):
    click_df = click_df.sort_values('click_timestamp') #按时间戳排序
    def make_item_time_pair(df):
        return list(zip(df['click_article_id'], df['click_timestamp']))
    #对每个用户，生成包含若干[点击文章,时间戳]的列表
    user_item_time_df = click_df.groupby('user_id')['click_article_id', 'click_timestamp'].apply(lambda x: make_item_time_pair(x)).reset_index().rename(columns={0: 'item_time_list'})
    user_item_time_dict = dict(zip(user_item_time_df['user_id'], user_item_time_df['item_time_list'])) #生成字典
    return user_item_time_dict


##获得点击最多的前 k 个文章
def get_item_topk_click(click_df, k):
    topk_click = click_df['click_article_id'].value_counts().index[:k]
    return topk_click


## item相似度计算，基于物品的协同过滤
def itemcf_sim(df):
    user_item_time_dict = get_user_item_time(df)
    i2i_sim = {}
    item_cnt = defaultdict(int)  # 初始化文章-点击次数字典
    for user, item_time_list in tqdm(user_item_time_dict.items()):
        for i, i_click_time in item_time_list:  # 在每个用户对应的 item-time 列表下
            item_cnt[i] += 1  # 计算每个item的点击数
            i2i_sim.setdefault(i, {})  # 构建{item:{}}列表
            for j, j_click_time in item_time_list:  # 考虑与 item_i 不同的 item_j
                if (i == j):
                    continue  # 如果文章id相同，继续下一轮循环
                i2i_sim[i].setdefault(j, 0)  # 生成 {item_i:{item_j:0, item_k:0}} (i,j,k不同)的列表
                i2i_sim[i][j] += 1 / math.log(len(item_time_list) + 1)
                # 计算item_i和item_j的相关系数，这里 item_i,j 都属于 user 下的 item_time_list，因此 len(item_time_list) 不变。此时同一个user点击了item_i和item_j，说明它们相似度大。
                # 比如列表为 user: [item_1,item_2,item_2]，那么sim[item_1][item_2]=sim[item_2][item_1]=1/log(3+1)*2
    # 对用户循环，即sim对用户求和，表示item_i与item_j出现在同一用户列表下的概率
    i2i_sim_ = i2i_sim.copy()
    for i, related_items in i2i_sim.items():
        for j, wij in related_items.items():
            i2i_sim_[i][j] = wij / math.sqrt(
                item_cnt[i] * item_cnt[j])  # 计算文章之间的相似度，分母同时除以cnt[i]和cnt[j]是为了排除情况：热门文章item_i很可能与任意item_j同时被一个user点击。

    pickle.dump(i2i_sim_, open('itemcf_i2i_sim.pkl', 'wb'))  # 保存为 pkl 文件

    return i2i_sim_


## 文章推荐
# 基于文章协同过滤的召回：
# sim_item_topk：选择与当前文章最相似的前k个文章；
# recall_item_num：最后召回的文章数量；
# item_topk_click：点击次数最多的文章，用于召回补全
# return 召回列表：{item1:score1, item2:score2...}
def item_based_recommend(user_id, user_item_time_dict, i2i_sim, sim_item_topk, recall_item_num, item_topk_click):
    # 获取user_id的历史交互文章
    user_hist_items = user_item_time_dict[user_id]  # 得到 user_id 对应的 [(item,timestamp)] 列表
    user_hist_items_ = {user_id for user_id, _ in user_hist_items}  # 提取 item

    item_rank = {}
    for loc, (i, click_time) in enumerate(user_hist_items):  # enumerate函数提取item和timestamp，loc记录循环步数，从0开始
        for j, wij in sorted(i2i_sim[i].items(), key=lambda x: x[1], reverse=True)[
                      :sim_item_topk]:  # item_j 为与 item_i 最相似的前 k 个文章
            if j in user_hist_items_:
                continue  # 如果 item_j 出现在用户点击历史里，继续下一轮循环

                item_rank.setdefault(j, 0)  # 初始化item_rank
                item_rank[j] += wij  # item_rank[item_j]更新为关于item_i的相关系数wij

        # 如果与 item_i 最相似的前 k 个文章去掉已经被点击的文章，依然小于需要召回的文章数，需要用点击量最高的文章进行补全
        if len(item_rank) < recall_item_num:
            for i, item in enumerate(item_topk_click):  # i记录循环步数，item为topk的文章
                if item in item_rank.items():
                    continue  # 补全的文章不应该出现在已经推荐的列表中
                item_rank[item] = -i - 100  # 赋任意负数，保证先推荐的是与历史点击相似度高的文章
                if len(item_rank) == recall_item_num:
                    break  # 如果补全了k个文章，结束

        item_rank = sorted(item_rank.items(), key=lambda x: x[1], reverse=True)[:recall_item_num]
        return item_rank


## 开始推荐
user_recall_items_dict = collections.defaultdict(dict) #初始化字典

tst_click = pd.read_csv("testA_click_log.csv")#.head(10)

trn_click = pd.read_csv("train_click_log.csv")#.head(10)

all_click = pd.concat([trn_click, tst_click])

#print(tst_click)

user_item_time_dict = get_user_item_time(all_click)

i2i_sim = pickle.load(open('itemcf_i2i_sim.pkl', 'rb')) #读取sim矩阵

sim_item_topk = 10

recall_item_num = 10  # 召回文章数量

item_topk_click = get_item_topk_click(all_click, k=50)  # 获取最热门的前 50 篇文章

for user in tqdm(all_click['user_id'].unique()):
    user_recall_items_dict[user] = item_based_recommend(user, user_item_time_dict, i2i_sim, sim_item_topk,
                                                        recall_item_num, item_topk_click)
    # 对每一个用户推荐文章

##召回字典转化成 DataFrame
user_item_score_list = []

for user, items in tqdm(user_recall_items_dict.items()):
    for item, score in items:  # score为 item与 user历史点击列表的相似度
        user_item_score_list.append([user, item, score])

recall_df = pd.DataFrame(user_item_score_list, columns=['user_id', 'click_article_id', 'pred_score'])

# print(recall_df)


## 生成提交文件
def submit(recall_df, topk=5, model_name=None):
    recall_df = recall_df.sort_values(by=['user_id', 'pred_score'])  # 对每个 user 的打分列表排序
    recall_df['rank'] = recall_df.groupby(['user_id'])['pred_score'].rank(ascending=False,
                                                                          method='first')  # 对每个user列表中的打分记录排序

    # 判断每个用户是否有5篇文章及以上
    tmp = recall_df.groupby('user_id').apply(lambda x: x['rank'].max())  # 打分排序的最大序号
    assert tmp.min() >= topk  # 要求 tmp 里面的最小值>=topk

    del recall_df['pred_score']  # 删除 recall_df里的打分列
    submit = recall_df[recall_df['rank'] <= topk].set_index(['user_id', 'rank']).unstack(-1).reset_index()
    # unstack函数，生成user列下，每个文章有对应排名的df

    submit.columns = [int(col) if isinstance(col, int) else col for col in submit.columns.droplevel(0)]
    # col为int型或者在submit.columns去掉索引中

    submit = submit.rename(
        columns={'': 'user_id', 1: 'article_1', 2: 'article_2', 3: 'article_3', 4: 'article_4', 5: 'article_5'})
    # 按照提交要求命名列名

    submit.to_csv('submit.csv', index=False, header=True)
    # return submit


#tst_click = pd.read_csv("testA_click_log.csv")

tst_users = tst_click['user_id'].unique() #把测试集中的用户识别出来
#trn_users = trn_click['user_id'].unique()
tst_recall = recall_df[recall_df['user_id'].isin(tst_users)]
#trn_recall = recall_df[recall_df['user_id'].isin(trn_users)]
#submit0 = submit(trn_recall, topk=5, model_name='itemcf_baseline')
submit1 = submit(tst_recall, topk=5, model_name='itemcf_baseline')
#print(submit0)
#print(submit1)


#dict = get_user_item_time(trn_click)
#print(list(dict.items())[:2]) #打印用户-点击文章-点击时间列表的前两行
#print(get_item_topk_click(trn_click,1)) #打印点击最多的文章id