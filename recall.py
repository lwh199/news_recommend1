import argparse
import os
import pickle
import random
import signal
import warnings
from collections import defaultdict
from itertools import permutations
from random import shuffle

import multitasking
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import Logger, evaluate  # 导入自定义的日志和评估函数

warnings.filterwarnings('ignore')  # 忽略所有警告

# 设置多线程参数
max_threads = multitasking.config['CPU_CORES']
multitasking.set_max_threads(max_threads)
multitasking.set_engine('process')
signal.signal(signal.SIGINT, multitasking.killall)  # 捕获中断信号并终止所有任务

random.seed(2020)  # 设置随机种子以确保结果可复现

# 解析命令行参数
parser = argparse.ArgumentParser(description='召回合并')
parser.add_argument('--mode', default='valid')  # 模式：验证或在线
parser.add_argument('--logfile', default='test.log')  # 日志文件名

args = parser.parse_args()

mode = args.mode
logfile = args.logfile

# 初始化日志
os.makedirs('../user_data/log', exist_ok=True)
log = Logger(f'../user_data/log/{logfile}').logger
log.info(f'召回合并: {mode}')  # 记录日志信息


def mms(df):
    """
    对每个用户的相似度进行最小最大缩放（Min-Max Scaling）
    :param df: DataFrame 包含 'user_id' 和 'sim_score'
    :return: 缩放后的相似度列表
    """
    user_score_max = {}
    user_score_min = {}

    # 获取每个用户下的相似度的最大值和最小值
    for user_id, g in df[['user_id', 'sim_score']].groupby('user_id'):
        scores = g['sim_score'].values.tolist()
        user_score_max[user_id] = scores[0]
        user_score_min[user_id] = scores[-1]

    ans = []
    for user_id, sim_score in tqdm(df[['user_id', 'sim_score']].values):
        # 应用 Min-Max Scaling 并加上一个小常数防止除零错误
        ans.append((sim_score - user_score_min[user_id]) /
                   (user_score_max[user_id] - user_score_min[user_id]) +
                   10**-3)
    return ans


def recall_result_sim(df1_, df2_):
    """
    计算两个召回结果集之间的相似度
    :param df1_: 第一个召回结果 DataFrame
    :param df2_: 第二个召回结果 DataFrame
    :return: 相似度得分
    """
    df1 = df1_.copy()
    df2 = df2_.copy()

    # 将第一个数据集转换为用户-文章集合字典
    user_item_ = df1.groupby('user_id')['article_id'].agg(set).reset_index()
    user_item_dict1 = dict(zip(user_item_['user_id'], user_item_['article_id']))

    # 将第二个数据集转换为用户-文章集合字典
    user_item_ = df2.groupby('user_id')['article_id'].agg(set).reset_index()
    user_item_dict2 = dict(zip(user_item_['user_id'], user_item_['article_id']))

    cnt = 0
    hit_cnt = 0

    # 计算交集数量与总数量的比例
    for user in user_item_dict1.keys():
        item_set1 = user_item_dict1[user]

        cnt += len(item_set1)

        if user in user_item_dict2:
            item_set2 = user_item_dict2[user]

            inters = item_set1 & item_set2
            hit_cnt += len(inters)

    return hit_cnt / cnt


if __name__ == '__main__':
    # 根据模式加载不同的数据集
    if mode == 'valid':
        df_click = pd.read_pickle('../user_data/data/offline/click.pkl')
        df_query = pd.read_pickle('../user_data/data/offline/query.pkl')

        recall_path = '../user_data/data/offline'
    else:
        df_click = pd.read_pickle('../user_data/data/online/click.pkl')
        df_query = pd.read_pickle('../user_data/data/online/query.pkl')

        recall_path = '../user_data/data/online'

    log.debug(f'max_threads {max_threads}')  # 记录最大线程数

    # 定义不同的召回方法及其权重
    recall_methods = ['itemcf', 'w2v', 'binetwork']
    weights = {'itemcf': 1, 'binetwork': 1, 'w2v': 0.1}
    recall_list = []
    recall_dict = {}

    # 加载每个召回方法的结果，并应用权重
    for recall_method in recall_methods:
        recall_result = pd.read_pickle(f'{recall_path}/recall_{recall_method}.pkl')
        weight = weights[recall_method]

        # 应用 Min-Max Scaling 并乘以权重
        recall_result['sim_score'] = mms(recall_result)
        recall_result['sim_score'] = recall_result['sim_score'] * weight

        recall_list.append(recall_result)
        recall_dict[recall_method] = recall_result

    # 计算不同召回方法之间的相似度
    for recall_method1, recall_method2 in permutations(recall_methods, 2):
        score = recall_result_sim(recall_dict[recall_method1], recall_dict[recall_method2])
        log.debug(f'召回相似度 {recall_method1}-{recall_method2}: {score}')

    # 合并所有召回结果
    recall_final = pd.concat(recall_list, sort=False)
    recall_score = recall_final[['user_id', 'article_id', 'sim_score']].groupby(['user_id', 'article_id'])['sim_score'].sum().reset_index()

    # 去重并保留最高分数的召回结果
    recall_final = recall_final[['user_id', 'article_id', 'label']].drop_duplicates(['user_id', 'article_id'])
    recall_final = recall_final.merge(recall_score, how='left')

    recall_final.sort_values(['user_id', 'sim_score'], inplace=True, ascending=[True, False])

    log.debug(f'recall_final.shape: {recall_final.shape}')
    log.debug(f'recall_final: {recall_final.head()}')

    # 删除无正样本的训练集用户
    gg = recall_final.groupby(['user_id'])
    useful_recall = []

    for user_id, g in tqdm(gg):
        if g['label'].isnull().sum() > 0:
            useful_recall.append(g)
        else:
            label_sum = g['label'].sum()
            if label_sum > 1:
                print('error', user_id)
            elif label_sum == 1:
                useful_recall.append(g)

    df_useful_recall = pd.concat(useful_recall, sort=False)
    log.debug(f'df_useful_recall: {df_useful_recall.head()}')

    df_useful_recall = df_useful_recall.sort_values(['user_id', 'sim_score'], ascending=[True, False]).reset_index(drop=True)

    # 计算相关指标
    if mode == 'valid':
        total = df_query[df_query['click_article_id'] != -1].user_id.nunique()
        hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(df_useful_recall[df_useful_recall['label'].notnull()], total)
        log.debug(f'召回合并后指标: {hitrate_5}, {mrr_5}, {hitrate_10}, {mrr_10}, {hitrate_20}, {mrr_20}, {hitrate_40}, {mrr_40}, {hitrate_50}, {mrr_50}')

    # 计算每个用户的平均召回数量
    df = df_useful_recall['user_id'].value_counts().reset_index()
    df.columns = ['user_id', 'cnt']
    log.debug(f"平均每个用户召回数量：{df['cnt'].mean()}")

    # 打印标签分布情况
    log.debug(f"标签分布: {df_useful_recall[df_useful_recall['label'].notnull()]['label'].value_counts()}")

    # 保存到本地
    if mode == 'valid':
        df_useful_recall.to_pickle('../user_data/data/offline/recall.pkl')
    else:
        df_useful_recall.to_pickle('../user_data/data/online/recall.pkl')