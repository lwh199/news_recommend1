import argparse
import math
import os
import pickle
import random
import signal
import warnings
from collections import defaultdict
from random import shuffle
from multiprocessing import Pool

import numpy as np
import pandas as pd
from annoy import AnnoyIndex
from gensim.models import Word2Vec
from tqdm import tqdm

from utils import Logger, evaluate

warnings.filterwarnings('ignore')

seed = 2020
random.seed(seed)

# 命令行参数
parser = argparse.ArgumentParser(description='w2v 召回')
parser.add_argument('--mode', default='invalid')
parser.add_argument('--logfile', default='test.log')

args = parser.parse_args()

mode = args.mode
logfile = args.logfile

# 初始化日志
os.makedirs('../user_data/log', exist_ok=True)
log = Logger(f'../user_data/log/{logfile}').logger
log.info(f'w2v 召回，mode: {mode}')


def word2vec(df_, f1, f2, model_path):
    """
    该函数用于从输入数据框中提取句子列表，并训练或加载Word2Vec模型。
    最终返回一个字典，键是文章ID，值是对应的词向量。

    参数:
    df_ (DataFrame): 输入的数据框，包含用户和点击的文章ID。
    f1 (str): 用户标识列名。
    f2 (str): 文章标识列名。
    model_path (str): Word2Vec模型保存路径。

    返回:
    article_vec_map (dict): 文章ID到词向量的映射字典。
    """

    # 创建数据框的副本以避免修改原始数据
    df = df_.copy()

    # 按用户分组，并将每个用户的点击文章ID列表聚合为一个列表
    tmp = df.groupby(f1, as_index=False)[f2].agg(
        {'{}_{}_list'.format(f1, f2): list})

    # 将聚合后的列表转换为Python列表
    sentences = tmp['{}_{}_list'.format(f1, f2)].values.tolist()
    
    # 删除临时DataFrame中的列表列以释放内存
    del tmp['{}_{}_list'.format(f1, f2)]

    # 初始化一个空列表，用于存储所有单词（文章ID）
    words = []
    
    # 遍历每个用户的点击文章ID列表，将其转换为字符串列表，并添加到words列表中
    for i in range(len(sentences)):
        x = [str(x) for x in sentences[i]]  # 将每个文章ID转换为字符串
        sentences[i] = x                    # 更新sentences列表
        words += x                          # 将文章ID添加到words列表

    # 检查是否已经存在预训练的Word2Vec模型
    if os.path.exists(f'{model_path}/w2v.m'):
        # 如果存在，则直接加载模型
        model = Word2Vec.load(f'{model_path}/w2v.m')
    else:
        # 如果不存在，则使用当前的句子列表训练新的Word2Vec模型
        model = Word2Vec(sentences=sentences,
                         vector_size=256,               # 词向量维度
                         window=3,               # 上下文窗口大小
                         min_count=1,            # 忽略出现次数少于min_count的词语
                         sg=1,                   # 使用skip-gram算法（sg=0表示CBOW）
                         hs=0,                   # 使用负采样（hs=1表示使用层次softmax）
                         seed=seed,              # 随机种子，确保结果可复现
                         negative=5,             # 负采样数量
                         workers=10,             # 并行线程数
                         epochs=1)                 # 训练迭代次数
        # 保存训练好的模型
        model.save(f'{model_path}/w2v.m')

    # 初始化一个字典，用于存储文章ID到词向量的映射
    article_vec_map = {}
    
    # 遍历所有唯一的文章ID
    for word in set(words):
        # 如果文章ID在Word2Vec模型中存在，则将其词向量添加到字典中
        if word in model.wv:
            article_vec_map[int(word)] = model.wv[word]

    # 返回文章ID到词向量的映射字典
    return article_vec_map


def recall(df_query, article_vec_map, article_index, user_item_dict, worker_id):
    """
    该函数用于从给定的查询数据框中召回与用户历史交互相关的文章。
    它使用Word2Vec模型和索引结构来计算相似度得分，并生成推荐列表。

    参数:
    df_query (DataFrame): 包含用户ID和目标文章ID的查询数据框。
    article_vec_map (dict): 文章ID到词向量的映射字典。
    article_index (AnnoyIndex or similar): 文章向量的索引结构，用于快速查找最近邻。
    user_item_dict (dict): 用户ID到交互过的文章ID列表的映射字典。
    worker_id (int): 工作者ID，用于区分不同的进程或任务。

    返回:
    None: 结果保存为pickle文件。
    """
    data_list = []

    # 遍历查询数据框中的每一对用户ID和目标文章ID
    for user_id, item_id in tqdm(df_query.values):
        rank = defaultdict(int)

        # 获取用户最近交互的文章（这里只取最后一篇文章）
        interacted_items = user_item_dict[user_id]
        interacted_items = interacted_items[-1:]

        # 对于每个交互过的文章，找到最相似的100篇文章
        for item in interacted_items:
            article_vec = article_vec_map[item]

            # 使用索引结构找到最相似的文章及其距离
            item_ids, distances = article_index.get_nns_by_vector(
                article_vec, 100, include_distances=True)
            
            # 将距离转换为相似度得分（距离越小，相似度越高）
            sim_scores = [2 - distance for distance in distances]

            # 更新相似文章的得分
            for relate_item, wij in zip(item_ids, sim_scores):
                if relate_item not in interacted_items:
                    rank.setdefault(relate_item, 0)
                    rank[relate_item] += wij

        # 按相似度得分排序，选取前50个最相似的文章
        sim_items = sorted(rank.items(), key=lambda d: d[1], reverse=True)[:50]
        item_ids = [item[0] for item in sim_items]
        item_sim_scores = [item[1] for item in sim_items]

        # 创建临时数据框存储结果
        df_temp = pd.DataFrame()
        df_temp['article_id'] = item_ids
        df_temp['sim_score'] = item_sim_scores
        df_temp['user_id'] = user_id

        # 标记目标文章（如果有）
        if item_id == -1:
            df_temp['label'] = np.nan
        else:
            df_temp['label'] = 0
            df_temp.loc[df_temp['article_id'] == item_id, 'label'] = 1

        # 重新排列列顺序并确保数据类型正确
        df_temp = df_temp[['user_id', 'article_id', 'sim_score', 'label']]
        df_temp['user_id'] = df_temp['user_id'].astype('int')
        df_temp['article_id'] = df_temp['article_id'].astype('int')

        # 将当前用户的推荐结果添加到结果列表中
        data_list.append(df_temp)

    # 合并所有用户的推荐结果
    df_data = pd.concat(data_list, sort=False)

    # 确保保存目录存在，并将结果保存为pickle文件
    os.makedirs('../user_data/tmp/w2v', exist_ok=True)
    df_data.to_pickle('../user_data/tmp/w2v/{}.pkl'.format(worker_id))


if __name__ == '__main__':
    if mode == 'valid':
        df_click = pd.read_pickle('../user_data/data/offline/click.pkl')
        df_query = pd.read_pickle('../user_data/data/offline/query.pkl')

        os.makedirs('../user_data/data/offline', exist_ok=True)
        os.makedirs('../user_data/model/offline', exist_ok=True)

        w2v_file = '../user_data/data/offline/article_w2v.pkl'
        model_path = '../user_data/model/offline'
    else:
        df_click = pd.read_pickle('../user_data/data/online/click.pkl')
        df_query = pd.read_pickle('../user_data/data/online/query.pkl')

        os.makedirs('../user_data/data/online', exist_ok=True)
        os.makedirs('../user_data/model/online', exist_ok=True)

        w2v_file = '../user_data/data/online/article_w2v.pkl'
        model_path = '../user_data/model/online'

    log.debug(f'df_click shape: {df_click.shape}')
    log.debug(f'{df_click.head()}')

    article_vec_map = word2vec(df_click, 'user_id', 'click_article_id', model_path)
    f = open(w2v_file, 'wb')
    pickle.dump(article_vec_map, f)
    f.close()

    # 将 embedding 建立索引
    article_index = AnnoyIndex(256, 'angular')
    article_index.set_seed(2020)

    for article_id, emb in tqdm(article_vec_map.items()):
        article_index.add_item(article_id, emb)

    article_index.build(100)

    user_item_ = df_click.groupby('user_id')['click_article_id'].agg(lambda x: list(x)).reset_index()
    user_item_dict = dict(zip(user_item_['user_id'], user_item_['click_article_id']))

    # 召回
    n_split = 4
    all_users = df_query['user_id'].unique()
    shuffle(all_users)
    total = len(all_users)
    n_len = total // n_split

    # 清空临时文件夹
    for path, _, file_list in os.walk('../user_data/tmp/w2v'):
        for file_name in file_list:
            os.remove(os.path.join(path, file_name))

    # pool = Pool(n_split)
    # for i in range(0, total, n_len):
    #     part_users = all_users[i:i + n_len]
    #     df_temp = df_query[df_query['user_id'].isin(part_users)]
    #     pool.apply_async(recall, args=(df_temp, article_vec_map, article_index, user_item_dict, i))

    # pool.close()
    # pool.join()


    #不使用多线程
    for i in range(0, total, n_len):
        part_users = all_users[i:i + n_len]
        df_temp = df_query[df_query['user_id'].isin(part_users)]
        recall(df_temp, article_vec_map, article_index, user_item_dict, i)


    log.info('合并任务')

    df_data_list = []
    for path, _, file_list in os.walk('../user_data/tmp/w2v'):
        for file_name in file_list:
            df_temp = pd.read_pickle(os.path.join(path, file_name))
            df_data_list.append(df_temp)

    # 使用 pd.concat 合并所有数据框
    df_data = pd.concat(df_data_list, ignore_index=True)
    
    # 必须加，对其进行排序
    df_data = df_data.sort_values(['user_id', 'sim_score'], ascending=[True, False]).reset_index(drop=True)
    log.debug(f'df_data.head: {df_data.head()}')

    # 计算召回指标
    if mode == 'valid':
        log.info(f'计算召回指标')

        total = df_query[df_query['click_article_id'] != -1].user_id.nunique()

        hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(
            df_data[df_data['label'].notnull()], total)

        log.debug(
            f'w2v: {hitrate_5}, {mrr_5}, {hitrate_10}, {mrr_10}, {hitrate_20}, {mrr_20}, {hitrate_40}, {mrr_40}, {hitrate_50}, {mrr_50}'
        )
    # 保存召回结果
    if mode == 'valid':
        df_data.to_pickle('../user_data/data/offline/recall_w2v.pkl')
    else:
        df_data.to_pickle('../user_data/data/online/recall_w2v.pkl')