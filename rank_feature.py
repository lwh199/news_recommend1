import argparse
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from pandarallel import pandarallel  # 并行处理库

from utils import Logger  # 自定义日志模块

pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.max_rows', None)  # 显示所有行

pandarallel.initialize()  # 初始化并行处理环境

warnings.filterwarnings('ignore')  # 忽略警告信息

seed = 2020  # 设置随机种子

# 命令行参数解析
parser = argparse.ArgumentParser(description='排序特征')
parser.add_argument('--mode', default='invalid')  # 模式：验证或在线
parser.add_argument('--logfile', default='test.log')  # 日志文件名

args = parser.parse_args()

mode = args.mode
logfile = args.logfile

# 初始化日志
os.makedirs('../user_data/log', exist_ok=True)
log = Logger(f'../user_data/log/{logfile}').logger
log.info(f'排序特征，mode: {mode}')  # 记录日志信息


def func_if_sum(x, user_item_dict, item_sim):
    """
    计算用户历史点击文章与目标文章的相似度总和
    :param x: DataFrame 行，包含 'user_id' 和 'article_id'
    :param user_item_dict: 用户点击文章字典
    :param item_sim: 物品相似度字典
    :return: 相似度总和
    """
    user_id = x['user_id']
    article_id = x['article_id']

    interacted_items = user_item_dict[user_id][::-1]  # 用户历史点击的文章列表（倒序）

    sim_sum = 0
    for loc, i in enumerate(interacted_items):
        try:
            sim_sum += item_sim[i][article_id] * (0.7 ** loc)  # 使用指数衰减加权相似度
        except Exception as e:
            pass
    return sim_sum


def func_if_last(x, user_item_dict, item_sim):
    """
    计算用户最后一次点击文章与目标文章的相似度
    :param x: DataFrame 行，包含 'user_id' 和 'article_id'
    :param user_item_dict: 用户点击文章字典
    :param item_sim: 物品相似度字典
    :return: 最后一次点击的相似度
    """
    user_id = x['user_id']
    article_id = x['article_id']

    last_item = user_item_dict[user_id][-1]  # 用户最后一次点击的文章

    sim = 0
    try:
        sim = item_sim[last_item][article_id]
    except Exception as e:
        pass
    return sim


def func_binetwork_sim_last(x, user_item_dict, binetwork_sim):
    """
    计算基于二元网络的用户最后一次点击文章与目标文章的相似度
    :param x: DataFrame 行，包含 'user_id' 和 'article_id'
    :param user_item_dict: 用户点击文章字典
    :param binetwork_sim: 二元网络相似度字典
    :return: 最后一次点击的相似度
    """
    user_id = x['user_id']
    article_id = x['article_id']

    last_item = user_item_dict[user_id][-1]

    sim = 0
    try:
        sim = binetwork_sim[last_item][article_id]
    except Exception as e:
        pass
    return sim


def cosine_distance(vector1, vector2):
    """
    计算两个向量之间的余弦距离
    :param vector1: 第一个向量
    :param vector2: 第二个向量
    :return: 余弦距离
    """
    if not isinstance(vector1, np.ndarray) or not isinstance(vector2, np.ndarray):
        return -1
    distance = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    return distance


def func_w2w_sum(x, num, user_item_dict, article_vec_map):
    """
    计算用户历史点击文章与目标文章的词向量相似度总和
    :param x: DataFrame 行，包含 'user_id' 和 'article_id'
    :param num: 考虑的历史点击文章数量
    :param user_item_dict: 用户点击文章字典
    :param article_vec_map: 文章词向量字典
    :return: 相似度总和
    """
    user_id = x['user_id']
    article_id = x['article_id']

    interacted_items = user_item_dict[user_id][::-1][:num]  # 用户历史点击的文章列表（倒序，取前num个）

    sim_sum = 0
    for loc, i in enumerate(interacted_items):
        try:
            sim_sum += cosine_distance(article_vec_map[article_id], article_vec_map[i])
        except Exception as e:
            pass
    return sim_sum


def func_w2w_last_sim(x, user_item_dict, article_vec_map):
    """
    计算用户最后一次点击文章与目标文章的词向量相似度
    :param x: DataFrame 行，包含 'user_id' 和 'article_id'
    :param user_item_dict: 用户点击文章字典
    :param article_vec_map: 文章词向量字典
    :return: 最后一次点击的相似度
    """
    user_id = x['user_id']
    article_id = x['article_id']

    last_item = user_item_dict[user_id][-1]

    sim = 0
    try:
        sim = cosine_distance(article_vec_map[article_id], article_vec_map[last_item])
    except Exception as e:
        pass
    return sim


if __name__ == '__main__':
    # 根据模式加载不同的数据集
    if mode == 'valid':
        df_feature = pd.read_pickle('../user_data/data/offline/recall.pkl')
        df_click = pd.read_pickle('../user_data/data/offline/click.pkl')
    else:
        df_feature = pd.read_pickle('../user_data/data/online/recall.pkl')
        df_click = pd.read_pickle('../user_data/data/online/click.pkl')

    # 文章特征
    log.debug(f'df_feature.shape: {df_feature.shape}')

    df_article = pd.read_csv('../tcdata/articles.csv')  # 加载文章特征数据
    df_article['created_at_ts'] = df_article['created_at_ts'] / 1000  # 时间戳转换为秒
    df_article['created_at_ts'] = df_article['created_at_ts'].astype('int')
    df_feature = df_feature.merge(df_article, how='left')  # 合并召回结果和文章特征
    df_feature['created_at_datetime'] = pd.to_datetime(df_feature['created_at_ts'], unit='s')

    log.debug(f'df_article.head(): {df_article.head()}')
    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    # 历史记录相关特征
    df_click.sort_values(['user_id', 'click_timestamp'], inplace=True)  # 按用户ID和点击时间排序
    df_click.rename(columns={'click_article_id': 'article_id'}, inplace=True)  # 重命名列
    df_click = df_click.merge(df_article, how='left')  # 合并点击数据和文章特征

    df_click['click_timestamp'] = df_click['click_timestamp'] / 1000  # 时间戳转换为秒
    df_click['click_datetime'] = pd.to_datetime(df_click['click_timestamp'], unit='s', errors='coerce')
    df_click['click_datetime_hour'] = df_click['click_datetime'].dt.hour  # 提取小时信息

    # 用户点击文章的创建时间差的平均值
    df_click['user_id_click_article_created_at_ts_diff'] = df_click.groupby(['user_id'])['created_at_ts'].diff()
    df_temp = df_click.groupby(['user_id'])['user_id_click_article_created_at_ts_diff'].mean().reset_index()
    df_temp.columns = ['user_id', 'user_id_click_article_created_at_ts_diff_mean']
    df_feature = df_feature.merge(df_temp, how='left')

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    # 用户点击文章的时间差的平均值
    df_click['user_id_click_diff'] = df_click.groupby(['user_id'])['click_timestamp'].diff()
    df_temp = df_click.groupby(['user_id'])['user_id_click_diff'].mean().reset_index()
    df_temp.columns = ['user_id', 'user_id_click_diff_mean']
    df_feature = df_feature.merge(df_temp, how='left')

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    df_click['click_timestamp_created_at_ts_diff'] = df_click['click_timestamp'] - df_click['created_at_ts']

    # 点击文章的创建时间差的统计值
    df_temp_mean = df_click.groupby(['user_id'])['click_timestamp_created_at_ts_diff'].mean().reset_index()
    df_temp_mean.columns = ['user_id', 'user_click_timestamp_created_at_ts_diff_mean']

    df_temp_std = df_click.groupby(['user_id'])['click_timestamp_created_at_ts_diff'].std().reset_index()
    df_temp_std.columns = ['user_id', 'user_click_timestamp_created_at_ts_diff_std']

    df_temp = df_temp_mean.merge(df_temp_std, on='user_id')
    df_feature = df_feature.merge(df_temp, how='left')

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    # 点击的新闻的 click_datetime_hour 统计值
    df_temp_std = df_click.groupby(['user_id'])['click_datetime_hour'].std().reset_index()
    df_temp_std.columns = ['user_id', 'user_click_datetime_hour_std']
    df_feature = df_feature.merge(df_temp_std, how='left')

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    # 点击的新闻的 words_count 统计值
    df_temp_mean = df_click.groupby(['user_id'])['words_count'].mean().reset_index()
    df_temp_mean.columns = ['user_id', 'user_clicked_article_words_count_mean']

    df_temp_last = df_click.groupby(['user_id'])['words_count'].apply(lambda x: x.iloc[-1]).reset_index()
    df_temp_last.columns = ['user_id', 'user_click_last_article_words_count']

    df_temp = df_temp_mean.merge(df_temp_last, on='user_id')
    df_feature = df_feature.merge(df_temp, how='left')

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    # 点击的新闻的 created_at_ts 统计值
    df_temp_last = df_click.groupby('user_id')['created_at_ts'].apply(lambda x: x.iloc[-1]).reset_index()
    df_temp_last.columns = ['user_id', 'user_click_last_article_created_time']

    df_temp_max = df_click.groupby('user_id')['created_at_ts'].max().reset_index()
    df_temp_max.columns = ['user_id', 'user_clicked_article_created_time_max']

    df_temp = df_temp_last.merge(df_temp_max, on='user_id')
    df_feature = df_feature.merge(df_temp, how='left')

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    # 点击的新闻的 click_timestamp 统计值
    df_temp_last = df_click.groupby('user_id')['click_timestamp'].apply(lambda x: x.iloc[-1]).reset_index()
    df_temp_last.columns = ['user_id', 'user_click_last_article_click_time']

    df_temp_mean = df_click.groupby('user_id')['click_timestamp'].mean().reset_index()
    df_temp_mean.columns = ['user_id', 'user_clicked_article_click_time_mean']

    df_temp = df_temp_last.merge(df_temp_mean, on='user_id')
    df_feature = df_feature.merge(df_temp, how='left')

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    df_feature['user_last_click_created_at_ts_diff'] = df_feature['created_at_ts'] - df_feature['user_click_last_article_created_time']
    df_feature['user_last_click_timestamp_diff'] = df_feature['created_at_ts'] - df_feature['user_click_last_article_click_time']
    df_feature['user_last_click_words_count_diff'] = df_feature['words_count'] - df_feature['user_click_last_article_words_count']

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    # 计数统计
    for f in [['user_id'], ['article_id'], ['user_id', 'category_id']]:
        df_temp = df_click.groupby(f).size().reset_index()
        df_temp.columns = f + ['{}_cnt'.format('_'.join(f))]

        df_feature = df_feature.merge(df_temp, how='left')

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    # 召回相关特征
    ## itemcf 相关
    user_item_ = df_click.groupby('user_id')['article_id'].agg(list).reset_index()
    user_item_dict = dict(zip(user_item_['user_id'], user_item_['article_id']))

    if mode == 'valid':
        with open('../user_data/sim/offline/itemcf_sim.pkl', 'rb') as f:
            item_sim = pickle.load(f)
    else:
        with open('../user_data/sim/online/itemcf_sim.pkl', 'rb') as f:
            item_sim = pickle.load(f)

    # 用户历史点击物品与待预测物品相似度
    df_feature['user_clicked_article_itemcf_sim_sum'] = df_feature[['user_id', 'article_id']].parallel_apply(func_if_sum, axis=1, args=(user_item_dict, item_sim))
    df_feature['user_last_click_article_itemcf_sim'] = df_feature[['user_id', 'article_id']].parallel_apply(func_if_last, axis=1, args=(user_item_dict, item_sim))

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    ## binetwork 相关
    if mode == 'valid':
        with open('../user_data/sim/offline/binetwork_sim.pkl', 'rb') as f:
            binetwork_sim = pickle.load(f)
    else:
        with open('../user_data/sim/online/binetwork_sim.pkl', 'rb') as f:
            binetwork_sim = pickle.load(f)

    df_feature['user_last_click_article_binetwork_sim'] = df_feature[['user_id', 'article_id']].parallel_apply(func_binetwork_sim_last, axis=1, args=(user_item_dict, binetwork_sim))

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    ## w2v 相关
    if mode == 'valid':
        with open('../user_data/data/offline/article_w2v.pkl', 'rb') as f:
            article_vec_map = pickle.load(f)
    else:
        with open('../user_data/data/online/article_w2v.pkl', 'rb') as f:
            article_vec_map = pickle.load(f)

    # df_feature['user_clicked_article_w2v_sim_sum'] = df_feature[['user_id', 'article_id']].parallel_apply(lambda x: func_w2w_sum(x, 5, user_item_dict, article_vec_map), axis=1)
    # df_feature['user_last_click_article_w2v_sim'] = df_feature[['user_id', 'article_id']].parallel_apply(func_w2w_last_sim, axis=1, args=(user_item_dict, article_vec_map))
    df_feature['user_clicked_article_w2v_sim_sum'] = df_feature[['user_id', 'article_id']].apply(lambda x: func_w2w_sum(x, 5, user_item_dict, article_vec_map), axis=1)
    df_feature['user_last_click_article_w2v_sim'] = df_feature[['user_id', 'article_id']].apply(func_w2w_last_sim, axis=1, args=(user_item_dict, article_vec_map))

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    # 保存到本地
    if mode == 'valid':
        df_feature.to_pickle('../user_data/data/offline/features.pkl')
    else:
        df_feature.to_pickle('../user_data/data/online/features.pkl')