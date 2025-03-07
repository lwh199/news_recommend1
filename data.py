import argparse
import os
import random
from random import sample

import pandas as pd
from tqdm import tqdm

from utils import Logger

random.seed(2020)

# 命令行参数
parser = argparse.ArgumentParser(description='数据处理')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='test.log')

args = parser.parse_args()

mode = args.mode
logfile = args.logfile

# 初始化日志
os.makedirs('../user_data/log', exist_ok=True)
log = Logger(f'../user_data/log/{logfile}').logger
log.info(f'数据处理，mode: {mode}')


def data_offline(df_train_click, df_test_click):
    # 从训练集点击数据中提取所有用户ID，并转换为列表
    train_users = df_train_click['user_id'].values.tolist()
    
    # 从训练集用户中随机采样50000个用户作为验证集用户
    val_users = set(sample(train_users, 50000))
    
    # 记录验证集用户的数量（去重后的数量）
    log.debug(f'val_users num: {len(set(val_users))}')

    # 初始化用于存储处理后的点击数据和验证查询数据的列表
    click_list = []
    valid_query_list = []

    # 按用户ID对训练集点击数据进行分组
    groups = df_train_click.groupby(['user_id'])
    
    # 遍历每个用户及其对应的点击数据
    for user_id, g in tqdm(groups, desc='Processing train users'):
        if user_id[0] in val_users:
            # 如果用户在验证集中，取出该用户的最后一条点击记录作为验证查询
            valid_query = g.tail(1)
            valid_query_list.append(valid_query[['user_id', 'click_article_id']])

            # 将该用户的其他点击记录（除了最后一条）作为训练集的一部分
            train_click = g.head(g.shape[0] - 1)
            click_list.append(train_click)
        else:
            # 如果用户不在验证集中，将其全部点击记录加入训练集
            click_list.append(g)

    # 将所有处理后的点击数据合并为一个新的DataFrame
    df_train_click = pd.concat(click_list, sort=False)
    
    # 将所有验证查询数据合并为一个新的DataFrame
   
    df_valid_query = pd.concat(valid_query_list, sort=False)

    # 从测试集点击数据中提取所有用户ID，并确保唯一性
    test_users = df_test_click['user_id'].unique()
    
    # 初始化用于存储测试查询数据的列表
    test_query_list = []

    # 遍历每个测试用户，创建一个查询记录（初始文章ID设为-1表示未知）
    for user in tqdm(test_users, desc='Processing test users'):
        test_query_list.append([user, -1])

    # 将所有测试查询数据转换为DataFrame
    df_test_query = pd.DataFrame(test_query_list, columns=['user_id', 'click_article_id'])

    # 合并验证查询数据和测试查询数据
    df_query = pd.concat([df_valid_query, df_test_query], sort=False).reset_index(drop=True)
    
    # 合并处理后的训练点击数据和测试点击数据，并按'user_id'和'click_timestamp'排序
    df_click = pd.concat([df_train_click, df_test_click], sort=False).reset_index(drop=True)
    df_click = df_click.sort_values(['user_id', 'click_timestamp']).reset_index(drop=True)

    # 记录最终生成的查询数据和点击数据的形状及前几条记录
    log.debug(f'df_query shape: {df_query.shape}, df_click shape: {df_click.shape}')
    log.debug(f'{df_query.head()}')
    log.debug(f'{df_click.head()}')

    # 确保保存文件的目录存在，如果不存在则创建它
    os.makedirs('../user_data/data/offline', exist_ok=True)

    # 将处理后的点击数据和查询数据保存为pickle文件
    df_click.to_pickle('../user_data/data/offline/click.pkl')
    df_query.to_pickle('../user_data/data/offline/query.pkl')




def data_online(df_train_click, df_test_click):
    # 从测试集点击数据中提取所有用户ID，并确保唯一性
    test_users = df_test_click['user_id'].unique()
    
    # 初始化用于存储测试查询数据的列表
    test_query_list = []

    # 遍历每个测试用户，创建一个查询记录（初始文章ID设为-1表示未知）
    for user in tqdm(test_users, desc='Processing test users'):
        test_query_list.append([user, -1])

    # 将所有测试查询数据转换为DataFrame
    df_test_query = pd.DataFrame(test_query_list, columns=['user_id', 'click_article_id'])

    # 设置最终的查询数据为测试查询数据
    df_query = df_test_query
    
    # 合并训练点击数据和测试点击数据，并重置索引
    df_click = pd.concat([df_train_click, df_test_click], sort=False).reset_index(drop=True)
    
    # 按'user_id'和'click_timestamp'对合并后的点击数据进行排序，并重置索引
    df_click = df_click.sort_values(['user_id', 'click_timestamp']).reset_index(drop=True)

    # 记录最终生成的查询数据和点击数据的形状及前几条记录
    log.debug(f'df_query shape: {df_query.shape}, df_click shape: {df_click.shape}')
    log.debug(f'{df_query.head()}')
    log.debug(f'{df_click.head()}')

    # 确保保存文件的目录存在，如果不存在则创建它
    os.makedirs('../data/online', exist_ok=True)

    # 将处理后的点击数据和查询数据保存为pickle文件
    df_click.to_pickle('../user_data/data/online/click.pkl')
    df_query.to_pickle('../user_data/data/online/query.pkl')


if __name__ == '__main__':
    df_train_click = pd.read_csv('../tcdata/train_click_log.csv')
    df_test_click = pd.read_csv('../tcdata/testA_click_log.csv')

    log.debug(
        f'df_train_click shape: {df_train_click.shape}, df_test_click shape: {df_test_click.shape}'
    )

    if mode == 'valid':
        data_offline(df_train_click, df_test_click)
    else:
        data_online(df_train_click, df_test_click)
