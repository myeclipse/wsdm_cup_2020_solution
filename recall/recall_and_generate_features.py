# -*- coding: utf-8 -*-
# encoding:utf-8
# @File  : generate_data.py
# @Author: liushuaipeng
# @Date  : 2019/12/28 10:44
# @Desc  :
import copy
import gc
import os
import pickle
import sys
import time
from multiprocessing.pool import Pool

import numpy as np
import pandas as pd
from tqdm import tqdm

if sys.version_info.major == 2:
    reload(sys)
    sys.setdefaultencoding('utf8')

corpus_folder = '../test/wsdm_diggsc_data_20191211'
# corpus_folder = '../../wsdm_diggsc_data_20191211'


used_recall_model_set = {
    'english_F1EXP_all_500_train_release', 'english_F1EXP_all_500_test',
    'english_bm25_all_500_train_release', 'english_bm25_all_500_test',
    'english_tfidf_all_500_train_release','english_tfidf_all_500_test',
}

#####reduce mem
import datetime


# 将召回的paperid和score融合
def load_data(recal_folder, name, max_recall_num, is_train):
    print(10 * '#', name, '#' * 10)
    recall_file = os.path.join(recal_folder, 'recall_{}.csv'.format(name))
    score_file = os.path.join(recal_folder, 'score_{}.csv'.format(name))

    line = open(recall_file, 'r').readline()
    MAX_RECALL_NUM = len(line.split(',')) - 1
    print('max_recall_num=', MAX_RECALL_NUM)

    recall_df = pd.read_csv(recall_file, sep=',', header=None,
                            names=['desc_id'] + [i for i in range(1, MAX_RECALL_NUM + 1)])
    recall_df = recall_df.fillna('')
    print(recall_file, '.shape=', recall_df.shape)

    score_df = pd.read_csv(score_file, sep=',', header=None,
                           names=['desc_id'] + [i for i in range(1, MAX_RECALL_NUM + 2)])
    score_df = score_df.fillna('')
    print(score_file, '.shape=', score_df.shape)

    assert len(recall_df) == len(score_df)
    assert recall_df.shape[1] == (score_df.shape[1] - 1)

    desc_id2paper = combine_descid_paperid_and_score(recall_df, score_df, max_recall_num, is_train)

    print(len(desc_id2paper))
    return [desc_id2paper, name]


def pool_load_recall_data(recal_folder, file_names, used_recall_num, is_train, f, worker=5):
    desc_id2paper_list = []
    cpu_worker = os.cpu_count()
    chunk_size = 1
    print('cpu 核心有：{}'.format(cpu_worker))
    if worker == -1 or worker > cpu_worker:
        worker = cpu_worker - 1
    print('使用cpu:{}'.format(worker))
    t1 = time.time()
    len_data = len(file_names)
    start = 0
    end = 0
    p = Pool(worker)
    res = []  # 保存的每个进程的返回值
    while end < len_data:
        end = start + chunk_size
        if end > len_data:
            end = len_data
        rslt = p.apply_async(f, (recal_folder, file_names[start], used_recall_num, is_train))
        start = end
        res.append(rslt)
    p.close()
    p.join()
    t2 = time.time()
    print('多线程耗时{}min'.format((t2 - t1) / 60))

    for i in res:
        desc_id2paper_list.append(i.get())
    return desc_id2paper_list


def combine_descid_paperid_and_score(recall_df, score_df, max_recall_num, is_train):
    desc_id2paper = {}
    for i, (idx1, recall_raw), (idx2, score_raw) in zip(tqdm(range(len(recall_df))), recall_df.iterrows(),
                                                        score_df.iterrows()):

        desc_id = recall_raw['desc_id']
        desc_id2 = score_raw['desc_id']

        if desc_id != desc_id2:
            print('desc_id=', desc_id)
            print('desc_id2=', desc_id2)

        assert desc_id == desc_id2

        paper2score = {}
        for i in range(1, max_recall_num + 1):
            paper_id = recall_raw[i]
            if is_train and paper_id == '5c0f7eebda562944ac8215e7':
                continue

            score = score_raw[i]
            paper2score[desc_id + "_" + paper_id] = score

        desc_id2paper[desc_id] = paper2score
    return desc_id2paper


def pool_construct_feature(file_name, recall_matrix_df_list, is_train, f, chunk_size, worker=5):
    print(20 * '#', 'loading ', file_name, 20 * '#')
    data_frame = pd.read_csv(os.path.join(corpus_folder, file_name), sep=',', header=0)
    data_frame = data_frame.fillna('')
    print(data_frame.shape)
    print('chunk_size=', chunk_size)

    cpu_worker = os.cpu_count()
    print('cpu 核心有：{}'.format(cpu_worker))
    if worker == -1 or worker > cpu_worker:
        worker = cpu_worker
    print('使用cpu:{}'.format(worker))
    t1 = time.time()
    len_data = len(data_frame)
    start = 0
    end = 0
    p = Pool(worker)
    res = []  # 保存的每个进程的返回值
    while end < len_data:
        end = start + chunk_size
        if end > len_data:
            end = len_data
        rslt = p.apply_async(f, (data_frame[start:end], copy.deepcopy(recall_matrix_df_list), is_train))
        print()
        start = end
        res.append(rslt)
    p.close()
    p.join()
    t2 = time.time()
    print('多线程耗时{}min'.format((t2 - t1) / 60))
    total_feature_matrix = np.concatenate([i.get()[0] for i in res], axis=0)
    labels, paper_id_list, descid_list = [], [], []
    for i in res:
        labels.extend(i.get()[1])
        descid_list.extend(i.get()[2])
        paper_id_list.extend(i.get()[3])

    print('total_feature_matrix.shape=', total_feature_matrix.shape)
    print('labels.shape=', np.array(labels).shape)
    lable_dict = {(i, labels.count(i)) for i in set(labels)}
    print('label distribution=', lable_dict)
    return total_feature_matrix, labels, descid_list, paper_id_list


def construct_feature(data_frame, desc_id2paper_list, is_train):
    print('construct_feature...')
    print(data_frame.shape)
    print(len(desc_id2paper_list))
    assert len(desc_id2paper_list) > 0
    feature_list = []
    labels = []
    descid_list = []
    paperid_list = []
    is_added_to_recall_data = []  # 0：表示召回数据；1：未召回，后被手动添加到训练集的数据

    not_find_num = 0.0
    for idx, (i, row) in zip(tqdm(range(len(data_frame))), data_frame.iterrows()):
        desc_id = row['description_id']
        has = False
        # desc_id对应的paper_id召回集合
        paper_id_set = set()
        for desc_id2paper, current_model_name in desc_id2paper_list:

            if current_model_name not in used_recall_model_set:
                continue

            paper2score = desc_id2paper[desc_id]
            paper_id_set = paper_id_set | set(paper2score.keys())

        # 创建train数据才执行
        added_paper_id = '-1'
        true_paper_id = '-1'
        if is_train:
            true_paper_id = row['paper_id']  # 真实匹配的paper_id
            if (desc_id + "_" + true_paper_id) not in paper_id_set:
                paper_id_set.add(desc_id + "_" + true_paper_id)
                added_paper_id = true_paper_id
                not_find_num += 1

        recall_paper_id_list = list(paper_id_set)
        assert len(recall_paper_id_list) > 0

        feature_matrix = np.zeros(shape=(len(recall_paper_id_list), len(desc_id2paper_list)))

        for row, paper_id in enumerate(recall_paper_id_list):
            paper_id = paper_id.split('_')[1]
            labels.append(int(true_paper_id == paper_id))
            descid_list.append(desc_id)
            paperid_list.append(paper_id)
            is_added_to_recall_data.append(int(added_paper_id == paper_id))

            for col, (desc_id2paper, current_model_name) in enumerate(desc_id2paper_list):
                score = find_score(desc_id2paper, desc_id, paper_id)
                feature_matrix[row][col] = score

        feature_list.append(feature_matrix)

    print('recall on trainset is ', 1 - not_find_num / len(data_frame))
    chuck_feature = np.concatenate(feature_list, axis=0)
    assert len(chuck_feature) == len(labels) == len(descid_list) == len(paperid_list)
    return chuck_feature, labels, descid_list, paperid_list, is_added_to_recall_data


def find_score(desc_id2paper, desc_id, paper_id):
    score = 0
    assert desc_id in desc_id2paper
    paper2score = desc_id2paper[desc_id]
    key = desc_id + '_' + paper_id
    if key in paper2score:
        score = paper2score[key]
    return score


def list_recall_names(recal_folder):
    names = set()
    if os.path.exists(recal_folder):
        files = os.listdir(recal_folder)
        for f in files:
            f = f.replace('recall_', '').replace('score_', '').replace('.csv', '')
            names.add(f)
    else:
        print(recal_folder, 'not exisit')
    name_list = sorted(list(names))
    return name_list


if __name__ == '__main__':
    TRAIN_USED_RECALL_NUM = 500
    TEST_USED_RECALL_NUM = 100

    start = time.time()
    #######################################train#################################################
    print(10 * "$$$$$$$$$$$$$", 'start generate train data.......')
    print('TRAIN_USED_RECALL_NUM=', TRAIN_USED_RECALL_NUM)
    train_recal_folder = './train'
    recall_names = list_recall_names(recal_folder=train_recal_folder)
    print(recall_names)
    pickle.dump(recall_names, open('pkl/recall_names_{}.pkl'.format(TRAIN_USED_RECALL_NUM), 'wb'), protocol=4)

    desc_id2paper_list = pool_load_recall_data(train_recal_folder, recall_names, TRAIN_USED_RECALL_NUM, is_train=True,
                                               f=load_data, worker=-1)

    print('recall_matrix_df_list.length=', len(desc_id2paper_list))

    data_frame = pd.read_csv(os.path.join(corpus_folder, 'train_release.csv'), sep=',', header=0)
    data_frame = data_frame.fillna('')
    print(data_frame.shape)
    train_feature_matrix, labels, descid_list, paper_id_list, is_added_to_recall_data = construct_feature(data_frame,
                                                                                                          desc_id2paper_list,
                                                                                                          is_train=True)

    with open('./pkl/train_label_{}.pkl'.format(TRAIN_USED_RECALL_NUM), 'wb')as f:
        pickle.dump(labels, f, protocol=4)
    with open('./pkl/train_feature_matrix_{}.pkl'.format(TRAIN_USED_RECALL_NUM), 'wb')as f:
        pickle.dump(train_feature_matrix, f, protocol=4)

    df = pd.DataFrame({'desc_id': descid_list, 'paper_id': paper_id_list, 'label': labels,
                       'is_added_to_recall_data': is_added_to_recall_data})
    df.to_csv('./data/train_{}.csv'.format(TRAIN_USED_RECALL_NUM), index=False)
    del desc_id2paper_list
    del train_feature_matrix
    del labels
    del descid_list
    del paper_id_list
    del is_added_to_recall_data
    gc.collect()
    end = time.time()
    print('\nuse time:', (end - start) / 60)
    #######################################test#################################################

    print('\n', 0 * "$$$$$$$$$$$$$", 'start generate test data.......')
    print('TEST_USED_RECALL_NUM=', TEST_USED_RECALL_NUM)
    test_recal_folder = './test'
    recall_names = [i.replace('train_release', 'test') for i in recall_names]
    print(recall_names)
    desc_id2paper_list = []

    desc_id2paper_list = pool_load_recall_data(test_recal_folder, recall_names, TEST_USED_RECALL_NUM, is_train=False,
                                               f=load_data, worker=-1)
    print('recall_matrix_df_list.length=', len(desc_id2paper_list))

    # 创建test数据
    data_frame = pd.read_csv(os.path.join(corpus_folder, 'test.csv'), sep=',', header=0)
    data_frame = data_frame.fillna('')
    print(data_frame.shape)
    test_feature_matrix, labels, descid_list, paper_id_list, is_added_to_recall_data = construct_feature(data_frame,
                                                                                                         desc_id2paper_list,
                                                                                                         is_train=False)

    with open('./pkl/test_feature_matrix_{}.pkl'.format(TEST_USED_RECALL_NUM), 'wb')as f:
        pickle.dump(test_feature_matrix, f, protocol=4)

    df = pd.DataFrame({'desc_id': descid_list, 'paper_id': paper_id_list})
    df.to_csv('./data/test_{}.csv'.format(TEST_USED_RECALL_NUM), index=False)
    end = time.time()
    print('\nuse time:', (end - start) / 60)
