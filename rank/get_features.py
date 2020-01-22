#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import gc
import  numpy as np
import  pandas as pd
import Levenshtein
from  tqdm import  tqdm
from fuzzywuzzy import fuzz
from gensim.summarization import bm25
from gensim.summarization.bm25 import BM25
from tqdm import tqdm_notebook
import os
import time
import math
from multiprocessing import Process,cpu_count,Manager,Pool
import collections
from sklearn.externals import joblib
from gensim import corpora,similarities,models
from util import *
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import pickle
import warnings
import time
import pickle
tqdm.pandas()
import textdistance
#get_ipython().run_line_magic('matplotlib', 'inline')
from nltk import word_tokenize,pos_tag

from util import pool_extract, pre_process

warnings.filterwarnings('ignore')

"""
大部分相似度特征
"""
tqdm.pandas()

# In[3]:


# 后面加载训练好的w2v模型时也需要有这个类的定义, 否则load会报找不到这个类的错误
class EpochSaver(CallbackAny2Vec):
    '''用于保存模型, 打印损失函数等等'''
    def __init__(self, savedir, save_name="word2vector.model"):
        os.makedirs(savedir, exist_ok=True)
        self.save_path = os.path.join(savedir, save_name)
        self.epoch = 0
        self.pre_loss = 0
        self.best_loss = 999999999.9
        self.since = time.time()

    def on_epoch_end(self, model):
        self.epoch += 1
        cum_loss = model.get_latest_training_loss() # 返回的是从第一个epoch累计的
        epoch_loss = cum_loss - self.pre_loss
        time_taken = time.time() - self.since
        print("Epoch %d, loss: %.2f, time: %dmin %ds" %
                    (self.epoch, epoch_loss, time_taken//60, time_taken%60))
        if self.best_loss > epoch_loss:
            self.best_loss = epoch_loss
            print("Better model. Best loss: %.2f" % self.best_loss)
            model.save(self.save_path)
            print("Model %s save done!" % self.save_path)

        self.pre_loss = cum_loss
        self.since = time.time()
# In[ ]:


##################################features works##########################################################
#n-gram距离
def get_df_grams(train_sample,values,cols):
    def create_ngram_set(input_list, ngram_value=2):
        return set(zip(*[input_list[i:] for i in range(ngram_value)]))

    def get_n_gram(df, values=2):
        train_query = df.values
        train_query = [[word for word in str(sen).replace("'", '').split(' ')] for sen in train_query]
        train_query_n = []
        for input_list in train_query:
            train_query_n_gram = set()
            for value in range(2, values + 1):
                train_query_n_gram = train_query_n_gram | create_ngram_set(input_list, value)
            train_query_n.append(train_query_n_gram)
        return train_query_n

    train_query = get_n_gram(train_sample[cols[0]], values)
    train_title = get_n_gram(train_sample[cols[1]], values)
    sim = list(map(lambda x, y: len(x) + len(y) - 2 * len(x & y),
                       train_query, train_title))
    sim_number_rate=list(map(lambda x, y:   len(x & y)/ len(x)  if len(x)!=0 else 0,
                       train_query, train_title))
    return sim ,sim_number_rate

def make_feature(data_or,vec_model):
    print('get features:')
    from gensim.models import Word2Vec
    vec_model = Word2Vec.load('pretrain_model/w2v_300.model')
    dictionary = corpora.Dictionary.load('temp_data/train_dictionary.dict')
    tfidf = models.TfidfModel.load("temp_data/train_tfidf.model")
    index = similarities.SparseMatrixSimilarity.load('temp_data/train_index.index')
    item_id_list = joblib.load('temp_data/paper_id.pkl')

    with open('temp_data/train_content.pkl','rb') as fr:
        corpus = pickle.load(fr)
    data = data_or.copy()

    data['abstract_pre'] = data['abstract_pre'].apply(
        lambda x: np.nan if str(x) == 'nan' or len(x) < 9 else x)

    data['abstract_pre'] = data['abstract_pre'].apply(
        lambda x: 'none' if str(x) == 'nan' or str(x).split(' ') == ['n', 'o', 'n', 'e'] else x)
    data['key_text_pre'] = data['key_text_pre'].fillna('none')
    data['description_text'] = data['description_text'].fillna('none')
    data['title_pro'] = data['title_pro'].fillna('none')
    data['description_text_pre'] = data['description_text_pre'].fillna('none')
    prefix = 'num_'
    
    # 长度
    data[prefix + 'key_text_len'] = data['key_text_pre'].apply(lambda x: len(x.split(' ')))

    # 长度append
    data[prefix + 'description_text_len'] = data['description_text'].apply(lambda x: len(x.split(' ')))

    data.loc[data[prefix + 'key_text_len'] < 7, 'key_text_pre'] = data[data[prefix + 'key_text_len'] < 7][
        'description_text'].apply(
        lambda x: ' '.join(pre_process(re.sub(r'[\[|,]+\*\*\#\#\*\*[\]|,]+', '', x)))).values

    # abstract是否为空
    data[prefix + 'cate_pa_isnull'] = data['abstract_pre'].apply(lambda x: 1 if str(x) == 'none' else 0)

    # key_words是否为空
    data[prefix + 'cate_pkeywords_isnull'] = data['keywords'].apply(lambda x: 1 if str(x) == 'nan' else 0)


    #描述在key_word中出现的次数
    def get_num_key(x,y):
        if str(y)=='nan':
            return -1
        y=y.strip(';').split(';')
        num=0
        for i in y:
            if i in x:
                num+=1
        return num

    data[prefix+'key_in_key_word_number']=list(map(lambda x,y: get_num_key(x,y),data['key_text_pre'],data['keywords']))
    #描述在key_word中出现的次数/key_words的个数
    data[prefix+'key_in_key_word_number_rate']=list(map(lambda x,y: 0 if x==-1 else x/len(y.strip(';').split(';')),data[prefix+'key_in_key_word_number'],
                                                data['keywords']))

    #append
    data[prefix+'key_in_key_word_number2']=list(map(lambda x,y: get_num_key(x,y),data['description_text'],data['keywords']))
    #描述在key_word中出现的次数/key_words的个数
    data[prefix+'key_in_key_word_number2_rate']=list(map(lambda x,y: 0 if x==-1 else x/len(y.strip(';').split(';')),data[prefix+'key_in_key_word_number2'],
                                                data['keywords']))

    # 描述在title出现单词的统计
    def get_num_common_words_and_ratio(merge, col):
        # merge data
        merge = merge[col]
        merge.columns = ['q1', 'q2']
        merge['q2'] = merge['q2'].apply(lambda x: 'none' if str(x) == 'nan' else x)

        q1_word_set = merge.q1.apply(lambda x: x.split(' ')).apply(set).values
        q2_word_set = merge.q2.apply(lambda x: x.split(' ')).apply(set).values

        q1_word_len = merge.q1.apply(lambda x: len(x.split(' '))).values
        q2_word_len = merge.q2.apply(lambda x: len(x.split(' '))).values

        q1_word_len_set = merge.q1.apply(lambda x: len(set(x.split(' ')))).values
        q2_word_len_set = merge.q2.apply(lambda x: len(set(x.split(' ')))).values

        result = [len(q1_word_set[i] & q2_word_set[i]) for i in range(len(q1_word_set))]
        result_ratio_q = [result[i] / q1_word_len[i] for i in range(len(q1_word_set))]
        result_ratio_t = [result[i] / q2_word_len[i] for i in range(len(q1_word_set))]

        result_ratio_q_set = [result[i] / q1_word_len_set[i] for i in range(len(q1_word_set))]
        result_ratio_t_set = [result[i] / q2_word_len_set[i] for i in range(len(q1_word_set))]

        return result, result_ratio_q, result_ratio_t, q1_word_len, q2_word_len, q1_word_len_set, q2_word_len_set, result_ratio_q_set, result_ratio_t_set

    data[prefix + 'common_words_k_pt'], \
    data[prefix + 'common_words_k_pt_k'], \
    data[prefix + 'common_words_k_pt_pt'], \
    data[prefix + 'k_len'], \
    data[prefix + 'pt_len'], \
    data[prefix + 'k_len_set'], \
    data[prefix + 'pt_len_set'], \
    data[prefix + 'common_words_k_pt_k_set'], \
    data[prefix + 'common_words_k_pt_pt_set'] = get_num_common_words_and_ratio(data, ['key_text_pre', 'title_pro'])

    data[prefix + 'common_words_k_at'], \
    data[prefix + 'common_words_k_at_k'], \
    data[prefix + 'common_words_k_at_at'], \
    data[prefix + 'k_len'], \
    data[prefix + 'at_len'], \
    data[prefix + 'k_len_set'], \
    data[prefix + 'at_len_set'], \
    data[prefix + 'common_words_k_at_k_set'], \
    data[prefix + 'common_words_k_at_at_set'] = get_num_common_words_and_ratio(data, ['key_text_pre', 'abstract_pre'])

    #append
    data[prefix + 'common_words_k_pt_2'], \
    data[prefix + 'common_words_k_pt_k_2'], \
    data[prefix + 'common_words_k_pt_pt_2'], \
    data[prefix + 'k_len_2'], \
    data[prefix + 'pt_len'], \
    data[prefix + 'k_len_set_2'], \
    data[prefix + 'pt_len_set'], \
    data[prefix + 'common_words_k_pt_k_set_2'], \
    data[prefix + 'common_words_k_pt_pt_set_2'] = get_num_common_words_and_ratio(data, ['description_text', 'title_pro'])

    data[prefix + 'common_words_k_at_2'], \
    data[prefix + 'common_words_k_at_k_2'], \
    data[prefix + 'common_words_k_at_at_2'], \
    data[prefix + 'k_len_2'], \
    data[prefix + 'at_len'], \
    data[prefix + 'k_len_set_2'], \
    data[prefix + 'at_len_set'], \
    data[prefix + 'common_words_k_at_k_set_2'], \
    data[prefix + 'common_words_k_at_at_set_2'] = get_num_common_words_and_ratio(data, ['description_text', 'abstract_pre'])



    # Jaccard 相似度
    def jaccard(x, y):
        if str(y) == 'nan':
            y = 'none'
        x = set(x)
        y = set(y)
        return float(len(x & y) / len(x | y))

    data[prefix + 'jaccard_sim_k_pt'] = list(map(lambda x, y: jaccard(x, y), data['key_text_pre'], data['title_pro']))
    data[prefix + 'jaccard_sim_k_pa'] = list(
        map(lambda x, y: jaccard(x, y), data['key_text_pre'], data['abstract_pre']))

    #append
    data[prefix + 'jaccard_sim_k_pt2'] = list(map(lambda x, y: jaccard(x, y), data['description_text'], data['title_pro']))
    data[prefix + 'jaccard_sim_k_pa2'] = list(
        map(lambda x, y: jaccard(x, y), data['key_text_pre'], data['description_text']))

    # 编辑距离
    print('get edict distance:')
    data[prefix + 'edict_distance_k_pt'] = list(
        map(lambda x, y: Levenshtein.distance(x, y) / (len(x)+1), tqdm(data['key_text_pre']), data['title_pro']))
    data[prefix + 'edict_jaro'] = list(
        map(lambda x, y: Levenshtein.jaro(x, y), tqdm(data['key_text_pre']), data['title_pro']))
    data[prefix + 'edict_ratio'] = list(
        map(lambda x, y: Levenshtein.ratio(x, y), tqdm(data['key_text_pre']), data['title_pro']))
    data[prefix + 'edict_jaro_winkler'] = list(
        map(lambda x, y: Levenshtein.jaro_winkler(x, y), tqdm(data['key_text_pre']), data['title_pro']))

    data[prefix + 'edict_distance_k_pa'] = list(
        map(lambda x, y: Levenshtein.distance(x, y) / (len(x)+1), tqdm(data['key_text_pre']),
            data['abstract_pre']))
    data[prefix + 'edict_jaro_pa'] = list(
        map(lambda x, y: Levenshtein.jaro(x, y), tqdm(data['key_text_pre']), data['abstract_pre']))
    data[prefix + 'edict_ratio_pa'] = list(
        map(lambda x, y: Levenshtein.ratio(x, y), tqdm(data['key_text_pre']), data['abstract_pre']))
    data[prefix + 'edict_jaro_winkler_pa'] = list(
        map(lambda x, y: Levenshtein.jaro_winkler(x, y), tqdm(data['key_text_pre']), data['abstract_pre']))

    #append
    print('get edict distance:')
    data[prefix + 'edict_distance_k_pt_2'] = list(
        map(lambda x, y: Levenshtein.distance(x, y) / (len(x)+1), tqdm(data['description_text']), data['title_pro']))
    data[prefix + 'edict_jaro_2'] = list(
        map(lambda x, y: Levenshtein.jaro(x, y), tqdm(data['description_text']), data['title_pro']))
    data[prefix + 'edict_ratio_2'] = list(
        map(lambda x, y: Levenshtein.ratio(x, y), tqdm(data['description_text']), data['title_pro']))
    data[prefix + 'edict_jaro_winkler_2'] = list(
        map(lambda x, y: Levenshtein.jaro_winkler(x, y), tqdm(data['description_text']), data['title_pro']))

    data[prefix + 'edict_distance_k_pa_2'] = list(
        map(lambda x, y: Levenshtein.distance(x, y) / (len(x)+1), tqdm(data['description_text']),
            data['abstract_pre']))
    data[prefix + 'edict_jaro_pa_2'] = list(
        map(lambda x, y: Levenshtein.jaro(x, y), tqdm(data['description_text']), data['abstract_pre']))
    data[prefix + 'edict_ratio_pa_2'] = list(
        map(lambda x, y: Levenshtein.ratio(x, y), tqdm(data['description_text']), data['abstract_pre']))
    data[prefix + 'edict_jaro_winkler_pa_2'] = list(
        map(lambda x, y: Levenshtein.jaro_winkler(x, y), tqdm(data['description_text']), data['abstract_pre']))

    #余弦相似度
    def get_sim(doc, corpus):
        corpus = corpus.split(' ')
        corpus_vec = [dictionary.doc2bow(corpus)]
        corpus_tfidf = tfidf[corpus_vec]
        featurenum = len(dictionary.token2id.keys())
        index_i = similarities.SparseMatrixSimilarity(corpus_tfidf, num_features=featurenum)
        doc = doc.split(' ')
        vec = dictionary.doc2bow(doc)
        vec_tfidf = tfidf[vec]
        sim = index_i.get_similarities(vec_tfidf)
        return sim[0]

    data[prefix + 'sim'] = list(map(lambda x, y: get_sim(x, y), tqdm(data['key_text_pre']), data['title_pro']))
    data[prefix + 'sim_pa'] = list(map(lambda x, y: get_sim(x, y), tqdm(data['key_text_pre']), data['abstract_pre']))

    #append
    data[prefix + 'sim_2'] = list(map(lambda x, y: get_sim(x, y), tqdm(data['description_text']), data['title_pro']))
    data[prefix + 'sim_pa_2'] = list(map(lambda x, y: get_sim(x, y), tqdm(data['description_text']), data['abstract_pre']))

    # tfidf
    def get_simlilary(query, title):
        def get_weight_counter_and_tf_idf(x, y):
            x = x.split()
            y = y.split()
            corups = x + y
            obj = dict(collections.Counter(corups))
            x_weight = []
            y_weight = []
            idfs = []
            for key in obj.keys():
                idf = 1
                w = obj[key]
                if key in x:
                    idf += 1
                    x_weight.append(w)
                else:
                    x_weight.append(0)
                if key in y:
                    idf += 1
                    y_weight.append(w)
                else:
                    y_weight.append(0)
                idfs.append(math.log(3.0 / idf) + 1)
            return [np.array(x_weight), np.array(y_weight), np.array(x_weight) * np.array(idfs),
                    np.array(y_weight) * np.array(idfs), np.array(list(obj.keys()))]

        weight = list(map(lambda x, y: get_weight_counter_and_tf_idf(x, y),
                          tqdm(query), title))
        x_weight_couner = []
        y_weight_couner = []
        x_weight_tfidf = []
        y_weight_tfidf = []
        words = []
        for i in weight:
            x_weight_couner.append(i[0])
            y_weight_couner.append(i[1])
            x_weight_tfidf.append(i[2])
            y_weight_tfidf.append(i[3])
            words.append(i[4])

        # 曼哈顿距离
        def mhd_simlilary(x, y):
            return np.linalg.norm(x - y, ord=1)

        mhd_simlilary_counter = list(map(lambda x, y: mhd_simlilary(x, y),
                                         x_weight_couner, y_weight_couner))
        mhd_simlilary_tfidf = list(map(lambda x, y: mhd_simlilary(x, y),
                                       x_weight_tfidf, y_weight_tfidf))

        # 余弦相似度
        def cos_simlilary(x, y):
            return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

        cos_simlilary_counter = list(map(lambda x, y: cos_simlilary(x, y),
                                         x_weight_couner, y_weight_couner))
        cos_simlilary_tfidf = list(map(lambda x, y: cos_simlilary(x, y),
                                       x_weight_tfidf, y_weight_tfidf))

        # 欧式距离
        def Euclidean_simlilary(x, y):
            return np.sqrt(np.sum(x - y) ** 2)

        Euclidean_simlilary_counter = list(map(lambda x, y: Euclidean_simlilary(x, y),
                                               x_weight_couner, y_weight_couner))
        Euclidean__simlilary_tfidf = list(map(lambda x, y: Euclidean_simlilary(x, y),
                                              x_weight_tfidf, y_weight_tfidf))

        return mhd_simlilary_counter, mhd_simlilary_tfidf, cos_simlilary_counter, \
               cos_simlilary_tfidf, Euclidean_simlilary_counter, Euclidean__simlilary_tfidf

    data[prefix + 'mhd_similiary'], data[prefix + 'tf_mhd_similiary'], \
    data[prefix + 'cos_similiary'], data[prefix + 'tf_cos_similiary'], \
    data[prefix + 'os_similiary'], data[prefix + 'tf_os_similiary'] = get_simlilary(data['key_text_pre'],data['title_pro'])


    data[prefix + 'mhd_similiary_pa'], data[prefix + 'tf_mhd_similiary_pa'], \
    data[prefix + 'cos_similiary_pa'], data[prefix + 'tf_cos_similiary_pa'], \
    data[prefix + 'os_similiary_pa'], data[prefix + 'tf_os_similiary_pa'] = get_simlilary(data['key_text_pre'],data['abstract_pre'])

    '词向量平均的相似度'

    def get_vec(x):
        vec = []
        for word in x.split():
            if word in vec_model:
                vec.append(vec_model[word])
        if len(vec) == 0:
            return np.nan
        else:
            return np.mean(np.array(vec), axis=0)

    data['key_text_pre_vec'] = data['key_text_pre'].progress_apply(lambda x: get_vec(x))
    data['title_pro_vec'] = data['title_pro'].progress_apply(lambda x: get_vec(x))
    data['abstract_pre_vec'] = data['abstract_pre'].progress_apply(lambda x: get_vec(x))
    data['description_text_vec'] = data['description_text'].progress_apply(lambda x: get_vec(x))

    # cos
    data[prefix + 'cos_mean_word2vec'] = list(map(lambda x, y: np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)),
                                                  tqdm(data['key_text_pre_vec']), data['title_pro_vec']))
    data[prefix + 'cos_mean_word2vec'] = data[prefix + 'cos_mean_word2vec'].progress_apply(
        lambda x: np.nan if np.isnan(x).any() else x)

    # 欧式距离
    data[prefix + 'os_mean_word2vec'] = list(map(lambda x, y: np.sqrt(np.sum((x - y) ** 2)),
                                                 tqdm(data['key_text_pre_vec']), data['title_pro_vec']))

    # mhd
    data[prefix + 'mhd_mean_word2vec'] = list(map(lambda x, y: np.nan if np.isnan(x).any() or np.isnan(y).any() else
    np.linalg.norm(x - y, ord=1), tqdm(data['key_text_pre_vec']), data['title_pro_vec']))


    # cos
    data[prefix + 'cos_mean_word2vec_pa'] = list(map(lambda x, y: np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)),
                                                  tqdm(data['key_text_pre_vec']), data['abstract_pre_vec']))
    data[prefix + 'cos_mean_word2vec_pa'] = data[prefix + 'cos_mean_word2vec_pa'].progress_apply(
        lambda x: np.nan if np.isnan(x).any() else x)

    # 欧式距离
    data[prefix + 'os_mean_word2vec_pa'] = list(map(lambda x, y: np.sqrt(np.sum((x - y) ** 2)),
                                                 tqdm(data['key_text_pre_vec']), data['abstract_pre_vec']))

    # mhd
    data[prefix + 'mhd_mean_word2vec_pa'] = list(map(lambda x, y: np.nan if np.isnan(x).any() or np.isnan(y).any() else
    np.linalg.norm(x - y, ord=1), tqdm(data['key_text_pre_vec']), data['abstract_pre_vec']))


    #append
    data[prefix + 'cos_mean_word2vec_2'] = list(map(lambda x, y: np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)),
                                                  tqdm(data['description_text_vec']), data['title_pro_vec']))
    data[prefix + 'cos_mean_word2vec_2'] = data[prefix + 'cos_mean_word2vec_2'].progress_apply(
        lambda x: np.nan if np.isnan(x).any() else x)

    # 欧式距离
    data[prefix + 'os_mean_word2vec_2'] = list(map(lambda x, y: np.sqrt(np.sum((x - y) ** 2)),
                                                 tqdm(data['description_text_vec']), data['title_pro_vec']))

    # mhd
    data[prefix + 'mhd_mean_word2vec_2'] = list(map(lambda x, y: np.nan if np.isnan(x).any() or np.isnan(y).any() else
    np.linalg.norm(x - y, ord=1), tqdm(data['description_text_vec']), data['title_pro_vec']))

    # cos
    data[prefix + 'cos_mean_word2vec_pa2'] = list(map(lambda x, y: np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)),
                                                  tqdm(data['description_text_vec']), data['abstract_pre_vec']))
    data[prefix + 'cos_mean_word2vec_pa2'] = data[prefix + 'cos_mean_word2vec_pa2'].progress_apply(
        lambda x: np.nan if np.isnan(x).any() else x)

    # 欧式距离
    data[prefix + 'os_mean_word2vec_pa2'] = list(map(lambda x, y: np.sqrt(np.sum((x - y) ** 2)),
                                                 tqdm(data['description_text_vec']), data['abstract_pre_vec']))

    # mhd
    data[prefix + 'mhd_mean_word2vec_pa2'] = list(map(lambda x, y: np.nan if np.isnan(x).any() or np.isnan(y).any() else
    np.linalg.norm(x - y, ord=1), tqdm(data['description_text_vec']), data['abstract_pre_vec']))




    #n-gram距离相关
    data[prefix+'n_gram_sim'],data[prefix+'sim_numeber_rate']=get_df_grams(data,2,['key_text_pre','title_pro'])
    data[prefix+'n_gram_sim_pa'],data[prefix+'sim_numeber_rate_pa']=get_df_grams(data,2,['key_text_pre','abstract_pre'])

    #append
    #n-gram距离相关
    data[prefix+'n_gram_sim_2'],data[prefix+'sim_numeber_rate_2']=get_df_grams(data,2,['description_text','title_pro'])
    data[prefix+'n_gram_sim_pa_2'],data[prefix+'sim_numeber_rate_pa_2']=get_df_grams(data,2,['description_text','abstract_pre'])

    
#################################################朋哥已做##################################
#     def apply_fun(df):
#         df.columns = ['d_id', 'key', 'doc']
#         df['d_id'] = df['d_id'].fillna('always_nan')
#         query_id_group = df.groupby(['d_id'])
#         bm_list = []
#         for name, group in tqdm(query_id_group):
#             corpus = group['doc'].values.tolist()
#             corpus = [sentence.strip().split() for sentence in corpus]
#             query = group['key'].values[0].strip().split()
#             bm25Model = BM25(corpus)
#             bmscore = bm25Model.get_scores(query)
#             bm_list.extend(bmscore)

#         return bm_list

#     data[prefix + 'bm25'] = apply_fun(data[['description_id', 'key_text_pre', 'title_pro']])
#     data[prefix + 'bm25_pa'] = apply_fun(data[['description_id', 'key_text_pre', 'abstract_pre']])

#     #append
#     data[prefix + 'bm25_2'] = apply_fun(data[['description_id', 'description_text', 'title_pro']])
#     data[prefix + 'bm25_pa_2'] = apply_fun(data[['description_id', 'description_text', 'abstract_pre']])


#     # get bm25
#     def get_bm25(p_id, query):
#         query = query.split(' ')
#         score = bm25Model.get_score(query, item_id_list.index(p_id))
#         return score

#     data[prefix + 'bm_25_all'] = list(map(lambda x, y: get_bm25(x, y), tqdm(data['paper_id']), data['key_text_pre']))
#     #append
#     data[prefix + 'bm_25_all_2'] = list(map(lambda x, y: get_bm25(x, y), tqdm(data['paper_id']), data['description_text']))
#################################################朋哥已做##################################
    data[prefix + 'Hamming_kt'] = list(map(lambda x, y: 
                                           textdistance.Hamming(qval=None).normalized_distance(x, y),
                                           tqdm(data['key_text_pre']), data['title_pro']))
    data[prefix + 'Hamming_dt'] = list(map(lambda x, y: 
                                           textdistance.Hamming(qval=None).normalized_distance(x, y),
                                           tqdm(data['description_text_pre']), data['title_pro']))
    
    data[prefix + 'Hamming_ka'] = list(map(lambda x, y: 
                                           textdistance.Hamming(qval=None).normalized_distance(x, y),
                                           tqdm(data['key_text_pre']), data['abstract_pre']))
    data[prefix + 'Hamming_da'] = list(map(lambda x, y: 
                                           textdistance.Hamming(qval=None).normalized_distance(x, y),
                                           tqdm(data['description_text_pre']), data['abstract_pre']))
    
    data[prefix + 'Hamming_sim_kt'] = list(map(lambda x, y: 
                                           textdistance.Hamming(qval=None).similarity(x, y),
                                           tqdm(data['key_text_pre']), data['title_pro']))
    data[prefix + 'Hamming_sim_dt'] = list(map(lambda x, y: 
                                           textdistance.Hamming(qval=None).similarity(x, y),
                                           tqdm(data['description_text_pre']), data['title_pro']))
    
    data[prefix + 'Hamming_sim_ka'] = list(map(lambda x, y: 
                                           textdistance.Hamming(qval=None).similarity(x, y),
                                           tqdm(data['key_text_pre']), data['abstract_pre']))
    data[prefix + 'Hamming_sim_da'] = list(map(lambda x, y: 
                                           textdistance.Hamming(qval=None).similarity(x, y),
                                           tqdm(data['description_text_pre']), data['abstract_pre']))
   
    def edit_distance(df,w1, w2):
        word1 = df[w1].split()
        word2 = df[w2].split()
        len1 = len(word1)
        len2 = len(word2)
        dp = np.zeros((len1 + 1, len2 + 1))
        for i in range(len1 + 1):
            dp[i][0] = i
        for j in range(len2 + 1):
            dp[0][j] = j

        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                delta = 0 if word1[i - 1] == word2[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j - 1] + delta, min(dp[i - 1][j] + 1, dp[i][j - 1] + 1))
        return dp[len1][len2]
    
    data[prefix + 'edit_distance_kt'] = data.apply(edit_distance, axis=1, 
                                                   args=('key_text_pre', 'title_pro'))
    data[prefix + 'edit_distance_dt'] = data.apply(edit_distance, axis=1, 
                                                   args=('description_text_pre', 'title_pro'))
    data[prefix + 'edit_distance_ka'] = data.apply(edit_distance, axis=1, 
                                                   args=('key_text_pre', 'abstract_pre'))
    data[prefix + 'edit_distance_da'] = data.apply(edit_distance, axis=1, 
                                                   args=('description_text_pre', 'abstract_pre'))
    
    def get_same_word_features(query, title):
        q_list = query.split()
        t_list = title.split()
        set_query = set(q_list)
        set_title = set(t_list)
        count_words = len(set_query.union(set_title))

        comwords = [word for word in t_list if word in q_list]
        comwords_set = set(comwords)
        unique_rate = len(comwords_set) / count_words

        same_word1 = [w for w in q_list if w in t_list]
        same_word2 = [w for w in t_list if w in q_list]
        same_len_rate = (len(same_word1) + len(same_word2)) / (len(q_list) + len(t_list))
        if len(comwords) > 0:
            com_index1 = len(comwords)
            same_word_q = com_index1 / len(q_list)
            same_word_t = com_index1 / len(t_list)

            for word in comwords_set:
                index_list = [i for i, x in enumerate(q_list) if x == word]
                com_index1 += sum(index_list)
            q_loc = com_index1 / (len(q_list) * len(comwords))
            com_index2 = len(comwords)
            for word in comwords_set:
                index_list = [i for i, x in enumerate(t_list) if x == word]
                com_index2 += sum(index_list)
            t_loc = com_index2 / (len(t_list) * len(comwords))

            same_w_set_q = len(comwords_set) / len(set_query)
            same_w_set_t = len(comwords_set) / len(set_title)
            word_set_rate = 2 * len(comwords_set) / (len(set_query) + len(set_title))

            com_set_query_index = len(comwords_set)
            for word in comwords_set:
                index_list = [i for i, x in enumerate(q_list) if x == word]
                if len(index_list) > 0:
                    com_set_query_index += index_list[0]
            loc_set_q = com_set_query_index / (len(q_list) * len(comwords_set))
            com_set_title_index = len(comwords_set)
            for word in comwords_set:
                index_list = [i for i, x in enumerate(t_list) if x == word]
                if len(index_list) > 0:
                    com_set_title_index += index_list[0]
            loc_set_t = com_set_title_index / (len(t_list) * len(comwords_set))
            set_rate = (len(comwords_set) / len(comwords))
        else:
            unique_rate, same_len_rate, same_word_q, same_word_t, q_loc, t_loc, same_w_set_q, same_w_set_t, word_set_rate, loc_set_q, loc_set_t, set_rate = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        return unique_rate, same_len_rate, same_word_q, same_word_t, q_loc, t_loc, same_w_set_q, same_w_set_t, word_set_rate, loc_set_q, loc_set_t, set_rate
    
    data[prefix+"unique_rate_kt"],data[prefix+"same_len_rate_kt"],data[prefix+"same_word_q_kt"],\
    data[prefix+"same_word_t_kt"],data[prefix+"q_loc_kt"],data[prefix+"t_loc_kt"],data[prefix+"same_w_set_q_kt"],data[prefix+"same_w_set_t_kt"],data[prefix+"word_set_rate_kt"],\
    data[prefix+"loc_set_q_kt"], data[prefix+"loc_set_t_kt"], data[prefix+"set_rate_kt"]= zip(
    *data.apply(lambda line: get_same_word_features(line["key_text_pre"], line["title_pro"]), axis=1))
    
    data[prefix+"unique_rate_dt"],data[prefix+"same_len_rate_dt"],data[prefix+"same_word_q_dt"],\
    data[prefix+"same_word_t_dt"],data[prefix+"q_loc_dt"],data[prefix+"t_loc_dt"],data[prefix+"same_w_set_q_dt"],data[prefix+"same_w_set_t_dt"],data[prefix+"word_set_rate_dt"],\
    data[prefix+"loc_set_q_dt"], data[prefix+"loc_set_t_dt"], data[prefix+"set_rate_dt"]= zip(
    *data.apply(lambda line: get_same_word_features(line["description_text_pre"], line["title_pro"]), axis=1))

    data[prefix+"unique_rate_ka"],data[prefix+"same_len_rate_ka"],data[prefix+"same_word_q_ka"],\
    data[prefix+"same_word_t_ka"],data[prefix+"q_loc_ka"],data[prefix+"t_loc_ka"],data[prefix+"same_w_set_q_ka"],data[prefix+"same_w_set_t_ka"],data[prefix+"word_set_rate_ka"],\
    data[prefix+"loc_set_q_ka"], data[prefix+"loc_set_t_ka"], data[prefix+"set_rate_ka"]= zip(
    *data.apply(lambda line: get_same_word_features(line["key_text_pre"], line["abstract_pre"]), axis=1))
    
    data[prefix+"unique_rate_da"],data[prefix+"same_len_rate_da"],data[prefix+"same_word_q_da"],\
    data[prefix+"same_word_t_da"],data[prefix+"q_loc_da"],data[prefix+"t_loc_da"],data[prefix+"same_w_set_q_da"],data[prefix+"same_w_set_t_da"],data[prefix+"word_set_rate_da"],\
    data[prefix+"loc_set_q_da"], data[prefix+"loc_set_t_da"], data[prefix+"set_rate_da"]= zip(
    *data.apply(lambda line: get_same_word_features(line["description_text_pre"], line["abstract_pre"]), axis=1))

    
    
    def get_df_grams_3(train_sample,values,cols):
        def create_ngram_set(input_list, ngram_value=3):
            return set(zip(*[input_list[i:] for i in range(ngram_value)]))

        def get_n_gram(df, values=3):
            train_query = df.values
            train_query = [[word for word in str(sen).replace("'", '').split(' ')] for sen in train_query]
            train_query_n = []
            for input_list in train_query:
                train_query_n_gram = set()
                for value in range(3, values + 1):
                    train_query_n_gram = train_query_n_gram | create_ngram_set(input_list, value)
                train_query_n.append(train_query_n_gram)
            return train_query_n

        train_query = get_n_gram(train_sample[cols[0]], values)
        train_title = get_n_gram(train_sample[cols[1]], values)
        sim = list(map(lambda x, y: len(x) + len(y) - 2 * len(x & y),
                           train_query, train_title))
        sim_number_rate=list(map(lambda x, y:   len(x & y)/ len(x)  if len(x)!=0 else 0,
                           train_query, train_title))
        return sim ,sim_number_rate
    data[prefix+'3_gram_sim'],data[prefix+'sim_numeber_rate_3']=get_df_grams_3(data,3,['key_text_pre','title_pro'])
    data[prefix+'3_gram_sim_pa'],data[prefix+'sim_numeber_rate_pa_3']=get_df_grams_3(data,3,['key_text_pre','abstract_pre'])

    #append
    #n-gram距离相关
    data[prefix+'3_gram_sim_2'],data[prefix+'sim_numeber_rate_2_3']=get_df_grams_3(data,3,['description_text_pre','title_pro'])
    data[prefix+'3_gram_sim_pa_2'],data[prefix+'sim_numeber_rate_pa_2_3']=get_df_grams_3(data,3,['description_text_pre','abstract_pre'])
    
    
    def get_son_str_feature(query, title):
        q_list = query.split()
        query_len = len(q_list)
        t_list = title.split()
        title_len = len(t_list)
        count1 = np.zeros((query_len + 1, title_len + 1))
        index = np.zeros((query_len + 1, title_len + 1))
        for i in range(1, query_len + 1):
            for j in range(1, title_len + 1):
                if q_list[i - 1] == t_list[j - 1]:
                    count1[i][j] = count1[i - 1][j - 1] + 1
                    index[i][j] = index[i - 1][j - 1] + j
                else:
                    count1[i][j] = 0
                    index[i][j] = 0
        max_count1 = count1.max()

        if max_count1 != 0:
            row = int(np.where(count1 == np.max(count1))[0][0])
            col = int(np.where(count1 == np.max(count1))[1][0])
            mean_pos = index[row][col] / (max_count1 * title_len)
            begin_loc = (col - max_count1 + 1) / title_len
            rows = np.where(count1 != 0.0)[0]
            cols = np.where(count1 != 0.0)[1]
            total_loc = 0
            for i in range(0, len(rows)):
                total_loc += index[rows[i]][cols[i]]
            density = total_loc / (query_len * title_len)
            rate_q_len = max_count1 / query_len
            rate_t_len = max_count1 / title_len
        else:
            begin_loc, mean_pos, total_loc, density, rate_q_len, rate_t_len = 0, 0, 0, 0, 0, 0
        return max_count1, begin_loc, mean_pos, total_loc, density, rate_q_len, rate_t_len    

    data[prefix+"long_same_max_count1_kt"], data[prefix+"long_same_local_begin_kt"], data[prefix+"long_same_local_mean_kt"],data[prefix+"long_same_total_loc_kt"],\
    data[prefix+"long_same_density_kt"], data[prefix+"long_same_rate_q_len_kt"], data[prefix+"long_same_rate_t_len_kt"]= zip(
        *data.apply(lambda line: get_son_str_feature(line["key_text_pre"], line["title_pro"]), axis=1))
    
    data[prefix+"long_same_max_count1_dt"], data[prefix+"long_same_local_begin_dt"], data[prefix+"long_same_local_mean_dt"],data[prefix+"long_same_total_loc_dt"],\
    data[prefix+"long_same_density_dt"], data[prefix+"long_same_rate_q_len_dt"], data[prefix+"long_same_rate_t_len_dt"]= zip(
        *data.apply(lambda line: get_son_str_feature(line["description_text_pre"], line["title_pro"]), axis=1))
    
    data[prefix+"long_same_max_count1_da"], data[prefix+"long_same_local_begin_da"], data[prefix+"long_same_local_mean_da"],data[prefix+"long_same_total_loc_da"],\
    data[prefix+"long_same_density_da"], data[prefix+"long_same_rate_q_len_da"], data[prefix+"long_same_rate_t_len_da"]= zip(
        *data.apply(lambda line: get_son_str_feature(line["description_text_pre"], line["abstract_pre"]), axis=1))
    
    data[prefix+"long_same_max_count1_ka"], data[prefix+"long_same_local_begin_ka"], data[prefix+"long_same_local_mean_ka"],data[prefix+"long_same_total_loc_ka"],\
    data[prefix+"long_same_density_ka"], data[prefix+"long_same_rate_q_len_ka"], data[prefix+"long_same_rate_t_len_ka"]= zip(
        *data.apply(lambda line: get_son_str_feature(line["key_text_pre"], line["abstract_pre"]), axis=1))
    
    def q_t_common_words(query, title):
        query = set(query.split(' '))
        title = set(title.split(' '))
        return len(query & title)
    
    data[prefix+'common_words_kt'] = data.apply(lambda index: q_t_common_words(index.key_text_pre, index.title_pro), axis=1)
    data[prefix+'common_words_dt'] = data.apply(lambda index: q_t_common_words(index.description_text_pre, index.title_pro), axis=1)
    data[prefix+'common_words_ka'] = data.apply(lambda index: q_t_common_words(index.key_text_pre, index.abstract_pre), axis=1)
    data[prefix+'common_words_da'] = data.apply(lambda index: q_t_common_words(index.description_text_pre, index.abstract_pre), axis=1)

    
    data['key_text_len'] = data['key_text_pre'].apply(lambda x: len(x.split(' ')))
    data['description_text_pre_len'] = data['description_text_pre'].apply(lambda x: len(x.split(' ')))
    data['title_pro_len'] = data['title_pro'].apply(lambda x: len(x.split(' ')))
    data['abstract_pre_len'] = data['abstract_pre'].apply(lambda x: len(x.split(' ')))
    
    
    data[prefix+'common_words_kt_rate_k'] = data[prefix+'common_words_kt'] / data['key_text_len']
    data[prefix+'common_words_kt_rate_t'] = data[prefix+'common_words_kt'] / data['title_pro_len']

    data[prefix+'common_words_dt_rate_d'] = data[prefix+'common_words_dt'] / data['description_text_pre_len']
    data[prefix+'common_words_dt_rate_t'] = data[prefix+'common_words_dt'] / data['title_pro_len']

    data[prefix+'common_words_ka_rate_k'] = data[prefix+'common_words_ka'] / data['key_text_len']
    data[prefix+'common_words_ka_rate_a'] = data[prefix+'common_words_ka'] / data['abstract_pre_len']

    data[prefix+'common_words_da_rate_d'] = data[prefix+'common_words_da'] / data['description_text_pre_len']
    data[prefix+'common_words_da_rate_a'] = data[prefix+'common_words_da'] / data['abstract_pre_len']

    
    
    
    
    feat = ['description_id','paper_id']
    for col in data.columns:
        if re.match('num_', col) != None:
            feat.append(col)

    data = data[feat]

    return data

def make_feature2(train_all,tft):

    print(train_all.isnull().sum())
    train_all['abstract_pre'] = train_all['abstract_pre'].apply(
            lambda x: np.nan if str(x) == 'nan' or len(x) < 9 else x)
    train_all[ 'title_pro'] = train_all[ 'title_pro'].fillna('none')
    train_all['abstract_pre'] = train_all['abstract_pre'].apply(
            lambda x: 'none' if str(x) == 'nan' or str(x).split(' ') == ['n', 'o', 'n', 'e'] else x)

    train_all['paper_content_pre'] = train_all['title_pro'].values + ' ' + train_all['abstract_pre'].values + ' ' + train_all[
            'keywords'].apply(lambda x: ' '.join(x.split(';') if str(x) != 'nan' else 'none')).values

    # 长度
    train_all['key_text_pre'].fillna('none',inplace=True)
    train_all['key_text_len'] = train_all['key_text_pre'].apply(lambda x: len(x.split(' ')))

    # 长度append
    train_all[ 'description_text_pre'] = train_all[ 'description_text_pre'].fillna('none')
    train_all[ 'description_text'] = train_all[ 'description_text'].fillna('none')
    train_all[ 'description_text_pre_len'] = train_all['description_text_pre'].apply(lambda x: len(x.split(' ')))

    train_all.loc[train_all[ 'key_text_len'] < 7, 'key_text_pre'] = train_all[train_all[ 'key_text_len'] < 7][
        'description_text'].apply(
        lambda x: ' '.join(pre_process(re.sub(r'[\[|,]+\*\*\#\#\*\*[\]|,]+', '', x)))).values





    def get_tf_sim(train_query_tf,train_title_tf):
        # 余弦
        v_num = np.array(train_query_tf.multiply(train_title_tf).sum(axis=1))[:, 0]
        v_den = np.array(np.sqrt(train_query_tf.multiply(train_query_tf).sum(axis=1)))[:, 0] * np.array(
                np.sqrt(train_title_tf.multiply(train_title_tf).sum(axis=1)))[:, 0]
        v_num[np.where(v_den == 0)] = 1
        v_den[np.where(v_den == 0)] = 1
        v_score1 = 1 - v_num / v_den

        # 欧式
        v_score = train_query_tf - train_title_tf
        v_score2 = np.sqrt(np.array(v_score.multiply(v_score).sum(axis=1))[:, 0])

        # 曼哈顿
        v_score = np.abs(train_query_tf - train_title_tf)
        v_score3 = v_score.sum(axis=1)

        return  v_score1,v_score2,v_score3




    features = train_all[['description_id','paper_id']]
    train_query_tf = tft.transform(train_all['key_text_pre'].values)
    train_query_tf2 = tft.transform(train_all['description_text_pre'].values)
    train_title_tf = tft.transform(train_all['title_pro'].values)
    train_title_tf2 = tft.transform(train_all['abstract_pre'].values)

    features['tfidf_cos'],features['tfidf_os'] ,features['tfidf_mhd']=get_tf_sim(train_query_tf,train_title_tf)
    features['tfidf_cos2'],features['tfidf_os2'] ,features['tfidf_mhd2']=get_tf_sim(train_query_tf,train_title_tf2)

    features['tfidf_cos_2'],features['tfidf_os_2'] ,features['tfidf_mhd_2']=get_tf_sim(train_query_tf2,train_title_tf)
    features['tfidf_cos2_2'],features['tfidf_os2_2'] ,features['tfidf_mhd2_2']=get_tf_sim(train_query_tf2,train_title_tf2)


    del  train_title_tf,train_query_tf,train_query_tf2,train_title_tf2
    gc.collect()


    #tfidf match share
    print('get tfidf match share:')

    def get_weight(count, eps=100, min_count=2):
        if count < min_count:
            return 0
        else:
            return 1 / (count + eps)


    def load_weight(data):
        words = [x for y in data for x in y.split()]
        counts = collections.Counter(words)
        weights = {word: get_weight(count) for word, count in counts.items()}
        del counts
        del words
        del data
        gc.collect()
        return weights


    def tfidf_match_share(queries, titles, weights):
        ret = []
        for i in tqdm(range(len(queries))):
            q, t = queries[i].split(), titles[i].split()
            q1words = {}
            q2words = {}
            for word in q:
                q1words[word] = 1
            for word in t:
                q2words[word] = 1
            if len(q1words) == 0 or len(q2words) == 0:
                # The computer-generated chaff includes a few questions that are nothing but stopwords
                R = 0
            else:
                shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + \
                                 [weights.get(w, 0) for w in q2words.keys() if w in q1words]
                total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]

                R = np.sum(shared_weights) / np.sum(total_weights)
            ret.append(R)
        return ret

    with open('temp_data/train_content.pkl','rb') as fr:
        content = pickle.load(fr)

    words = [x for y in content for x in y]
    counts = collections.Counter(words)
    del content
    gc.collect()

    weights = {word: get_weight(count) for word, count in counts.items()}
    del counts
    gc.collect()

    features['tfidf_match_share'] = tfidf_match_share(train_all['key_text_pre'].values, train_all['title_pro'].values, weights)
    features['tfidf_match_share_pa'] = tfidf_match_share(train_all['key_text_pre'].values, train_all['abstract_pre'].values, weights)

    features['tfidf_match_share_2'] = tfidf_match_share(train_all['description_text_pre'].values, train_all['title_pro'].values, weights)
    features['tfidf_match_share_pa_2'] = tfidf_match_share(train_all['description_text_pre'].values, train_all['abstract_pre'].values, weights)

    features.columns=['num_'+col for col in list(features.columns)]
    features = features.rename(columns={'num_description_id':'description_id','num_paper_id':'paper_id'})
    #features.to_csv('feat/test_data_merge_bm25_tfidf_20_featall_tfidf2.csv')
    return features





if __name__=='__main__':
    path = 'train_set/'


    word2vec_path = 'pretrain_model/w2v.model'
    vec_model = Word2Vec.load(word2vec_path)
    t1 = time.time()
    
#     test_data = pd.read_csv(path + 'test_data_merge_bm25_tfidf_20.csv')
#     test_feat=pool_extract(test_data,make_feature,vec_model,20000,15)
#     test_feat.to_csv('feat/test_data_merge_bm25_tfidf_20_feat_new.csv', index=False)
#     del test_data,test_feat
#     gc.collect()

    train_data = pd.read_csv(path + 'train_data_merge_bm25_tfidf_20.csv')
        
    train_feat=pool_extract(train_data,make_feature,vec_model,15000,15)
    train_feat.to_csv('feat/train_data_merge_bm25_tfidf_20_feat_new.csv',index=False)

    print('success')
    t2 = time.time()
    print((t2 - t1) / 60)