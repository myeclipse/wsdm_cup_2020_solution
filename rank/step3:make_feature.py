#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer


# In[2]:


recall_train = pd.read_csv('../final_data/train_50.csv')

recall_train.columns = ['description_id', 'paper_id', 'label', 'is_added_to_recall_data']

train_data = pd.read_csv('../final_data/train_pre.csv')

canditate = pd.read_csv('../final_data/candidate_paper_pre.csv')


# In[3]:


recall_test = pd.read_csv('../final_data/test2_50.csv')
test_data = pd.read_csv('../final_data/test2_pre.csv')
canditate = pd.read_csv('../final_data/candidate_paper_pre.csv')

recall_test.columns=['description_id','paper_id']


# In[6]:


train = pd.merge(recall_train,train_data[['description_id',  'description_text', 
                                          'key_text','key_text_pre', 
                                          'description_text_pre']],on='description_id',how='left')

train = pd.merge(train,canditate,on='paper_id',how='left')



test = pd.merge(recall_test,test_data[['description_id',  'description_text', 
                                          'key_text','key_text_pre', 
                                          'description_text_pre']],on='description_id',how='left')

test = pd.merge(test,canditate,on='paper_id',how='left')

import os
os.makedir('../pretrain_model')

corpus = []
train_description = train['description_text_pre'].dropna().drop_duplicates().apply(lambda x:x.split(' ')).values.tolist()
test_description = test['description_text_pre'].dropna().drop_duplicates().apply(lambda x:x.split(' ')).values.tolist()
corpus = corpus+train_description
corpus = corpus+test_description
train_title = train['title_pro'].dropna().drop_duplicates().apply(lambda x:x.split(' ')).values.tolist()
test_title = test['title_pro'].dropna().drop_duplicates().apply(lambda x:x.split(' ')).values.tolist()
corpus = corpus+train_title
corpus = corpus+test_title
train_abstract = train['abstract_pre'].dropna().drop_duplicates().apply(lambda x:x.split(' ')).values.tolist()
test_abstract = test['abstract_pre'].dropna().drop_duplicates().apply(lambda x:x.split(' ')).values.tolist()
corpus = corpus+train_abstract
corpus = corpus+test_abstract


from gensim.models import Word2Vec

w2v_model = Word2Vec(corpus, size=300, window=8, min_count=0, workers=20, sg=1,iter=9)
w2v_model.save('pretrain_model/w2v_300.model')
w2v_model = Word2Vec.load('pretrain_model/w2v_300.model')


from gensim.models import FastText

ft_model = FastText(corpus, size=300, window=4, min_count=0, workers=20,iter=9)
ft_model.save('pretrain_model/ft_300.model')
ft_model = FastText.load('pretrain_model/ft_300.model')

corpus = []
train_description = train['description_text_pre'].dropna().drop_duplicates().values.tolist()
test_description = test['description_text_pre'].dropna().drop_duplicates().values.tolist()
corpus = corpus+train_description
corpus = corpus+test_description
train_title = train['title_pro'].dropna().drop_duplicates().values.tolist()
test_title = test['title_pro'].dropna().drop_duplicates().values.tolist()
corpus = corpus+train_title
corpus = corpus+test_title
train_abstract = train['abstract_pre'].dropna().drop_duplicates().values.tolist()
test_abstract = test['abstract_pre'].dropna().drop_duplicates().values.tolist()
corpus = corpus+train_abstract
corpus = corpus+test_abstract


tfidf2 = TfidfVectorizer()

tfidf2.fit(corpus)

tfidf_path='pretrain_model/sklearn_train_tfidf.model'
with open(tfidf_path,'wb') as handle:
    pickle.dump(tfidf2,handle)
    
tfidf_path='pretrain_model/sklearn_train_tfidf.model'
with open(tfidf_path,'rb') as handle:
    tft=pickle.load(handle)
print(tft)


from gensim import corpora,similarities,models
import os
candi_item_id=list(canditate['paper_id'].values)

#保存论文的id
with open('../final_data/paper_id.pkl', 'wb') as fw:
    pickle.dump(candi_item_id,fw)
    
    
canditate_data = canditate['title_pro'].fillna('').values+' '+canditate['abstract_pre'].fillna('').values+' '+canditate['keywords'].fillna('').apply(lambda x: x.replace(';',' ').lower()).values

if not os.path.exists('../final_data/train_content.pkl'):
    with open('../final_data/train_content.pkl','wb') as fw:
        canditate_data = [text.split(' ') for text in canditate_data]
        pickle.dump(canditate_data,fw)
else:
    with open('temp_data/train_content.pkl','rb') as fr:
        canditate_data = pickle.load(fr)
        
dictionary = corpora.Dictionary(canditate_data)
corpus = [dictionary.doc2bow(text) for text in canditate_data]
tfidf_model = models.TfidfModel(corpus, dictionary=dictionary)
corpus_tfidf = tfidf_model[corpus]

dictionary.save('../final_data/train_dictionary.dict')  # 保存生成的词典
tfidf_model.save('../final_data/train_tfidf.model')
corpora.MmCorpus.serialize('../final_data/train_corpuse.mm', corpus)
featurenum = len(dictionary.token2id.keys())  # 通过token2id得到特征数
# 稀疏矩阵相似度，从而建立索引,我们用待检索的文档向量初始化一个相似度计算的对象
index = similarities.SparseMatrixSimilarity(corpus_tfidf, num_features=featurenum)    #这是文档的index
index.save('../final_data/train_index.index')


from util import pool_extract
from get_features import *


for i in range(test.shape[0]//300000+1):
    print (i)
    test_temp = test.iloc[i*300000:(i+1)*300000,:]
    test_temp['title_pro'] = test_temp['title_pro'].fillna('none')
    test_feat=pool_extract(test_temp,make_feature,1,20000,15)
    test_feat.to_csv('../final_data/test2_data_feature1.csv_'+str(i), index=False)
    test_feat=make_feature2(test_temp,tft)
    test_feat.to_csv('../final_data/test2_data_feature2.csv_'+str(i), index=False)
    import gc
    del test_temp
    gc.collect()

test_feat=pool_extract(recall_test,make_feature,1,20000,15)
test_feat.to_csv('../final_data/test2_data_feature1.csv', index=False)

test_feat=make_feature2(recall_test,tft)
test_feat.to_csv('../final_data/test2_data_feature2.csv', index=False)

for i in range(train.shape[0]//300000):
    print (i)
    train_temp = train.iloc[i*300000:(i+1)*300000,:]
    train_temp['title_pro'] = train_temp['title_pro'].fillna('none')
    train_feat=pool_extract(train_temp,make_feature,1,20000,15)
    train_feat.to_csv('../final_data/train_data_feature1.csv_'+str(i), index=False)
    train_feat=make_feature2(train_temp,tft)
    train_feat.to_csv('../final_data/train_data_feature2.csv_'+str(i), index=False)
    import gc
    del train_temp
    gc.collect()



