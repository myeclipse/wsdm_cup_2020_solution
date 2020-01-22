import pandas as pd
from  tqdm import  tqdm
from gensim import corpora,similarities,models
import pandas as pd
import pickle
from util import pre_process
import os

path_temp = '../final_data/'

papers = pd.read_csv('../data/candidate_paper.csv')
papers=papers[papers['paper_id'].notnull()]
papers['abstract'] = papers['abstract'].fillna('none')
papers['title'] = papers['title'].fillna('none')
papers['keywords'] = papers['keywords'].fillna('none')


train=papers['title'].values+' '+papers['abstract'].values+' '+papers['keywords'].apply(lambda x: x.replace(';',' ')).values
train_item_id=list(papers['paper_id'].values)

with open(path_temp+'paper_id.pkl', 'wb') as fw:
    pickle.dump(train_item_id,fw)


if not os.path.exists(path_temp+'train_content.pkl'):
    with open(path_temp+'train_content.pkl','wb') as fw:
        train = list(map(lambda x: pre_process(x), tqdm(train)))
        pickle.dump(train,fw)
else:
    with open(path_temp+'train_content.pkl','rb') as fr:
        train = pickle.load(fr)


dictionary = corpora.Dictionary(train)
corpus = [dictionary.doc2bow(text) for text in train]

# corpus是一个返回bow向量的迭代器。下面代码将完成对corpus中出现的每一个特征的IDF值的统计工作
tfidf_model = models.TfidfModel(corpus, dictionary=dictionary)
corpus_tfidf = tfidf_model[corpus]

dictionary.save(path_temp+'train_dictionary.dict')  # 保存生成的词典
tfidf_model.save(path_temp+'train_tfidf.model')
corpora.MmCorpus.serialize(path_temp+'train_corpuse.mm', corpus)
featurenum = len(dictionary.token2id.keys())  # 通过token2id得到特征数
# 稀疏矩阵相似度，从而建立索引,我们用待检索的文档向量初始化一个相似度计算的对象
index = similarities.SparseMatrixSimilarity(corpus_tfidf, num_features=featurenum)    #这是文档的index
index.save(path_temp+'train_index.index')
