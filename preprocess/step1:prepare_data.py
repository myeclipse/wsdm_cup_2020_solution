#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re
import time
from tqdm import tqdm
from util import pre_process
import pickle
from nltk import word_tokenize,pos_tag

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

path='../data/'


# In[2]:


train=pd.read_csv(path+'train_release.csv')
# test=pd.read_csv(path+'validation.csv')
canditate = pd.read_csv(path+'candidate_paper_for_wsdm2020.csv')
train=train[~train['description_id'].isnull()]
test2 = pd.read_csv('./new_data/test.csv')



def digest(text):
    backup = text[:]
    text = text.replace('al.', '').split('. ')
    t=''
    pre_text=[]
    len_text=len(text)-1
    add=True
    pre=''
    while len_text>=0:
        index=text[len_text]
        index+=pre
        if len(index.split(' '))<=3 :
            add=False
            pre=index+pre
        else:
            add=True
            pre=''
        if add:
            pre_text.append(index)
        len_text-=1
    if len(pre_text)==0:
        pre_text=text
    pre_text.reverse()
    for index in pre_text:
        if index.find('[**##**]') != -1:
            index = re.sub(r'[\[|,]+\*\*\#\#\*\*[\]|,]+','',index)
            index+='. '
            t+=index
    return t


# In[5]:


test2['key_text']=test2['description_text'].apply(lambda x:digest(x))
test2['key_text_pre']=test2['key_text'].progress_apply(lambda x:' '.join(pre_process(x)))
test2['description_text_pre']=test2['description_text'].progress_apply(lambda x:' '.join(pre_process(x)))
test2.to_csv('../final_data/test2_pre.csv',index=False)


# In[ ]:


train['key_text']=train['description_text'].apply(lambda x:digest(x))
test['key_text']=test['description_text'].apply(lambda x:digest(x))
train['key_text_pre']=train['key_text'].progress_apply(lambda x:' '.join(pre_process(x)))
test['key_text_pre']=test['key_text'].progress_apply(lambda x:' '.join(pre_process(x)))


train['description_text_pre']=train['description_text'].progress_apply(lambda x:' '.join(pre_process(x)))
test['description_text_pre']=test['description_text'].progress_apply(lambda x:' '.join(pre_process(x)))

train.to_csv('../final_data/train_pre.csv',index=False)
test.to_csv('../final_data/test_pre.csv',index=False)

canditate['title_pro']=canditate['title'].progress_apply(lambda x:' '.join(pre_process(x)))
canditate['abstract_pre']=canditate['abstract'].progress_apply(lambda x:' '.join(pre_process(x) if str(x)!='nan' else 'none'))

canditate.to_csv('../final_data/candidate_paper_pre.csv',index=False)


# In[ ]:




