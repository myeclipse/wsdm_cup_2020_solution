#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import torch
import re
import transformers
import os
import random
import time
os.environ['CUDA_VISIBLE_DEVICES']='2,3,4,5'
from multiprocessing import Process,cpu_count,Manager,Pool

from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import Dataset, Subset, DataLoader

from transformers import *


# In[2]:


MODELS = [(BertModel,  BertTokenizer,  'bert-base-uncased')]
for model_class, tokenizer_class in MODELS:
    # Load pretrained model/tokenizer
    tokenizer = tokenizer_class.from_pretrained('../digg/scibert_scivocab_uncased')
    model = model_class.from_pretrained('../digg/scibert_scivocab_uncased')


# In[3]:


class BertTrans(nn.Module):
    def __init__(self,model):
        super(BertTrans, self).__init__()

        self.bert = model
        self.linear_origin = nn.Linear(768, 1)
    def forward(self, input_ids_1=None):

        origin_data = self.bert(input_ids_1)[0]
        output = self.linear_origin(origin_data[:,0,:])
        logits = torch.sigmoid(output)
        return logits


# In[4]:


max_seq_length = 300


# In[5]:


recall_train = pd.read_csv('./data/train_30.csv')

recall_train.columns = ['description_id', 'paper_id', 'label', 'is_added_to_recall_data']

train_data = pd.read_csv('./data/train_pre.csv')
#test_data = pd.read_csv('./data/test_pre.csv')
canditate = pd.read_csv('./data/candidate_paper_pre.csv')

train = pd.merge(recall_train,train_data[['description_id',  'description_text', 
                                          'key_text','key_text_pre', 
                                          'description_text_pre']],on='description_id',how='left')

train = pd.merge(train,canditate,on='paper_id',how='left')


# In[6]:


def convert_data(data, max_seq_length_a=100, max_seq_length_b=200, tokenizer=None):
    all_tokens = []
    longer = 0
    for row in data.itertuples():
        tokens_a = tokenizer.tokenize(getattr(row, "key_text_pre"))
        tokens_b = tokenizer.tokenize(getattr(row, "candi_text"))
        if len(tokens_a)>max_seq_length_a:
            tokens_a = tokens_a[:max_seq_length_a]
            longer += 1
        if len(tokens_a)<max_seq_length_a:
            tokens_a = tokens_a+[0] * (max_seq_length_a - len(tokens_a))
        if len(tokens_b)>max_seq_length_b:
            tokens_b = tokens_b[:max_seq_length_b]
            longer += 1
        if len(tokens_b)<max_seq_length_b:
            tokens_b = tokens_b+[0] * (max_seq_length_b - len(tokens_b))
        one_token = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens_a+["[SEP]"]+tokens_b+["[SEP]"])
        all_tokens.append(one_token)
    data['bert_token'] = all_tokens
    return data


# In[7]:


def pool_extract(data, f ,chunk_size, max_seq_length,tokenizer, worker=4):
    cpu_worker = cpu_count()
    print('cpu core:{}'.format(cpu_worker))
    if worker == -1 or worker > cpu_worker:
        worker = cpu_worker
    print('use cpu:{}'.format(worker))
    t1 = time.time()
    len_data = len(data)
    start = 0
    end = 0
    p = Pool(worker)
    res = []  # 保存的每个进程的返回值
    while end < len_data:
        end = start + chunk_size
        if end > len_data:
            end = len_data
        rslt = p.apply_async(f, (data[start:end],100,200,tokenizer))
        start = end
        res.append(rslt)
    p.close()
    p.join()
    for tmp in [i.get() for i in res]:
        print (tmp.shape) 
    t2 = time.time()
    print((t2 - t1)/60)
    results = pd.concat([i.get() for i in res], axis=0, ignore_index=True)
    return results



train['candi_text'] = train['title_pro'].fillna('')+train['abstract_pre'].fillna('')
train['key_text_pre'] = train['key_text_pre'].fillna('')
train = pool_extract(train,convert_data,50000,max_seq_length,tokenizer,15)


import pickle
pickle.dump(train[['bert_token','label']],open('train_set/bert_data_new.pkl','wb'))
#train = pickle.load(open('train_set/bert_data_new.pkl','rb'))

x_torch = torch.tensor(train['bert_token'].values.tolist(), dtype=torch.long)#.cuda()
y_train_torch = torch.tensor(train['label'][:, np.newaxis],
                             dtype=torch.float32)#.cuda()


# In[9]:


class MyDataset(Dataset):
    def __init__(self, data1,labels):
        self.data1= data1
        self.labels = labels  # 我的例子中label是一样的，如果你的不同，再增加一个即可

    def __getitem__(self, index):    
        img1,target = self.data1[index], self.labels[index]
        return img1,target

    def __len__(self):
        return len(self.data1) # 我的例子中len(self.data1) = len(self.data2)

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# In[10]:


batch_size = 96
n_epochs=4
loss_fn = torch.nn.BCELoss()


# In[22]:


train_dataset = MyDataset(x_torch, y_train_torch)


# In[23]:


from apex.parallel import DistributedDataParallel as DDP
from apex.fp16_utils import *
from apex import amp, optimizers
from apex.multi_tensor_apply import multi_tensor_applier
#model = BertTrans(model)
model = torch.load('./model_bert_temp_300_final.pkl_0').module
#model.load_state_dict(load_model)
model.cuda()
n_gpu=4

param_optimizer = list(model.named_parameters())
no_decay = [ 'LayerNorm.bias', 'LayerNorm.weight','bias']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=0.001, correct_bias=True)

###########################################################################
#add apex
amp.register_float_function(torch, 'sigmoid')
model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
model = torch.nn.DataParallel(model)
total_steps = (x_torch.shape[0]/768*n_epochs/batch_size )
scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100,
                                                      num_training_steps=total_steps)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
#all_test_preds = []
checkpoint_weights = [2 ** epoch for epoch in range(n_epochs)]
accumulation_steps = 3


# In[ ]:


for epoch in range(n_epochs):
    start_time = time.time()

    scheduler.step()

    model.train()
    avg_loss = 0.
    optimizer.zero_grad()
    count = 0
    for data in tqdm(train_loader, disable=False):
        x_batch = data[:-1][0].cuda()
        y_batch = data[-1].cuda()
        y_pred = model(x_batch)
        loss = loss_fn(y_pred, y_batch)
        #loss.backward()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()
        model.zero_grad()
        avg_loss += loss.item() / len(train_loader)
        #each_loss = loss.item()/((count+1)*batch_size)
        count = count+1
#         model.eval()
        #print('loss={:.4f}'.format( each_loss),flush=True)
    elapsed_time = time.time() - start_time
    print('Epoch {}/{} \t loss={:.4f} \t time={:.2f}s'.format(
          epoch + 1, n_epochs, avg_loss, elapsed_time))
    torch.save(model, './model_bert_temp_300_final.pkl_'+str(epoch))


# In[11]:


model = torch.load('./model_bert_temp_300.pkl_1')

model.cuda()

model.eval()

path = '../final_data/'
test_2 = pd.read_csv(path + 'test_data_pre.csv')

test['candi_text'] = test['title_pro'].fillna('')+test['abstract_pre'].fillna('')
test['key_text_pre'] = test['key_text_pre'].fillna('')
test = pool_extract(test,convert_data,50000,max_seq_length,tokenizer,15)


import pickle
pickle.dump(test,open('../final_data/bert_data_test2.pkl','wb'))
test_final = pickle.load(open('../final_data/bert_data_test_final.pkl','rb'))
# test_before = pickle.load(open('train_set/bert_data_test2.pkl','rb'))



class MyDataset_test(Dataset):
    def __init__(self, data1):
        self.data1= data1

    def __getitem__(self, index):    
        img1 = self.data1[index]
        return img1

    def __len__(self):
        return len(self.data1) # 我的例子中len(self.data1) = len(self.data2)


test_final = test_final.reset_index(drop=True)



test_torch = torch.tensor(test_final['bert_token'].values.tolist(), dtype=torch.long)#.cuda()
test_torch_dataset = MyDataset_test(test_torch)
test_loader = torch.utils.data.DataLoader(test_torch_dataset, batch_size=batch_size, shuffle=False)


# In[21]:


test_preds = np.zeros((test_final.shape[0], 1))
for i, x_batch in enumerate(test_loader):
    if i%1000==0:
        print (i)
    y_pred = model(x_batch.cuda()).detach().cpu().numpy()
    test_preds[i * batch_size:(i+1) * batch_size, :] = y_pred


# In[22]:


test_final['preb']=test_preds

test_final[['description_id','paper_id','preb']].to_csv('pred/pred_scibert3_epoch_1.csv_update',index=False)


# In[23]:


pred_final = test_final[['description_id','paper_id','preb']]
pred_final = pd.concat([pred_final,pd.read_csv('pred/pred_scibert3_epoch_1.csv')])


# In[25]:


test_use_tag = pickle.load(open('train_set/bert_data_test_final.pkl','rb'))[['description_id','paper_id']]


# In[26]:


test_use_tag.shape


# In[27]:


pred_final = pd.merge(test_use_tag,pred_final,on=['description_id','paper_id'],how='left')


# In[31]:


pred_final.to_csv('pred/the_last_one_bert.csv',index=False)





