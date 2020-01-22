import pandas as pd
# pd.set_option('display.max_columns',2000)
import numpy as np
from tqdm import tqdm
import gc
import pickle
import lightgbm as lgb
test_temp = pd.read_pickle('../final_dir/test_temp.csv')
print(test_temp.shape)
pred1 = pd.read_csv('../final_dir/the_last_one_bert.csv')
test_temp = test_temp.merge(pred1,on=['description_id','paper_id'],how='left')
pred2 = np.load('../final_dir/lgb_wanzheng_test_5cv.npy')
# test_temp['prd1'] = pred1['']
test_temp['preb2'] = pred2
rank_feature =['preb','preb2']
rank_test = test_temp.groupby('description_id')[rank_feature].rank(ascending=True)

rank_test.columns = [i+'_rank' for i in rank_test.columns]

test_temp= pd.concat([test_temp,rank_test],axis=1)
# test_temp.head()
###
a = test_temp
a['score'] = 4*a['preb_rank']+6*a['preb2_rank']
a = a.sort_values('score',ascending=False)
sub1 = a.groupby('description_id')['paper_id'].apply(list).apply(lambda x:x+['q','v','e']).apply(lambda x:x[0])
sub2 = a.groupby('description_id')['paper_id'].apply(list).apply(lambda x:x+['q','v','e']).apply(lambda x:x[1])
sub3 = a.groupby('description_id')['paper_id'].apply(list).apply(lambda x:x+['q','v','e']).apply(lambda x:x[2])
sub = pd.concat([sub1,sub2,sub3],axis=1).reset_index()
sub.columns=['description_id','pred1','pred2','pred3']
sub.to_csv('../submit/1_16_lgb_bert_final.csv',index=False,header=None)