import pandas as pd
# pd.set_option('display.max_columns',2000)
import numpy as np
from tqdm import tqdm
import lightgbm as lgb
import gc
import pickle

path = '../feat_final/'
train_our = pd.read_pickle(path+'train_feature.pkl')
test_our = pd.read_pickle(path+'test_feature.pkl')
print(train_our.shape,test_our.shape)

path = '../final_data_dir/'
train_30 = pd.read_csv(path+'train_30.csv')
print(train_30.shape)
train_30.columns = ['description_id', 'paper_id', 'label', 'is_added_to_recall_data']
test_30 = pd.read_csv(path+'test_30.csv')
test_30.columns = ['description_id', 'paper_id']
print(test_30.shape)

### data path of train matrix
train_feats = pd.read_pickle(path+'train_feature_matrix_30.pkl')
#train_feats = pickle.load(open(path+'train_feature_matrix_30.pkl','rb'))
print(train_feats.shape)
###data path of test matrix
test_feats = pd.read_pickle(path+'test_feature_matrix_30.pkl')
print(test_feats.shape)

for i in tqdm(range(120)):
    train_30['feat_'+str(i)] = train_feats[:,i]
    test_30['feat_'+str(i)] = test_feats[:,i]
print(train_30.shape,test_30.shape)
train_30 =train_30.merge(train_our,on=['description_id','paper_id'],how='left')
print(train_30.shape)
test_30 =test_30.merge(test_our,on=['description_id','paper_id'],how='left')
print(test_30.shape)
### get rank features
rank_feature = [i for i in test_30.columns if 'feat_' in i or 'num_' in i]
print('features need to be ranked:',len(rank_feature))
###about test features
rank_test = test_30.groupby('description_id')[rank_feature].rank(ascending=False)
rank_test.columns = [i+'_rank' for i in rank_test.columns]
test_30 = pd.concat([test_30,rank_test],axis=1)
###
rank_test = test_30.groupby('paper_id')[rank_feature].rank(ascending=False)
rank_test.columns = [i+'_paper_rank' for i in rank_test.columns]
test_30 = pd.concat([test_30,rank_test],axis=1)
test_30['num_des_count'] = test_30.groupby('description_id')['description_id'].transform('size')
test_30['num_ref_count'] = test_30.groupby('paper_id')['paper_id'].transform('size')
for i in test_30.columns.tolist():
    if '_rank' in i:
        test_30[i+'ratio'] = test_30[i]/test_30['num_des_count']
print('test features done.')
###about train features
rank_train = train_30.groupby('description_id')[rank_feature].rank(ascending=False)
rank_train.columns = [i+'_rank' for i in rank_train.columns]
train_30 = pd.concat([train_30,rank_train],axis=1)
rank_train = train_30.groupby('paper_id')[rank_feature].rank(ascending=False)
rank_train.columns = [i+'_paper_rank' for i in rank_train.columns]
train_30 = pd.concat([train_30,rank_train],axis=1)
train_30['num_des_count'] = train_30.groupby('description_id')['description_id'].transform('size')
train_30['num_ref_count'] = train_30.groupby('paper_id')['paper_id'].transform('size')
train_30.loc[train_30['num_des_count'].isnull(),'num_des_count'] = 50
train_30.loc[train_30['description_id'].isnull(),'description_id'] = -1
for i in tqdm(train_30.columns.tolist()):
    if 'rank' in i:
        train_30[i+'ratio'] = train_30[i]/train_30['num_des_count']
## for lgb lambdarank group useage
train_temp = train_30[['description_id','paper_id']]
train_temp['description_id'] = train_temp['description_id'].fillna(-1)
print(train_temp.shape)
test_temp = test_30[['description_id','paper_id']]
test_temp.to_csv('../final_dir/test_temp.csv',index=False)
#### about lgb model
train_y=train_30[['label']]
train_30.drop(['description_id','paper_id','label','is_added_to_recall_data'],axis=1,inplace=True)
test_30.drop(['description_id','paper_id'],axis=1,inplace=True)
print(train_30.shape,test_30.shape)
train_x=train_30
test_x=test_30
print(train_x.shape,test_x.shape,train_y.shape)
def lgb_train(train_x,train_y,valid_x,valid_y,group_train,group_valid):
    params = {
        'boosting_type': 'gbdt',
        'objective' : 'lambdarank',
        'metric': 'map',
        'num_leaves':64,
        'lambda_l1':1,
        'lambda_l2':0.1,
        'max_depth': -1,
        'learning_rate': 0.1,
        'min_child_samples':5,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'random_state':2019,
        'device': 'gpu',
        'gpu_platform_id': 1,
        'gpu_device_id': 1,
        'num_threads':30
#         'scale_pos_weight':len(train_y[train_y['label']==0])/len(train_y[train_y['label']==1])

    }
    lgb_train = lgb.Dataset(train_x,label=train_y,group=group_train,)
    lgb_validate = lgb.Dataset(valid_x, valid_y, reference=lgb_train,group=group_valid)
    model = lgb.train(params, lgb_train, valid_sets=(lgb_train,lgb_validate), num_boost_round=4000, early_stopping_rounds=100,verbose_eval=100)

    return model
from sklearn.model_selection import StratifiedKFold,GroupKFold,KFold
def LGB_CV(train_x, train_y, test_x, n_splits=5):
    folds = KFold(n_splits=n_splits, shuffle=False, random_state=2020)
    oof = np.zeros(train_x.shape[0])
    test_preds = np.zeros(test_x.shape[0])
    for fold_, (train_idx, val_idx) in enumerate(folds.split(train_x, train_y)):
        print("fold n°{}".format(fold_ + 1))
        train_flod_x=train_x.iloc[train_idx]
        train_flod_y=train_y.iloc[train_idx]
        validate_flod_x=train_x.iloc[val_idx]
        validate_flod_y=train_y.iloc[val_idx]
        # display(train_flod_x.head(2))
        ###get group
#             train_group = []
        fold_temp = train_temp.iloc[train_idx]
        print(fold_temp.shape)
        fold_temp['group'] = fold_temp.groupby('description_id')['description_id'].transform('size')
        temp = fold_temp[['description_id','group']].drop_duplicates()
        train_group = temp['group'].tolist()
        print(sum(train_group))
#             for i in train_30.iloc[train_idx]['description_id'].unique():
#                 train_group.append(len(train_30.iloc[train_idx][train_30.iloc[train_idx]['description_id']==i]))
#             valid_group = []
#             for i in train_30.iloc[val_idx]['description_id'].unique():
#                 valid_group.append(len(train_30.iloc[val_idx][train_30.iloc[val_idx]['description_id']==i]))
        fold_temp = train_temp.iloc[val_idx]
        print(fold_temp.shape)
        fold_temp['group'] = fold_temp.groupby('description_id')['description_id'].transform('size')
        temp = fold_temp[['description_id','group']].drop_duplicates()
        valid_group = temp['group'].tolist()
        print(sum(valid_group))
        ###
        print('traning...')
        model=lgb_train(train_flod_x,train_flod_y,validate_flod_x,validate_flod_y,train_group,valid_group)
        pd.to_pickle(model,f'../gbdt_models/lgb_rank_fold_{fold_}_model')
        pickled_model = pd.read_pickle(f'../gbdt_models/lgb_rank_fold_{fold_}_model')
        oof[val_idx] = pickled_model.predict(validate_flod_x)
        test_preds += pickled_model.predict(test_x)

    return oof,test_preds/5

train_preds,preds= LGB_CV(train_x, train_y, test_x, n_splits=5)
np.save('../final_dir/lgb_wanzheng_train_5cv.npy',train_preds)
np.save('../final_dir/lgb_wanzheng_test_5cv.npy',preds)
test_temp = test_30[['description_id','paper_id']]
test_30 = test_temp
test_30['preb']=preds
test_30 = test_30.sort_values('preb',ascending=False)
test_data = test_30
sub1 = test_data.groupby('description_id')['paper_id'].apply(list).apply(lambda x:x+['q','v','e']).apply(lambda x:x[0])
sub2 = test_data.groupby('description_id')['paper_id'].apply(list).apply(lambda x:x+['q','v','e']).apply(lambda x:x[1])
sub3 = test_data.groupby('description_id')['paper_id'].apply(list).apply(lambda x:x+['q','v','e']).apply(lambda x:x[2])
sub_wanzheng = pd.concat([sub1,sub2,sub3],axis=1).reset_index()
sub_wanzheng.columns=['description_id','pred1','pred2','pred3']
###
sub_wanzheng.to_csv('../submit/sub_lgb.csv',index=False,header=None)