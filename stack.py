#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 23:09:59 2019

@author: chenhaibin
"""
'''
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import RepeatedKFold
import pandas as pd
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
from sklearn.model_selection import KFold
'''
    
'''
first = pd.read_csv("/Users/chenhaibin/Desktop/python/ccf_consumer_score/融合/0.5_bayesian_blending_2.csv")


first['score'] = first['score'].apply(lambda x: int(np.round(x)))

first[['id','score']].to_csv('blending_last_2.csv', index=False)
'''

'''
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_absolute_error
# 将lgb和xgb的结果进行stacking
answer = pd.read_csv('/Users/chenhaibin/Desktop/python/ccf_consumer_score/融合/baseline_github_lightgbm_xgboost.csv')
train_stack = np.vstack([answer['validate_lightgbm_github'],answer['validate_xgboost_github'],answer['validate_lightgbm_origin']]).transpose()
test_stack = np.vstack([answer['lightgbm_github_good'], answer['xgboost_github'],answer['lightgbm_orgin']]).transpose()

folds_stack = RepeatedKFold(n_splits=5, n_repeats=2, random_state=2019)
oof_stack = np.zeros(train_stack.shape[0])
predictions = np.zeros(test_stack.shape[0])
train_data = pd.read_csv('train_dataset.csv')
target=train_data['信用分']
for fold_, (trn_idx, val_idx) in enumerate(folds_stack.split(train_stack,target)):
    print("fold {}".format(fold_))
    trn_data, trn_y = train_stack[trn_idx], target.iloc[trn_idx].values
    val_data, val_y = train_stack[val_idx], target.iloc[val_idx].values
    
    clf_3 = BayesianRidge()
    #clf_3 = LogisticRegression()
    clf_3.fit(trn_data, trn_y)
    
    oof_stack[val_idx] = clf_3.predict(val_data)
    predictions += clf_3.predict(test_stack) / 10
    
#贝叶斯岭回归 error 14.64
#lr  34...
print(mean_absolute_error(target.values, oof_stack))
'''

'''
first = pd.read_csv('/Users/chenhaibin/Desktop/python/ccf_consumer_score/rmse_lightgbm最佳/rmse_lightgbm_19.csv')
tmp = pd.concat([first['score_mae'], first['score_rmse']], axis=1)
tmp = tmp.sort_values('score_mae')
tmp['ranks'] = list(range(tmp.shape[0]))
tmp['true_score'] = tmp['score_mae'].values
tmp.loc[tmp.ranks<5000,'true_score']  = tmp.loc[tmp.ranks< 5000,'score_rmse'].values *0.6 + tmp.loc[tmp.ranks< 5000,'score_mae'].values * 0.4
tmp.loc[tmp.ranks>45000,'true_score']  = tmp.loc[tmp.ranks> 45000,'score_rmse'].values *0.6 + tmp.loc[tmp.ranks> 45000,'score_mae'].values * 0.4
tmp = tmp.sort_index()

second = pd.read_csv('/Users/chenhaibin/Desktop/python/ccf_consumer_score/rmse_lightgbm最佳/rmse_xgboost.csv')
tmp2 = pd.concat([second['score_mae'], second['score_rmse']], axis=1)
tmp2 = tmp2.sort_values('score_mae')
tmp2['ranks'] = list(range(tmp2.shape[0]))
tmp2['true_score'] = tmp2['score_mae'].values
tmp2.loc[tmp2.ranks<5000,'true_score']  = tmp2.loc[tmp2.ranks< 5000,'score_rmse'].values *0.6 + tmp2.loc[tmp2.ranks< 5000,'score_mae'].values * 0.4
tmp2.loc[tmp2.ranks>45000,'true_score']  = tmp2.loc[tmp2.ranks> 45000,'score_rmse'].values *0.6 + tmp2.loc[tmp2.ranks> 45000,'score_mae'].values * 0.4
tmp2 = tmp2.sort_index()

third = pd.read_csv('/Users/chenhaibin/Desktop/python/ccf_consumer_score/rmse_lightgbm最佳/rmse_github_gbm.csv')
tmp3 = pd.concat([third['score_mae'], third['score_rmse']], axis=1)
tmp3 = tmp3.sort_values('score_mae')
tmp3['ranks'] = list(range(tmp3.shape[0]))
tmp3['true_score'] = tmp3['score_mae'].values
tmp3.loc[tmp.ranks<5000,'true_score'] = tmp3.loc[tmp3.ranks< 5000,'score_rmse'].values *0.6 + tmp3.loc[tmp.ranks< 5000,'score_mae'].values * 0.4
tmp3.loc[tmp.ranks>45000,'true_score'] = tmp3.loc[tmp3.ranks> 45000,'score_rmse'].values *0.6 + tmp3.loc[tmp.ranks> 45000,'score_mae'].values * 0.4
tmp3 = tmp3.sort_index()



forth = pd.read_csv('/Users/chenhaibin/Desktop/python/ccf_consumer_score/rmse_lightgbm最佳/rmse_catboost.csv')

fifth = pd.read_csv('/Users/chenhaibin/Desktop/python/ccf_consumer_score/融合/baseline_github_lightgbm_xgboost.csv')



tmp['final_score'] = (tmp['true_score']*5+tmp2['true_score']*3+\
     tmp3['true_score']*4+forth['score_rmse']*2+fifth['xgboost_origin_good'])/15
   
tmp['final_score'] = tmp['final_score'].apply(lambda x: int(np.round(x)))
'''

'''
first = pd.read_csv('/Users/chenhaibin/Desktop/python/ccf_consumer_score/融合/0.5第一层_dataset1.csv')
second = pd.read_csv('/Users/chenhaibin/Desktop/python/ccf_consumer_score/融合/0.5第一层_dataset2.csv')
dataset_d1 = pd.concat([first['dataset_d1_1'], first['dataset_d1_2'],first['dataset_d1_3'],first['dataset_d1_4']], axis=1)
dataset_d2 = pd.concat([second['dataset_d2_1'], second['dataset_d2_2'],second['dataset_d2_3'],second['dataset_d2_4']], axis=1)
y_d2 = first['y_d2']
oof_stack = np.zeros(len(dataset_d1))
y_submission = np.zeros(len(dataset_d2))
#0.0639466601764

folds_stack = RepeatedKFold(n_splits=5, n_repeats=2, random_state=2019)


dataset_d1 = np.array(dataset_d1)
dataset_d2 = np.array(dataset_d2)
y_d2 = np.array(y_d2)


for fold_, (trn_idx, val_idx) in enumerate(folds_stack.split(dataset_d1,y_d2)):
    print("fold {}".format(fold_))
    trn_data, trn_y = dataset_d1[trn_idx], y_d2[trn_idx]
    val_data, val_y = dataset_d1[val_idx], y_d2[val_idx]
    
    clf_3 = BayesianRidge()
    clf_3.fit(trn_data, trn_y)
    oof_stack[val_idx] = clf_3.predict(val_data)
    y_submission += clf_3.predict(dataset_d2)/10
    
absolute_error = mean_absolute_error(y_d2, oof_stack)
print("all absolute_error is "+str(absolute_error))
fold_score = 1 / (1 + absolute_error)
print("all l1 score is "+str(fold_score))
'''

'''
en_amount = 3

params = {
    'learning_rate': 0.003,
    'boosting_type': 'gbdt',
    'objective': 'regression_l1',
    'metric': 'mae',
    'nthread': 8
}

for seed in range(en_amount):
    NFOLDS=5
    kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
    kf = kfold.split(dataset_d1, y_d2)
    valid_best_l2_all = 0

    for i, (train_fold, validate) in enumerate(kf):
        print('fold: ',i, ' training')
        X_train, X_validate, label_train, label_validate = \
        dataset_d1.iloc[train_fold, :], dataset_d1.iloc[validate, :], \
        y_d2[train_fold], y_d2[validate]
        dtrain = lgb.Dataset(X_train, label_train)
        dvalid = lgb.Dataset(X_validate, label_validate, reference=dtrain)
        bst = lgb.train(params, dtrain, num_boost_round=15000, valid_sets=dvalid, verbose_eval=-1,early_stopping_rounds=250)
        y_submission += bst.predict(dataset_d2, num_iteration=bst.best_iteration)
        print("l1 score "+str(bst.best_score['valid_0']['l1']))
        valid_best_l2_all += bst.best_score['valid_0']['l1']
        oof_stack[validate] = bst.predict(X_validate,num_iteration=bst.best_iteration)
    absolute_error = mean_absolute_error(y_d2, oof_stack)
    print("all absolute_error is "+str(absolute_error))
    
valid_best_l2_all /= NFOLDS
valid_best_l2_all /= en_amount
y_submission /= NFOLDS
y_submission /= en_amount
print('cv score for valid is: ', 1/(1+valid_best_l2_all))
#lightgbm 0.06380419912
'''