#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 17:33:38 2019

@author: chenhaibin
"""

import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

#input
train_data = pd.read_csv("train_dataset.csv")
test_data = pd.read_csv("test_dataset.csv")

#feature_engineering
train_data.columns = ['uid','true_name_flag','age','is_uni_student_flag','black_list_flag',\
                     '4g_unhealth_flag','net_age_till_now','top_up_month_diff','top_up_amount',\
                     'recent_6month_avg_use','total_account_fee','curr_month_balance',\
                     'curr_overdue_flag','cost_sensitivity','connect_num','freq_shopping_flag',\
                     'recent_3month_shopping_count','wanda_flag','sam_flag','movie_flag',\
                     'tour_flag','sport_flag','online_shopping_count','express_count',\
                     'finance_app_count','video_app_count','flight_count','train_count',\
                     'tour_app_count','score']
test_data.columns = train_data.columns[:-1]

#去掉5个最没用的特征
train_data_use = train_data.drop(['uid','score','is_uni_student_flag','true_name_flag','sam_flag','black_list_flag','wanda_flag'], axis=1)
test_data_use = test_data.drop(['uid','is_uni_student_flag','true_name_flag','sam_flag','black_list_flag','wanda_flag'], axis=1)

#入网年龄 降了0.01
def get_begin_age(a, b):
    return a - b / 12
train_data['net_age_begin'] = train_data.apply(lambda row: get_begin_age(row['age'], row['net_age_till_now']), axis=1)



def pay_remain_ratio(a,b):
    if b == 0:
        return 0
    else:
        return a / b
#扣钱后剩的钱和月费用的比例 
train_data['after_pay_remain_ratio'] = train_data.apply(lambda row: pay_remain_ratio(row['curr_month_balance'],row['total_account_fee']), axis=1)

#扣钱后剩的钱和平均消费的比例
train_data['after_pay_remain_average_ratio'] = train_data.apply(lambda row: pay_remain_ratio(row['curr_month_balance'],row['recent_6month_avg_use']), axis=1)

#下面这里都是全部除一遍就完事了
train_data['after_pay_remain_average_ratio'] = train_data.apply(lambda row: pay_remain_ratio(row['top_up_amount'],row['recent_6month_avg_use']), axis=1)
train_data['after_pay_remain_average_ratio'] = train_data.apply(lambda row: pay_remain_ratio(row['total_account_fee'],row['recent_6month_avg_use']), axis=1)
train_data['after_pay_remain_average_ratio'] = train_data.apply(lambda row: pay_remain_ratio(row['top_up_amount'],row['curr_month_balance']), axis=1)
train_data['after_pay_remain_average_ratio'] = train_data.apply(lambda row: pay_remain_ratio(row['total_account_fee'],row['top_up_amount']), axis=1)
train_data['charge_online'] = train_data['top_up_amount'].apply(lambda x: 1 if x % 1 != 0 else 0)

#计数特征
def feature_count(data, features=[]):
    if len(set(features)) != len(features):
        print('equal feature !!!!')
        return data
    new_feature = 'count'
    for i in features:
        new_feature += '_' + i.replace('add_', '')
    try:
        del data[new_feature]
    except:
        pass
    temp = data.groupby(features).size().reset_index().rename(columns={0: new_feature})
    
    data = data.merge(temp, 'left', on=features)
    return data

train_data = feature_count(train_data, ['top_up_amount'])
train_data = feature_count(train_data, ['net_age_till_now']) #go
train_data = feature_count(train_data, ['age']) #go
train_data = feature_count(train_data, ['top_up_month_diff'])
train_data = feature_count(train_data, ['connect_num']) #go

train_data['diff_between_average_acount'] = train_data['recent_6month_avg_use'] - train_data['total_account_fee']
train_data['diff_between_average_up'] = train_data['recent_6month_avg_use'] - train_data['top_up_amount']
train_data['diff_between_curr_up'] = train_data['curr_month_balance'] - train_data['top_up_amount']

train_data['diff_between_fee_up'] = train_data['total_account_fee'] - train_data['top_up_amount']
train_data['diff_between_curr_up'] = train_data['curr_month_balance'] - train_data['top_up_amount']
train_data['diff_between_average_up'] = train_data['curr_month_balance'] - train_data['recent_6month_avg_use']
#average
train_data['mean_fee'] = (train_data['recent_6month_avg_use'] - train_data['total_account_fee']) / 2

#parameter
'''
params = {
    'learning_rate': 0.01,
    'boosting_type': 'gbdt',
    'objective': 'regression_l1',
    'metric': 'mae',
    'feature_fraction': 0.6,
    'bagging_fraction': 0.8,
    'bagging_freq': 2,
    'num_leaves': 31,
    'verbose': -1,
    'max_depth': 5,
    'lambda_l2': 5, 'lambda_l1': 0
}
'''
params = {
    'learning_rate': 0.01,
    'boosting_type': 'gbdt',
    'objective': 'regression_l1',
    'metric': 'mae',
    'max_depth': 5,
    'num_leaves': 27,
    'min_sum_hessian_in_leaf': 0.0001, 
    'min_data_in_leaf': 24
}

min_merror = float('Inf')
best_params = {}
train_label = train_data['score']
train_data = lgb.Dataset(train_data_use, train_label, silent=True)

'''
print("调参1：提高准确率")
min_merror = float('Inf')
#for max_depth in range(3,7,1):
    #for num_leaves in range(10,60,3):
for max_depth in range(4,7,1):
    for num_leaves in range(25,35,2):
        params['num_leaves'] = num_leaves
        params['max_depth'] = max_depth
        cv_results = lgb.cv(params,
                            train_data,
                            seed=89,
                            early_stopping_rounds=100,
                            verbose_eval=True,
                            num_boost_round=10000 #默认10
                            )
        mean_merror = pd.Series(cv_results['l1-mean']).min()
            
        if mean_merror < min_merror:
            min_merror = mean_merror
            best_params['num_leaves'] = num_leaves
            best_params['max_depth'] = max_depth
            
params['num_leaves'] = best_params['num_leaves']
params['max_depth'] = best_params['max_depth']
print(best_params)
print('best n_estimators:', len(cv_results['l1-mean']))
print('best cv score:', cv_results['l1-mean'][-1])
'''
'''
{'num_leaves': 27, 'max_depth': 5}
best n_estimators: 3650
best cv score: 14.788533076
'''

'''
print("调参2：降低过拟合")
for min_sum_hessian_in_leaf in [1e-4,1e-3,1e-2]:
    for min_data_in_leaf in range(12,32,4):
#for min_sum_hessian_in_leaf in [1e-3]:
    #for min_data_in_leaf in range(20,21,1):
            params['min_sum_hessian_in_leaf'] = min_sum_hessian_in_leaf
            params['min_data_in_leaf'] = min_data_in_leaf
            
            cv_results = lgb.cv(
                                params,
                                train_data,
                                seed=89,
                                early_stopping_rounds=100,
                                verbose_eval=True,
                                num_boost_round=10000
                                )
                    
            mean_merror = pd.Series(cv_results['l1-mean']).min()
            boost_rounds = pd.Series(cv_results['l1-mean']).idxmin()

            if mean_merror < min_merror:
                min_merror = mean_merror
                best_params['min_sum_hessian_in_leaf']= min_sum_hessian_in_leaf
                best_params['min_data_in_leaf'] = min_data_in_leaf

params['min_data_in_leaf'] = best_params['min_data_in_leaf']
params['min_sum_hessian_in_leaf'] = best_params['min_sum_hessian_in_leaf']
print(best_params)
print('best n_estimators:', len(cv_results['l1-mean']))
print('best cv score:', cv_results['l1-mean'][-1])
'''

'''
{'min_sum_hessian_in_leaf': 0.0001, 'min_data_in_leaf': 24}
best n_estimators: 2921
best cv score: 14.7871626736
'''


print("调参3：降低过拟合")
for feature_fraction in [0.5,0.6,0.7]:
    for bagging_fraction in [0.7,0.8,0.9]:
        for bagging_freq in range(0,5,1):
            params['feature_fraction'] = feature_fraction
            params['bagging_fraction'] = bagging_fraction
            params['bagging_freq'] = bagging_freq
            
            cv_results = lgb.cv(
                                params,
                                train_data,
                                seed=89,
                                early_stopping_rounds=100,
                                verbose_eval=True,
                                num_boost_round=10000
                                )
                    
            mean_merror = pd.Series(cv_results['l1-mean']).min()
            boost_rounds = pd.Series(cv_results['l1-mean']).idxmin()

            if mean_merror < min_merror:
                min_merror = mean_merror
                best_params['feature_fraction'] = feature_fraction
                best_params['bagging_fraction'] = bagging_fraction
                best_params['bagging_freq'] = bagging_freq

params['feature_fraction'] = best_params['feature_fraction']
params['bagging_fraction'] = best_params['bagging_fraction']
params['bagging_freq'] = best_params['bagging_freq']
print(best_params)
print('best n_estimators:', len(cv_results['l1-mean']))
print('best cv score:', cv_results['l1-mean'][-1])


'''
cv_pred = np.zeros(test_data.shape[0])
valid_best_l2_all = 0

feature_importance_df = pd.DataFrame()
count = 0
for i, (train_fold, validate) in enumerate(kf):
    print('fold: ',i, ' training')
    X_train, X_validate, label_train, label_validate = \
    train_data_use.iloc[train_fold, :], train_data_use.iloc[validate, :], \
    train_label[train_fold], train_label[validate]
    dtrain = lgb.Dataset(X_train, label_train)
    dvalid = lgb.Dataset(X_validate, label_validate, reference=dtrain)
    bst = lgb.train(params, dtrain, num_boost_round=10000, valid_sets=dvalid, verbose_eval=-1,early_stopping_rounds=50)
    cv_pred += bst.predict(test_data_use, num_iteration=bst.best_iteration)
    valid_best_l2_all += bst.best_score['valid_0']['l1']

    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = list(X_train.columns)
    fold_importance_df["importance"] = bst.feature_importance(importance_type='gain', iteration=bst.best_iteration)
    fold_importance_df["fold"] = count + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    count += 1

cv_pred /= NFOLDS
valid_best_l2_all /= NFOLDS
print('cv score for valid is: ', 1/(1+valid_best_l2_all))
'''