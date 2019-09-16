#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 21:28:34 2019

@author: chenhaibin
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

import xgboost as xgb
    
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

cut_feature = ['age','net_age_till_now','top_up_month_diff','top_up_amount',\
               'recent_6month_avg_use','total_account_fee','curr_month_balance',\
               'connect_num',\
               'recent_3month_shopping_count','online_shopping_count','express_count',\
                     'finance_app_count','video_app_count','flight_count','train_count',\
                     'tour_app_count']
#train
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.show()
    return best_features

#做截断
for col in cut_feature:
    high = np.percentile(train_data[col].values, 99.8)
    low = np.percentile(train_data[col].values, 0.2)
    train_data.loc[train_data[col] > high, col] = high
    train_data.loc[train_data[col] < low, col] = low #loc按要求取元素
    test_data.loc[test_data[col] > high, col] = high
    test_data.loc[test_data[col] < low, col] = low #loc按要求取元素


#太大的值做平滑处理
for col in ['online_shopping_count','finance_app_count','video_app_count']:
    train_data[col] = train_data[col].map(lambda x: np.log1p(x))
    test_data[col] = test_data[col].map(lambda x: np.log1p(x))

#入网年龄 降了0.01
def get_begin_age(a, b):
    return a - b / 12
train_data['net_age_begin'] = train_data.apply(lambda row: get_begin_age(row['age'], row['net_age_till_now']), axis=1)

#交通类汇总
train_data['all_traffic'] = train_data['flight_count'] + train_data['train_count']

#wealthy flag
train_data['wealthy_flag'] = train_data['sam_flag'] + train_data['wanda_flag']

def pay_remain_ratio(a,b):
    if b == 0:
        return 0
    else:
        return a / b
#扣钱后剩的钱和月费用的比例 
train_data['after_pay_remain_ratio'] = train_data.apply(lambda row: pay_remain_ratio(row['curr_month_balance'],row['total_account_fee']+1), axis=1)

#扣钱后剩的钱和平均消费的比例
train_data['after_pay_remain_average_ratio'] = train_data.apply(lambda row: pay_remain_ratio(row['curr_month_balance'],row['recent_6month_avg_use']+1), axis=1)

#下面这里都是全部除一遍就完事了
train_data['after_pay_remain_average_ratio'] = train_data.apply(lambda row: pay_remain_ratio(row['top_up_amount'],row['recent_6month_avg_use']+1), axis=1)
train_data['after_pay_remain_average_ratio'] = train_data.apply(lambda row: pay_remain_ratio(row['total_account_fee'],row['recent_6month_avg_use']+1), axis=1)
train_data['after_pay_remain_average_ratio'] = train_data.apply(lambda row: pay_remain_ratio(row['top_up_amount'],row['curr_month_balance']+1), axis=1)
train_data['after_pay_remain_average_ratio'] = train_data.apply(lambda row: pay_remain_ratio(row['total_account_fee'],row['top_up_amount']+1), axis=1)
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
#平滑处理

for col in ['count_connect_num','count_net_age_till_now',\
            'count_top_up_amount','count_top_up_month_diff']:
    train_data[col] = train_data[col].map(lambda x: np.log1p(x))
   
train_data['diff_between_average_acount'] = train_data['recent_6month_avg_use'] - train_data['total_account_fee']
train_data['diff_between_average_up'] = train_data['recent_6month_avg_use'] - train_data['top_up_amount']
train_data['diff_between_curr_up'] = train_data['curr_month_balance'] - train_data['top_up_amount']

train_data['diff_between_fee_up'] = train_data['total_account_fee'] - train_data['top_up_amount']
train_data['diff_between_curr_up'] = train_data['curr_month_balance'] - train_data['top_up_amount']
train_data['diff_between_average_up'] = train_data['curr_month_balance'] - train_data['recent_6month_avg_use']
#average
train_data['mean_fee'] = (train_data['recent_6month_avg_use'] + train_data['total_account_fee']) / 2

#train_data['不确定角色'] = 0
#train_data['不确定角色'][train_data['age'] == 0] = 1
    
train_data['primary_student'] = 0
train_data['primary_student'][(train_data['age'] > 0) & (train_data['age'] < 18) & (train_data['is_uni_student_flag'] == 0)] = 1
    
train_data['univer_student'] = 0
train_data['univer_student'][((train_data['age'] >= 18) & (train_data['age'] < 23)) | (train_data['is_uni_student_flag'] == 1)] = 1
    
train_data['work_in_10'] = 0
train_data['work_in_10'][((train_data['age'] >= 23) & (train_data['age'] < 30))] = 1
    
train_data['work_out_10'] = 0
train_data['work_out_10'][((train_data['age'] >= 30) & (train_data['age'] < 45))] = 1
    
train_data['old'] = 0
train_data['old'][(train_data['age'] >= 45)] = 1

# 是否新互联网用户 用户网龄分段，<12，[12,  36)，[36,  120)，[120,  +)
train_data['new_account'] = 0
train_data['new_account'][(train_data['net_age_till_now'] < 12)] = 1
    
train_data['3_year_account'] = 0
train_data['3_year_account'][(train_data['net_age_till_now'] >= 12) & (train_data['net_age_till_now'] < 36)] = 1
    
train_data['10_year_account'] = 0
train_data['10_year_account'][(train_data['net_age_till_now'] >= 36) & (train_data['net_age_till_now'] < 120)] = 1
    
train_data['old_account'] = 0
train_data['old_account'][(train_data['net_age_till_now'] >= 120)] = 1
    
#test_data
test_data['net_age_begin'] = test_data.apply(lambda row: get_begin_age(row['age'], row['net_age_till_now']), axis=1)
test_data['all_traffic'] = test_data['flight_count'] + test_data['train_count']
test_data['wealthy_flag'] = test_data['sam_flag'] + test_data['wanda_flag']
test_data['after_pay_remain_ratio'] = test_data.apply(lambda row: pay_remain_ratio(row['curr_month_balance'],row['total_account_fee']+1), axis=1)
test_data['after_pay_remain_average_ratio'] = test_data.apply(lambda row: pay_remain_ratio(row['curr_month_balance'],row['recent_6month_avg_use']+1), axis=1)
test_data['after_pay_remain_average_ratio'] = test_data.apply(lambda row: pay_remain_ratio(row['top_up_amount'],row['recent_6month_avg_use']+1), axis=1)
test_data['after_pay_remain_average_ratio'] = test_data.apply(lambda row: pay_remain_ratio(row['total_account_fee'],row['recent_6month_avg_use']+1), axis=1)
test_data['after_pay_remain_average_ratio'] = test_data.apply(lambda row: pay_remain_ratio(row['top_up_amount'],row['curr_month_balance']+1), axis=1)
test_data['after_pay_remain_average_ratio'] = test_data.apply(lambda row: pay_remain_ratio(row['total_account_fee'],row['top_up_amount']+1), axis=1)
test_data['charge_online'] = test_data['top_up_amount'].apply(lambda x: 1 if x % 1 != 0 else 0)
test_data = feature_count(test_data, ['top_up_amount'])
test_data = feature_count(test_data, ['net_age_till_now']) #go
test_data = feature_count(test_data, ['age']) #go
test_data = feature_count(test_data, ['top_up_month_diff'])
test_data = feature_count(test_data, ['connect_num']) #go

for col in ['count_connect_num','count_net_age_till_now',\
            'count_top_up_amount','count_top_up_month_diff']:
    test_data[col] = test_data[col].map(lambda x: np.log1p(x))   

test_data['diff_between_average_acount'] = test_data['recent_6month_avg_use'] - test_data['total_account_fee']
test_data['diff_between_average_up'] = test_data['recent_6month_avg_use'] - test_data['top_up_amount']
test_data['diff_between_curr_up'] = test_data['curr_month_balance'] - test_data['top_up_amount']
test_data['diff_between_fee_up'] = test_data['total_account_fee'] - test_data['top_up_amount']
test_data['diff_between_curr_up'] = test_data['curr_month_balance'] - test_data['top_up_amount']
test_data['diff_between_average_up'] = test_data['curr_month_balance'] - test_data['recent_6month_avg_use']
test_data['mean_fee'] = (test_data['recent_6month_avg_use'] + test_data['total_account_fee']) / 2
#test_data['不确定角色'] = 0
#test_data['不确定角色'][test_data['age'] == 0] = 1
test_data['primary_student'] = 0
test_data['primary_student'][(test_data['age'] > 0) & (test_data['age'] < 18) & (test_data['is_uni_student_flag'] == 0)] = 1
test_data['univer_student'] = 0
test_data['univer_student'][((test_data['age'] >= 18) & (test_data['age'] < 23)) | (test_data['is_uni_student_flag'] == 1)] = 1
test_data['work_in_10'] = 0
test_data['work_in_10'][((test_data['age'] >= 23) & (test_data['age'] < 30))] = 1
test_data['work_out_10'] = 0
test_data['work_out_10'][((test_data['age'] >= 30) & (test_data['age'] < 45))] = 1
test_data['old'] = 0
test_data['old'][(test_data['age'] >= 45)] = 1
test_data['new_account'] = 0
test_data['new_account'][(test_data['net_age_till_now'] < 12)] = 1
test_data['3_year_account'] = 0
test_data['3_year_account'][(test_data['net_age_till_now'] >= 12) & (test_data['net_age_till_now'] < 36)] = 1
test_data['10_year_account'] = 0
test_data['10_year_account'][(test_data['net_age_till_now'] >= 36) & (test_data['net_age_till_now'] < 120)] = 1
test_data['old_account'] = 0
test_data['old_account'][(test_data['net_age_till_now'] >= 120)] = 1

train_label = train_data['score']

#去掉3个
train_data_use = train_data.drop(['uid','score','is_uni_student_flag','true_name_flag','black_list_flag'], axis=1)
test_data_use = test_data.drop(['uid','is_uni_student_flag','true_name_flag','black_list_flag'], axis=1)

#再去掉
train_data_use = train_data_use.drop(['work_in_10','10_year_account'], axis=1)
test_data_use = test_data_use.drop(['work_in_10','10_year_account'], axis=1)

NFOLDS = 5
kfold = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=2019)
kf = kfold.split(train_data_use,train_label)

# init var
cv_pred = np.zeros(test_data.shape[0])
cv_score = 0
count = 0

# model=None
for i, (train_index, val_index) in enumerate(kf):
    print('fold: ',i, ' training')
    X_train, X_val, Y_train, Y_val = train_data_use.iloc[train_index, :], train_data_use.iloc[val_index, :], \
    train_label[train_index], train_label[val_index]
    
    params = {'learning_rate': 0.003, 'n_estimators': 8000, 'max_depth': 6, 'min_child_weight': 10, 'seed': 0,
            'subsample': 0.6, 'colsample_bytree': 0.5, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 5, 'n_jobs':20}
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, Y_train, eval_metric=mean_absolute_error)
    
    cv_pred += model.predict(test_data_use)
    cv_score += (1 / (mean_absolute_error(Y_val, model.predict(X_val)) + 1))

test_pred = cv_pred / NFOLDS
cv_score /= NFOLDS
print('xgb score is: ', cv_score)

valdata = xgb.DMatrix(X_val.values, label=Y_val.values)

train_score = (1 / (mean_absolute_error(Y_train, model.predict(X_train)) + 1))
print ('train score = ', train_score)

test_score = (1 / (mean_absolute_error(Y_val, model.predict(X_val)) + 1))
print ('test score = ', test_score)

test_data_sub = test_data[['uid']]

test_data_sub['score'] = cv_pred
test_data_sub.columns = ['id','score']
test_data_sub['score'] = test_data_sub['score'].apply(lambda x: int(np.round(x)))
test_data_sub[['id','score']].to_csv('xgboost_1.csv', index=False)

#xgb 0.063718 lightgbm+xgboost 0.06381



