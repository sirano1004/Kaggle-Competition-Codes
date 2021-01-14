# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 22:19:00 2020

@author: Gyu
"""

import os
file_dir = 'D:\Google Drive\Kaggle_Comp\RIIT'
os.chdir(file_dir)


import pickle
import pandas as pd
import numpy as np
import gc
from sklearn.metrics import roc_auc_score
from collections import defaultdict, deque
from tqdm.notebook import tqdm
import lightgbm as lgb



train_pickle = 'cv1_train.pickle'
valid_pickle = 'cv1_valid.pickle'
question_file = 'question3.csv'
debug = False
validaten_flg = False


# read data
feld_needed = ['user_id','content_id','answered_correctly','prior_question_elapsed_time','prior_question_had_explanation']
train = pd.read_pickle(train_pickle)[feld_needed]
valid = pd.read_pickle(valid_pickle)[feld_needed]
if debug:
    train = train[:1000000]
    valid = valid[:10000]
    
    
train = train.loc[train.answered_correctly != -1].reset_index(drop=True)
valid = valid.loc[valid.answered_correctly != -1].reset_index(drop=True)
_=gc.collect()

prior_question_elapsed_time_mean = train.prior_question_elapsed_time.dropna().values.mean()

content_df = train[['content_id','answered_correctly']].groupby(['content_id']).agg(['mean','std']).reset_index()
content_df.columns = ['content_id', 'answered_correctly_avg_c','answered_correctly_std_c']


train_time_diff = pd.read_csv('train_time_diff.csv')
train = pd.concat([train,train_time_diff], axis = 1)
train.time_diff.loc[train.time_diff >= 1e6] = 1e6
del(train_time_diff)
_=gc.collect()


content_df2 = train[['content_id','time_diff','prior_question_elapsed_time']].groupby(['content_id']).agg(['median']).reset_index()
content_df2.columns = ['content_id', 'time_diff_average_c','prior_elaps_time_average_c']

content_df = content_df.merge(content_df2, on = 'content_id', how = 'left')

content_df2 = train[['content_id','prior_question_had_explanation']].groupby(['content_id']).agg(['mean']).reset_index()
content_df2.columns = ['content_id','prior_has_explanation_average_c']
content_df = content_df.merge(content_df2, on = 'content_id', how = 'left')

del(content_df2)


#train = train[-90000000:]
_=gc.collect()


train_ascu = pd.read_csv('train_ascu.csv')
#train_ascu = train_ascu[-90000000:]
_=gc.collect()


train = pd.concat([train,train_ascu], axis = 1)
del(train_ascu)
_=gc.collect()


train_user_prev_q_a = pd.read_csv('train_user_prev_q_a.csv')
#train_user_prev_q_a = train_user_prev_q_a[-90000000:]
_=gc.collect()


train = pd.concat([train,train_user_prev_q_a['user_prev_tag_lag1']], axis = 1)
del(train_user_prev_q_a)
_=gc.collect()


train_near_past = pd.read_csv('train_near_past.csv')
#train_near_past = train_near_past[-90000000:]
_ = gc.collect()


train = pd.concat([train,train_near_past[['last_60','part_last_30']]], axis = 1)
del(train_near_past)
_=gc.collect()


valid_ascu = pd.read_csv('valid_ascu.csv')
valid_time_diff = pd.read_csv('valid_time_diff.csv')
valid_user_prev_q_a = pd.read_csv('valid_user_prev_q_a.csv')
valid_near_past = pd.read_csv('valid_near_past.csv')


valid = pd.concat([valid,valid_ascu,valid_time_diff,valid_user_prev_q_a,valid_near_past], axis = 1)
valid.time_diff.loc[valid.time_diff >= 1e6] = 1e6
del(valid_ascu)
del(valid_time_diff)
del(valid_user_prev_q_a)
del(valid_near_past)


train['prior_question_elapsed_time'] = train.prior_question_elapsed_time.fillna(prior_question_elapsed_time_mean)
valid['prior_question_elapsed_time'] = valid.prior_question_elapsed_time.fillna(prior_question_elapsed_time_mean)

train.prior_question_had_explanation = train.prior_question_had_explanation.fillna(False).astype('int8')
valid.prior_question_had_explanation = valid.prior_question_had_explanation.fillna(False).astype('int8')


train['answered_correctly_avg_u'] = train['answered_correctly_sum_u']/train['count_u']
valid['answered_correctly_avg_u'] = valid['answered_correctly_sum_u']/valid['count_u']


train['part_answered_correctly_avg_u'] = train['part_answered_correctly_sum_u']/train['part_count_u']
valid['part_answered_correctly_avg_u'] = valid['part_answered_correctly_sum_u']/valid['part_count_u']

questions_df = pd.read_csv(question_file)
#questions_df.tags.fillna('-1-1', inplace = True)
questions_df['tag_sum'] = pd.factorize(questions_df.tag_sum)[0]



train = pd.merge(train, questions_df[['content_id', 'part','tag_sum']], on = 'content_id', how = 'left')
valid = pd.merge(valid, questions_df[['content_id', 'part','tag_sum']], on = 'content_id', how = 'left')



train = pd.merge(train, content_df, on=['content_id'], how="left")
valid = pd.merge(valid, content_df, on=['content_id'], how="left")



train["time_diff_lag1"] = train.groupby('user_id').time_diff.shift()
train["time_diff_lag2"] = train.groupby('user_id').time_diff_lag1.shift()

valid["time_diff_lag1"] = valid.groupby('user_id').time_diff.shift()
valid["time_diff_lag2"] = valid.groupby('user_id').time_diff_lag1.shift()



del(questions_df)
del(content_df)
_=gc.collect()



TARGET = 'answered_correctly'
FEATS = ['answered_correctly_avg_u', 'last_60', 'part_last_30', 'part_answered_correctly_avg_u', 'answered_correctly_avg_c','answered_correctly_std_c','prior_has_explanation_average_c', 'time_diff_average_c','prior_elaps_time_average_c'
         ,'count_u', 'part_count_u', 'answered_correctly_sum_u', 'part_answered_correctly_sum_u', 'part', 'prior_question_elapsed_time','time_diff','tag_sum','user_prev_tag_lag1', 'time_diff_lag1','time_diff_lag2','time_diff_correct','time_diff_incorrect']
dro_cols = list(set(train.columns) - set(FEATS))
y_tr = train[TARGET]
y_va = valid[TARGET]
train.drop(dro_cols, axis=1, inplace=True)
valid.drop(dro_cols, axis=1, inplace=True)
_=gc.collect()



lgb_train = lgb.Dataset(train[FEATS], y_tr, categorical_feature = ['tag_sum', 'user_prev_tag_lag1', 'part'],free_raw_data=False)
lgb_valid = lgb.Dataset(valid[FEATS], y_va, categorical_feature = ['tag_sum', 'user_prev_tag_lag1', 'part'], reference=lgb_train,free_raw_data=False)
del train, y_tr
_=gc.collect()


#parameters = {'num_leaves': 501, 'max_bin': 804, 'max_depth': 12, 'min_child_weight': 5, 'feature_fraction': 0.41464777403082415, 'bagging_fraction': 0.8822187573608076, 'bagging_freq': 7, 'min_child_samples': 66, 'lambda_l1': 1.976094455751495e-08, 'lambda_l2': 3.693925899782827e-05}
parameters = {'num_leaves': 320, 'max_bin': 757, 'max_depth': 14, 'min_child_weight': 4, 'feature_fraction': 0.5600322463517959, 'bagging_fraction': 0.9944120638945314, 'bagging_freq': 2, 'min_child_samples': 37, 'lambda_l1': 0.0618462226804493, 'lambda_l2': 4.618748544670815e-05}
parameters['objective'] = 'binary'
parameters['metric'] = 'auc'
parameters['early_stopping_rounds'] = 15
parameters['boosting'] = 'gbdt' 



model = lgb.train(
                    parameters, 
                    lgb_train,
                    valid_sets=[lgb_train, lgb_valid],
                    verbose_eval=100,
                    num_boost_round=10000,
                    #callbacks=[lgb.reset_parameter(learning_rate = lambda current_round: 0.2*0.998**(current_round))]
                    #early_stopping_rounds=15
                )

print('auc:', roc_auc_score(y_va, model.predict(valid[FEATS])))
_ = lgb.plot_importance(model)



import matplotlib.pyplot as plt
lgb.plot_importance(model, importance_type = 'gain')
plt.show()


import joblib# save model

joblib.dump(model, 'lgb_model3.pkl')
# load model
#gbm_pickle = joblib.load('lgb_model.pkl')


