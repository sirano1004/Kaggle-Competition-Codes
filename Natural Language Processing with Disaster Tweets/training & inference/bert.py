# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 15:24:46 2020

@author: gouri
"""

import os
path="D:\\Google Drive\\Kaggle_Comp\\nlp_getting_started\\Gourika"
os.chdir(path)
import numpy as np # For numerical fast numerical calculations
#import matplotlib.pyplot as plt # For making plots
import pandas as pd # Deals with data
#import seaborn as sns # Makes beautiful plots
#from sklearn.preprocessing import StandardScaler # Testing sklearn
import tensorflow as tf# Imports tensorflow
tf.compat.v1.enable_eager_execution()
#import keras # Imports keras

from tensorflow.keras.layers import Dense, Input, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.losses import binary_crossentropy
#from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.regularizers import l2, l1
import tensorflow_hub as hub
from keras import backend as K

import tokenization



def bert_encode(texts, tokenizer, max_len=512):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for text in texts:
        text = tokenizer.tokenize(text)
            
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)
        
        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len
        
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
    
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)


def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))



def build_model(bert_layer, max_len=512):
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")
    meta_input = Input(shape=(16,), dtype=tf.float32, name = 'meta_input')
    pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    clf_output = pooled_output#sequence_output[:, 0, :]
    clf_output = concatenate([clf_output,meta_input])
    out = Dense(60, activation='sigmoid',kernel_regularizer= l1(0.01),activity_regularizer= l1(0.01))(clf_output)
    out = Dense(1, activation = 'sigmoid',activity_regularizer= l1(0.01),kernel_regularizer= l1(0.01))(out)
    model = Model(inputs=[input_word_ids, input_mask, segment_ids, meta_input], outputs=out)
    model.compile(Adam(lr=2e-6), loss=binary_crossentropy, metrics=['acc',f1_m,precision_m, recall_m])
    
    return model

module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1"
bert_layer = hub.KerasLayer(module_url, trainable=True)

path2="D:\\Google Drive\\Kaggle_Comp\\nlp_getting_started"
os.chdir(path2)

combined_processed = pd.read_pickle('combined_processed.pickle')

train = combined_processed[0:7613]
test = combined_processed[7613:len(combined_processed)]

meta_train = train.iloc[:,0:20]
meta_test = test.iloc[:,0:20]
meta_train = meta_train.drop(columns=['target','text', 'text_cleaned','id'])
meta_test = meta_test.drop(columns=['target','text', 'text_cleaned','id'])
meta_train['keyword'] = meta_train.keyword.isna().astype(float)
meta_train['location'] = meta_train.location.isna().astype(float)
meta_train["capital_lower_ration"] = meta_train.capital_count.div(1+meta_train.lower_count)

meta_test['keyword'] = meta_test.keyword.isna().astype(float)
meta_test['location'] = meta_test.location.isna().astype(float)
meta_test["capital_lower_ration"] = meta_test.capital_count.div(1+meta_test.lower_count)


meta_train = meta_train.astype(float)
meta_test = meta_test.astype(float)


train = train[["target","text_cleaned"]]
test = test["text_cleaned"]

vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

train_input = bert_encode(train.text_cleaned.values, tokenizer, max_len=160)
test_input = bert_encode(test.values, tokenizer, max_len=160)
train_labels = train.target.values

train_input = train_input + (meta_train.to_numpy(),)
test_input = test_input + (meta_test.to_numpy(),)


model = build_model(bert_layer, max_len=160)
model.summary()


train_history = model.fit(
    train_input, train_labels,
    validation_split=0.1,
    epochs=1,
    batch_size=16
)

y_pred2 = model.predict(test_input)

path2="D:\\Google Drive\\Kaggle_Comp\\nlp_getting_started"
os.chdir(path2)

submission = pd.read_csv("sample_submission.csv")
submission["target"] = np.rint(y_pred2).astype(int)

os.chdir(path)

submission.to_csv("results_0217_12.csv", index=False)
submission["target"] = y_pred2
submission.to_csv("results_0217_12_prop.csv", index=False)

