# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 23:00:09 2021

@author: Gyu
"""
import os
file_dir = 'D:\Google Drive\Kaggle_Comp\jane_street'
os.chdir(file_dir)

TRAINING = True

import warnings
warnings.filterwarnings('ignore')

#if TRAINING:
#    import cudf
#    import cupy as cp

import os, gc
import pandas as pd
import numpy as np
from hyperopt import hp, fmin, tpe, Trials
from hyperopt.pyll.base import scope
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from joblib import dump, load

import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Concatenate, Lambda, GaussianNoise, Activation, multiply, add, Multiply
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.metrics import AUC, Accuracy, CategoricalAccuracy

import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection._split import _BaseKFold, indexable, _num_samples
from sklearn.utils.validation import _deprecate_positional_args


import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection._split import _BaseKFold, indexable, _num_samples
from sklearn.utils.validation import _deprecate_positional_args
import kerastuner as kt


TARGET = 'action'
FEATS = ['feature_{}'.format(int(i)) for i in range(130)]

if TRAINING:
    print('Loading...')
    #train = cudf.read_csv('/kaggle/input/jane-street-market-prediction/train.csv')
    train = pd.read_csv('train.csv')
    TARGET = 'action'
    FEATS = ['feature_{}'.format(int(i)) for i in range(130)]

    print('Filling...')
    #train = train.query('date > 85').reset_index(drop = True)
    train = train.query('weight > 0').reset_index(drop = True)
    #train['action'] = (train['resp'] > 0).astype('int')
    train['action'] =  (  (train['resp_1'] > 0.0000 ) & (train['resp_2'] > 0.0000 ) & (train['resp_3'] > 0.0000 ) & (train['resp_4'] > 0.0000 ) &  (train['resp'] > 0.0000 )   ).astype('int')
    resp_cols = ['resp_1', 'resp_2', 'resp_3', 'resp', 'resp_4']
    #train = train.to_pandas()
    y = np.stack([(train[c] > 0.00000).astype('int') for c in resp_cols]).T #Multitarget
    wr = train.weight*train['resp'].to_numpy()
    Y = wr*y[:,3]
    X = train[FEATS].to_numpy()
    date = train['date'].values
    print('Finish.')
    
nan_feat = (train[FEATS].isnull().sum()>0)
NAN_FEAT = nan_feat[nan_feat == True].index
nan_feat_bool = nan_feat.values


if TRAINING:
    f_mean = train[FEATS].median().values
    mask = np.isnan(X).astype(int)
    X_out = np.nan_to_num(X)
    
    mask2 = np.isnan(X[:,nan_feat_bool]).astype(int)
    X[:, nan_feat_bool] = X_out[:, nan_feat_bool] + mask2 * f_mean[nan_feat_bool]
    #count_nan = np.sum(mask,axis = 1).reshape(-1,1)
    #del(train)
    #_= gc.collect()
    with open('f_mean.npy', 'wb') as f:
        np.save(f, f_mean)
        
    #X_stat = np.absolute(X[:,1:] - f_mean[1:])
    #X = np.concatenate((X, X_stat, mask, count_nan), axis = 1)

def custom_loss(y_true, y_pred):
    return 100 * tf.keras.losses.MSE(y_true,y_pred)

def metrics2(y_true, y_pred):
    return K.sum(y_pred)

def metrics(y_true, y_pred):
    Pi = np.bincount(y_true, y_pred)
    t = np.sum(Pi) / np.sqrt(np.sum(Pi ** 2)) * np.sqrt(250 / len(Pi))
    u = min(max(t, 0), 6) * np.sum(Pi)
    #print('\n', round(u,5))
    return u


def create_autoencoder(input_dim,output_dim,noise=0.1):
    i = Input(130)
    mask = Input(130)
    encoded = BatchNormalization()(i)
    encoded = GaussianNoise(noise)(encoded)
    
    encoded = Dense(96, activation = 'elu')(encoded)
    encoded = Dense(64,activation='linear')(encoded)
    decoded = Dense(96, activation = 'elu')(encoded)
    decoded = Dense(input_dim)(decoded)
    masked_decoded = multiply([mask, decoded], name = 'decoded')

    input2 = Input(88)
    
    x = Concatenate()([i, encoded, input2])
    x = Dense(512)(x)
    x = BatchNormalization()(x)
    x = Lambda(tf.keras.activations.swish)(x)
    x = Dropout(0.5)(x)    
    
    x = Dense(512)(x)
    x = BatchNormalization()(x)
    x = Lambda(tf.keras.activations.swish)(x)
    x = Dropout(0.5)(x)
    
    x = Dense(300)(x)
    x = BatchNormalization()(x)
    x = Lambda(tf.keras.activations.swish)(x)
    x = Dropout(0.42)(x)
    
    
    x = Dense(64)(x)
    x = BatchNormalization()(x)
    x = Lambda(tf.keras.activations.swish)(x)
    x = Dropout(0)(x)
    
    x = Dense(output_dim, activation='sigmoid', name='label_output')(x)
    x2 = tf.math.reduce_mean(x, axis = -1)
    x2 = tf.where(x2>0.5, 1, 0)
    x2 = tf.cast(x2, tf.float32)
    wr = Input(1)
    x2 = Multiply(name = 'return_out')([x2,wr])    
    encoder = Model(inputs=i,outputs=encoded)
    
    autoencoder = Model(inputs=[i, mask, input2, wr],outputs=[masked_decoded, x, x2])
    autoencoder.compile(optimizer=Adam(0.0005),loss={'decoded':'mse', 'label_output':BinaryCrossentropy(label_smoothing = 0.0845), 'return_out':custom_loss}, metrics = {'label_output': AUC(name = 'auc'), 'return_out': metrics2})
    return autoencoder, encoder

autoencoder, encoder = create_autoencoder(130, 5, noise=0.1)
    
if TRAINING:
    #autoencoder.compile(optimizer='adam',loss={'decoded':'mse', 'label_output':BinaryCrossentropy(label_smoothing = 0.0845)}, metrics = {'label_output': AUC(name = 'auc')})
    X_train, X_test = X[:1193206], X[1268962:]
    y_train, y_test = y[:1193206], y[1268962:]
    Y_train, Y_test = Y[:1193206], Y[1268962:]
    mask_train, mask_test = mask[:1193206], mask[1268962:]
    X_train_out, X_test_out = X_out[:1193206], X_out[1268962:]
    mask2_train, mask2_test = mask2[:1193206], mask2[1268962:]
    wr_train, wr_test = wr[:1193206], wr[1268962:]
    autoencoder.fit([X_train, 1 - mask_train, mask2_train, wr_train],[X_train_out, y_train, Y_train],
                    validation_data = ([X_test, 1 - mask_test, mask2_test, wr_test],[X_test_out, y_test, Y_test]),
                    epochs=1002,
                    batch_size=4000, 
                    callbacks=[EarlyStopping('val_loss',patience=10,restore_best_weights=True)])
    encoder.save_weights('./encoder.hdf5')
else:
    encoder.load_weights('encoder.hdf5')
encoder.trainable = False