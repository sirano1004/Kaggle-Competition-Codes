# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 21:46:35 2020

@author: Gyu
"""
import os
import pandas as pd
import numpy as np
import pickle

path="D:\\Google Drive\\Kaggle_Comp\\nlp_getting_started"

#1) Importing data
train_pd = pd.read_csv("train.csv") 
test_pd = pd.read_csv("test.csv") 

#################################################################################################################
#train_processed = pd.read_pickle('train_processed.pickle')
#test_processed = pd.read_pickle('test_processed.pickle')
os.chdir(path)

combined_processed = pd.read_pickle('combined_processed_base.pkl')
feat = pd.read_pickle('pooled_embedding_result.pkl')
#feat = pd.DataFrame(feat)
#feat.to_pickle('cls_embedding_result.pkl')
combined_processed = pd.concat([combined_processed,feat], axis = 1)
keyword = combined_processed.keyword.isna()
location = combined_processed.location.isna()
combined_processed["keyword"] = keyword
combined_processed["location"] = location


train_processed = combined_processed[0:len(train_pd)]
test_processed = combined_processed[len(train_pd):len(combined_processed)] 
y_train = train_processed["target"]
train_processed = train_processed.drop(columns = ["id", "target","text","text_cleaned"])
test_processed = test_processed.drop(columns = ["id", "target","text","text_cleaned"])
train_processed = train_processed.astype(float)
test_processed = test_processed.astype(float)




# capital lower ratio
train_processed["capital_lower_ration"] = train_processed.capital_count.div(1+train_processed.lower_count)
test_processed["capital_lower_ration"] = test_processed.capital_count.div(1+test_processed.lower_count)





## NN

import pandas
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.regularizers import l1
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras import backend as K
from keras.preprocessing import pad_sequences

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



#fit the model
model = Sequential()
model.add(Dense(70, input_dim=1040, activation='relu', kernel_regularizer = l1(0.01)))
model.add(Dense(1, activation='sigmoid',activity_regularizer = l1(0.01)))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc',f1_m,precision_m, recall_m])
model.fit(train_processed, y_train, epochs=2, batch_size=32, shuffle=True, validation_split=0.10,verbose = 2)

y_pred2 = model.predict(test_processed)

submission = pd.read_csv("sample_submission.csv")
submission["target"] = np.rint(y_pred2).astype(int)
submission.to_csv("results_0217_3.csv", index=False)

# 2.2) Tokeninze and pad
# tokenize
combined_texts = combined_processed["tokened"].astype(str)
tokenize = keras.preprocessing.text.Tokenizer(num_words=None, split=' ', char_level = False)
tokenize.fit_on_texts(combined_texts)
words_size = len(tokenize.word_index) +1

# encode to integer
combined_int = tokenize.texts_to_sequences(combined_texts)

# pad the vectors to equal length
maximumlength = 900
#maximumfeature = 20000
combined_pad = pad_sequences(combined_int, maxlen = maximumlength, padding ='post')
train_padded = combined_pad[0:7613]
test_padded = combined_pad[7613:len(combined_pad)]

# 2.3) Pretrained word embeddings
#1)  FASTEXT 
# Loading
#embeddings_index = dict()
#fastext = open('wiki-news-300d-1M.vec', encoding = "utf8")
#for obs in fastext[1:len(fastext)]:
#    values = obs.split(' ')
#    vocab = values[0]
#    coefficients = np.asarray(values[1:], dtype = 'float32')
#    embeddings_index[vocab] = coefficients
#fastext.close()

import io
def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.asarray(tokens[1:], dtype = 'float32')
    return data

fastext = load_vectors('wiki_news_300d_1M.vec')


twt = "glove.twitter.27B.200d.txt"
def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
twitter = dict(get_coefs(*o.strip().split()) for o in open(twt,encoding='utf-8'))

# embedding matrix
embeddings_matrix = np.zeros((words_size, 501))

for word, index in tokenize.word_index.items():
  #  if index >= maximumfeature:
   #     continue
    embedding_vector_array_ft = fastext.get(word)
    embedding_vector_array_twt = twitter.get(word)
    if fastext.get(word) is not None:
        if twitter.get(word) is not None:
            embedding_vector_array_twt = np.concatenate((embedding_vector_array_twt, np.array([1])))
            embeddings_matrix[index] = np.concatenate((embedding_vector_array_ft, embedding_vector_array_twt)) 
        else:
            embeddings_matrix[index] = np.concatenate((embedding_vector_array_ft, np.zeros(201)))
        
        # 355,931 * 500    
file = open('embeddings_save_1206', 'wb')
pickle.dump(embeddings_matrix, file)

# 3) Modelling
import os
path = "C:\\Users\\Gyu\\Google Drive\\Kaggle\\CheckPoint2\\data"
os.chdir(path)
import pickle
file2 = open('embeddings_save_1206', 'rb')
embeddings_matrix = pickle.load(file2)
words_size = 28796 #355931
embed_size = 501

from keras.layers import Embedding
from keras.models import Model
from keras.layers import SpatialDropout1D
from keras import optimizers
from keras.layers import Bidirectional, Input, Dense
from keras.layers import CuDNNLSTM, CuDNNGRU, GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate

#from sklearn.model_selection import train_test_split
#from sklearn.metrics import roc_auc_score
# get y
#import os
#path="C:\\Users\\Gyu\\Google Drive\\Kaggle\\CheckPoint1\\data\\original"
#os.chdir(path)
#train_pd = pd.read_csv("train.csv") 
#test_pd = pd.read_csv("test.csv") 
y_train = train_pd["target"]

#get x
path2="C:\\Users\\Gyu\\Google Drive\\Kaggle\\CheckPoint2\\data"
os.chdir(path2)
combined_processed = pd.read_pickle('combined_processed.pickle')
combined_texts = combined_processed["processed"].astype(str)
tokenize = keras.preprocessing.text.Tokenizer(num_words=None, split=' ', char_level = False)
tokenize.fit_on_texts(combined_texts)
words_size = len(tokenize.word_index) +1
combined_int = tokenize.texts_to_sequences(combined_texts)
maximumlength = 900
combined_pad = pad_sequences(combined_int, maxlen = maximumlength, padding ='post')
train_padded = combined_pad[0:159571]
test_padded = combined_pad[159571:len(combined_pad)]

#### Modelling
inp = Input(shape=(maximumlength, )) 
x = Embedding(words_size, embed_size, weights=[embeddings_matrix], input_length=maximumlength, trainable=False)(inp)
x = SpatialDropout1D(0.5)(x)
x = Bidirectional(CuDNNLSTM(40,return_sequences=True))(x)
x = Bidirectional(CuDNNGRU(40,return_sequences=True))(x) 

average_pool = GlobalAveragePooling1D()(x)
maximum_pool = GlobalMaxPooling1D()(x)
out = concatenate([average_pool, maximum_pool])

out = Dense(200, activation="relu")(out)
out = Dense(1, activation="sigmoid")(out)

model = Model(inputs=inp, outputs=out)
adam = optimizers.adam(clipvalue=0.5)
model.compile(loss='binary_crossentropy', optimizer = adam, metrics=['accuracy'])

model.fit(train_padded, y_train, epochs=15, batch_size=256, shuffle=True, validation_split=0.1,verbose = 2)
y_pred = model.predict(test_padded)

submission = pd.read_csv("sample_submission.csv")
submission[["id"]] = test_pd[["id"]]
submission["target"] = y_pred
submission.to_csv("results_1207.csv", index=False)


# Ensemble##############################################################################################
model1 = pd.read_csv("results_1206.csv")
model2 = pd.read_csv("results_1207.csv")
# Random Forest
model3 = pd.read_csv("result_rf.csv")


model1 = model1[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].multiply(0.45)
model2 = model2[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].multiply(0.45)
model3 = model3[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].multiply(0.1)

temp = model1.add(model2, fill_value = 0)
temp2 = temp.add(model3, fill_value = 0)



#model12 = model1.add(model2, fill_value=0)
#model12 = model12[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].div(int(2))

submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = temp2[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]]
submission.to_csv("results_5.csv", index=False)



