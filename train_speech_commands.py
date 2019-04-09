import os
import sys
import glob
import commands
import time
import numpy as np
from pylab import *
import pandas as pd
import random
import json
from itertools import groupby, islice, cycle
from sklearn.preprocessing import LabelEncoder
sys.path.insert(0, 'audioset_scripts')
from vggish_input import *
from models import *
from keras.callbacks import *
from sklearn.cluster import KMeans
from sklearn import metrics
from speech_commands_utils import *


df, df_sub, df_word = get_speech_commands_df()
# bunch of things to tune
subs_set = sorted(set(subs))
words_set = sorted(set(words))

# TUNING PARAMS
# pick the number of modalities - N
N = 3
batch_size = 100
n_class = len(words_set)
n_subs = len(subs_set)

# 1. the N modalities are each different subjects
# 2. the N modalities are matched for the same word - since the task is to classify the word

# choose about 10 different N for test subjects
# number of folds
test_folds = 20
np.random.seed(320)
test_ix = np.random.choice(len(subs_set), N*test_folds, replace=False).tolist()
test_subs = np.array(subs_set)[test_ix]
train_subs = [i for i in subs_set if i not in test_subs]

# get the samples
train_df = df[df['subject'].isin(train_subs)]
test_df = df[df['subject'].isin(test_subs)]

train_df_word_gp = train_df.groupby(['word'])
test_df_word_gp = test_df.groupby(['word'])

# train samples
print('---- training data -----')
train_samples = [list(train_df_word_gp.get_group(i).wav) for i in words_set]
print([(i, len(train_samples[ix])) for ix,i in enumerate(words_set)])

print('---- test data -----')
test_samples = [list(test_df_word_gp.get_group(i).wav) for i in words_set]
print([(i, len(test_samples[ix])) for ix,i in enumerate(words_set)])

# create rcycle list - we need congruent samples for all modalities
train_rcycle = [rcycle(k) for k in train_samples]
test_rcycle = [rcycle(k) for k in test_samples]

# data generator
def train_speech_data_generator(batch_size = 100, n_modalities = N):
    while True:
        # pick 100 (batch size) indices with replacement on the 30 (n_class) words
        word_indices = np.random.choice(n_class, batch_size, replace=True)
        # pick words using islice on top of rcycle
        wav_file_batch = [list(islice(train_rcycle[i], n_modalities)) for i in word_indices]
        labels_batch = [words_set[i] for i in word_indices]
        wav_batch = list(WavLoad(wav_file_batch))
        yield (wav_batch, np.zeros(batch_size))

def test_speech_data_generator(batch_size = 100, n_modalities = N):
    while True:
        # pick 100 (batch size) indices with replacement on the 30 (n_class) words
        word_indices = np.random.choice(n_class, batch_size, replace=True)
        # pick words using islice on top of rcycle
        wav_file_batch = [list(islice(test_rcycle[i], n_modalities)) for i in word_indices]
        wav_batch = list(WavLoad(wav_file_batch))
        yield (wav_batch, word_indices)

N_train = len(train_df)
N_steps = N_train//batch_size
N_epochs = 100
N_test_steps = len(test_df)//batch_size

TRAIN = True
TEST = False

train_generator = train_speech_data_generator()

if TRAIN:
    test_generator = test_speech_data_generator() #batch_size = len(test_df)//N, n_modalities = N)
# train the model
# create some callbacks!
    for act_ in ['sigmoid']:
# create model
        print act_, '-------------------------------------------------------'
        model = create_model(act_ = act_, n_modalities = N)
        model.summary()

        checkpointer = ModelCheckpoint(filepath="01_17_SGD_act_%s_weights_%d-modal_%d-batch.{epoch:02d}-{val_loss:.4f}.hdf5" % (act_, N, batch_size), verbose=1, save_best_only=True, save_weights_only=True)
        early_stopping = EarlyStopping(min_delta = 1e-4, patience = 5)  
        csv_logger = CSVLogger('SGD_log_01_17_2019_act_%s.csv' % (act_), append=True, separator=';') 
        callbacks_ = [checkpointer, early_stopping, csv_logger]
# train_model!
        model.fit_generator(train_generator, steps_per_epoch = N_steps, epochs = N_epochs, shuffle = True, validation_data = test_generator, validation_steps = N_test_steps, callbacks = callbacks_, verbose = 1)


if TEST:
    act_ = 'sigmoid'
    model = create_model(act_ = act_, n_modalities = N)
    test_generator = test_speech_data_generator(batch_size = len(test_df)//N, n_modalities = N)
    X_test, y_test = test_generator.next()
    model.load_weights('SGD_act_sigmoid_weights_3-modal_100-batch.50--0.2001.hdf5')
    # create a model with only one branch out of N
    pick_layers = range(0, len(model.layers)-1, N)
    model_part = Sequential()
    for i in pick_layers: model_part.add(model.layers[i])
    model_part.summary()
    for x1 in X_test:
        x1_emb = model_part.predict(x1)
        kmeans = KMeans(n_clusters=30).fit(x1_emb)
        print metrics.homogeneity_score(y_test, kmeans.labels_)

