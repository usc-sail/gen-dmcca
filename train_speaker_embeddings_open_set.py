'''
@KS - leave 15 words out! - extreme
'''

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
import itertools
from itertools import groupby, islice, cycle
from sklearn.preprocessing import LabelEncoder
sys.path.insert(0, 'audioset_scripts')
from vggish_input import *
from models import *
from keras.callbacks import *
from sklearn.cluster import KMeans
from sklearn import metrics
from utils import *
from speech_commands_utils import *
from collections import Counter

# load subjects/words info!
df, df_sub, df_word = get_speech_commands_df()

# bunch of things to tune
subs_set = sorted(set(df.subject))
words_set = sorted(set(df.word))

np.random.seed(320)
seeds_ = random_integers(1000, 10000, size=10)

N_leave_out = 150
N = 3
batch_size = 100

# leave out half the words!
words_to_leave_out = words_set[:15]
words_to_keep = words_set[15:]

# choose about 10 different N for test subjects
# number of folds
for s_ in seeds_[1:]:
    np.random.seed(s_)
    leave_out_subjects = np.random.choice(len(subs_set), N_leave_out, replace=False)
    test_subs_ = np.array(subs_set)[leave_out_subjects]
    
    # pick out the subs
    # need atleast 3*5 = 15 repetitions per subject
    N_min_words_per_subj = 15 # each char. has atleast some utterances
    train_subs_ = [i for i in subs_set if i not in test_subs_]
    train_df_ = df[df['subject'].isin(train_subs_)]
    samples_per_dict = Counter(train_df_.subject)
    train_subs = [i for i in train_subs_ if samples_per_dict[i] > N_min_words_per_subj]
    train_df_all_words = df[df['subject'].isin(train_subs)]
    # only half of the words
    train_df = train_df_all_words[train_df_all_words['word'].isin(words_to_keep)]
    train_df_left_out_words = train_df_all_words[train_df_all_words['word'].isin(words_to_leave_out)]

    v = Counter(train_df.subject).values()
    
    # do train-test df
    test_df_ = df[df['subject'].isin(test_subs_)]
    test_df = test_df_[test_df_['word'].isin(words_to_keep)]
    test_subs = sorted(set(test_df['subject']))
    test_df_left_out_words = test_df_[test_df_['word'].isin(words_to_leave_out)]

    print s_, len(train_df_), len(test_df_), len(train_df), len(test_df), min(v), mean(v), max(v)

    N_train = len(train_df)
    N_test = len(test_df)
    N_subs_train = len(set(train_df.subject))
    N_subs_test = len(test_subs)
    
    # pick the number of modalities - N
    N_class = len(words_set)
    N_subs = len(subs_set)

    # 1. the N modalities are each different words
    # 2. the N modalities are matched for the same subject - since the task is
    # to classify the subj

    train_df_subject_gp = train_df.groupby(['subject'])
    test_df_subject_gp = test_df.groupby(['subject'])

    # train samples
    print('---- training data -----')
    train_samples = [list(train_df_subject_gp.get_group(i).wav) for i in
                     train_subs]
    print([(i, len(train_samples[ix])) for ix,i in enumerate(train_subs)])

    print('---- test data -----')
    test_samples = [list(test_df_subject_gp.get_group(i).wav) for i in test_subs]
    print([(i, len(test_samples[ix])) for ix,i in enumerate(test_subs)])

    out_dir = './SPK_EMB_LWO_LSO_seed_%s/' % s_
    if not os.path.isdir(out_dir): os.makedirs(out_dir)
    
    # create rcycle list - we need congruent samples for all modalities
    train_rcycle = [rcycle(k) for k_ix, k in enumerate(train_samples)]
    
    # create rcycle for test
    test_rcycle = [rcycle(k) for k_ix,k in enumerate(test_samples)]

    # data generator
    def train_speech_data_generator(batch_size = 100, n_modalities = N):
        while True:
            # pick 100 (batch size) indices with replacement on the 30 (n_class) words
            subj_indices = np.random.choice(N_subs_train, batch_size,
                                            replace = False)
            # pick words using islice on top of rcycle
            wav_file_batch = [list(islice(train_rcycle[i], n_modalities)) for i
                              in subj_indices]
            labels_batch = [train_subs[i] for i in subj_indices]
            wav_batch = list(WavLoad(wav_file_batch))
            yield (wav_batch, np.zeros(batch_size))

    def test_speech_data_generator(batch_size = 100, n_modalities = N):
        while True:
            # pick 100 (batch size) indices with replacement on the 30 (n_class) words
            subj_indices = np.random.choice(N_subs_test, batch_size,
                                            replace = False)
            # pick words using islice on top of rcycle
            wav_file_batch = [list(islice(test_rcycle[i], n_modalities)) for i
                              in subj_indices]
            wav_batch = list(WavLoad(wav_file_batch))
            yield (wav_batch, subj_indices)


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

            checkpointer = ModelCheckpoint(filepath=out_dir + "LWO_LSO_seed-%s_02_02_SGD_act_%s_weights_%d-modal_%d-batch.{epoch:02d}-{val_loss:.4f}.hdf5" % (s_, act_, N, batch_size), verbose=1, save_best_only=True, save_weights_only=True)
            early_stopping = EarlyStopping(min_delta = 1e-4, patience = 5)
            csv_logger = CSVLogger(out_dir + 'LWO_LSO_seed-%s_SGD_log_02_02_2019_act_%s.csv' % (s_, act_), append=True, separator=';') 
            callbacks_ = [checkpointer, early_stopping, csv_logger]
    # train_model!
            model.fit_generator(train_generator, steps_per_epoch = N_steps, epochs = N_epochs, shuffle = True, validation_data = test_generator, validation_steps = N_test_steps, callbacks = callbacks_, verbose = 1)

