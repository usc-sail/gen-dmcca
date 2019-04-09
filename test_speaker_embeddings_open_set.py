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
for s_ in seeds_[:1]:
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
    
    # data generator
    def test_data_generator_one_branch(data_rcycle, batch_size, to_label_map):
        while True:
            # pick words using islice on top of rcycle
            wav_file_batch = list(islice(data_rcycle, batch_size))
            label_list = [to_label_map[os.path.basename(i).split('_')[0]] for i in wav_file_batch]
            wav_batch = WavLoad_OneBranch(wav_file_batch)
            yield (wav_batch, wav_file_batch, label_list)

    
    def evaluate_model(model, test_generator, N_test_steps, N,
                       all_test_samples_list):
         # create a model with only one branch out of N
        pick_layers = range(0, len(model.layers)-1, N)
        for n_ in range(N):
            print n_
            model_part = Sequential()
            for i in pick_layers: model_part.add(model.layers[i + n_])
            #model_part.summary()
            X_emb_all = []
            y_all = []
            subs_all = []
            l_all = []
            for n_iter in tqdm(range(N_test_steps)):
                X_t, l_, y_t = test_generator.next()
                X_emb_ = model_part.predict(X_t) 
                X_emb_all.append(X_emb_)
                y_all += y_t
                l_all += l_
                subs_all += [test_sub_dict[i] for i in y_t]
            remaining_test = list(set(all_test_samples_list).difference(set(l_all)))
            y_all += [test_sub_to_label_dict[os.path.basename(i).split('_')[0]] for i in remaining_test]
            subs_all += [os.path.basename(i).split('_')[0] for i in
                         remaining_test]
            remaining_X_test = WavLoad_OneBranch(remaining_test)
            X_emb_all.append(model_part.predict(remaining_X_test))
            
            print ''
            X_emb = np.vstack(X_emb_all)
            print 'kmeans -----', len(unique(y_all))
            kmeans = KMeans(n_clusters=len(unique(y_all))).fit(X_emb)
            print "BRANCH %d >>" % n_, metrics.homogeneity_completeness_v_measure(y_all, kmeans.labels_), ' :: SVM :: ',
            svm_results = run_svc_pipeline_doubleCV(X_emb, np.array(y_all), param_search=False)
            print '%.02f $\pm$ %.02f' % (np.mean(svm_results[-1]), np.std(svm_results[-1]))
            tsne_ = TSNE(n_components=2, verbose=1)
            tsne_embeddings = tsne_.fit_transform(X_emb)
            df =  pd.DataFrame(columns=['x', 'y', 'label'])
            df.x = list(tsne_embeddings[:,0])
            df.y = list(tsne_embeddings[:,1])
            df.label = subs_all
            np.savez("SPK_EMB_LWO_LSO_seed_%d/seen_subjects_LSO_LWO_mcca_embeddings_and_svm_branch_%d" % (s_, n_), {'svm': svm_results, 'emb': X_emb, 'labels': y_all})
            df.to_csv("SPK_EMB_LWO_LSO_seed_%d/seen_subjects_LWO_LWO_tsne_embeddings_branch_%d.csv" % (s_, n_))
            #figure()
            #sn.lmplot(data=df, x='x', y='y', hue='label', fit_reg=False, legend=True, legend_out = True, markers=markers, palette=pal)
    

    TEST = True
    if TEST:
        # 1. first test on unseen subjects and unseen words
        #now_test_df = test_df_left_out_words # set the right DF!
        
        # 2. Seen subjects but unseen words
        now_test_df = train_df_left_out_words 

        # below is just a template!
        all_test_samples_list = now_test_df['wav']
        test_left_out_words_rcycle = rcycle(all_test_samples_list)
        test_subs = sorted(set(now_test_df.subject))
        test_sub_to_label_dict = dict(zip(test_subs, range(len(test_subs))))
        test_sub_dict = dict(zip(range(len(test_subs)), test_subs))

        batch_size = 500
        N_test_steps = len(now_test_df)//batch_size
        act_ = 'sigmoid'
        model = create_model(act_ = act_, n_modalities = N)

        best_saved_wt = sorted(glob.glob('SPK_EMB_LWO_LSO_seed_%d/LWO_LSO*hdf5' % s_), key=lambda a : int(a.split('batch.')[-1].split('-')[0]))[-1]
        print("best saved file ==== ", best_saved_wt)
        model.load_weights(best_saved_wt)
        
        
        test_generator = test_data_generator_one_branch(test_left_out_words_rcycle, 
                                       batch_size = batch_size, 
                                       to_label_map = test_sub_to_label_dict)
        evaluate_model(model, test_generator, N_test_steps, N, all_test_samples_list)
        

    
    
    
    
    
