import os
import sys
import glob
import numpy as np
from pylab import *
import pandas as pd
import random
import seaborn as sn
import json
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from itertools import groupby, islice, cycle
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn import datasets, svm, preprocessing 
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import *
from sklearn.metrics import *
sn.set_context('paper')

# decorating plots
def list_markers():
    markers = []
    for m in Line2D.markers:
        try:
            if len(m) == 1 and m != ' ':
                markers.append(m)
        except TypeError:
            pass
    styles = markers + [
        r'$\lambda$',
        r'$\bowtie$',
        r'$\circlearrowleft$',
        r'$\clubsuit$',
        r'$\checkmark$']
    return markers
markers_ = list_markers()
markers = markers_ + markers_[:30-len(markers_)]
pal = sn.color_palette("Set1", n_colors=30, desat=.5)

# SVM cross val
def run_svc_pipeline_doubleCV(X, y, dev_split=5, C_=0.015, n_splits_=10,
                              param_search=False, n_jobs_=18, verbose = False): 
    # use different splits with different random states for CV-param search
    svc = svm.SVC(kernel='linear', C = C_) 
    #svc = svm.LinearSVC(C = C_) 

    pipeline_estimators = [('scale', preprocessing.StandardScaler()), ('svm',
                                                                       svc) ]
    svc_pipeline = Pipeline(pipeline_estimators)

    if param_search:
        if verbose: print('param search is ONNN!')
        C_search = sorted( list(np.logspace(-5,0,10)) + [0.1,5,10,20,50,100] )
        param_grid = dict( scale=[None], svm__C=C_search )

        sk_folds = StratifiedKFold(n_splits=dev_split, shuffle=False,
                                   random_state=1964)
        grid_search = GridSearchCV(svc_pipeline, param_grid=param_grid,
                                   n_jobs=n_jobs_, cv=sk_folds.split(X,y),
                                   verbose=verbose)
        grid_search.fit(X, y)
        # find the best C value
        which_C = np.argmax(grid_search.cv_results_['mean_test_score'])
        best_C = C_search[which_C]
    else:
        best_C = C_

    print('estimated the best C for svm to be', best_C)
    sk_folds = StratifiedKFold(n_splits=n_splits_, shuffle=False, random_state=320)
    all_scores = []
    all_y_test = []
    all_pred = []
    for train_index, test_index in sk_folds.split(X, y):
        if verbose: print 'run -',
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        svc_pipeline = Pipeline(pipeline_estimators)
        svc_pipeline.named_steps['svm'].C = best_C
        
        svc_pipeline.fit(X_train, y_train)
        y_pred = svc_pipeline.predict(X_test)
        score = svc_pipeline.score(X_test, y_test)
        if verbose: print score	
        all_y_test.append(y_test)
        all_pred.append(y_pred)
        all_scores.append(score)
    return all_y_test, all_pred, all_scores

def create_svc_pipeline(X, y, X_test, y_test, dev_split=10, C_=0.015, n_splits_=10,
                              param_search=False, n_jobs_=18, verbose = False): 
    # use different splits with different random states for CV-param search
    svc = svm.SVC(kernel='linear', C = C_) 
    #svc = svm.LinearSVC(C = C_) 

    pipeline_estimators = [('scale', preprocessing.StandardScaler()), ('svm',
                                                                       svc) ]
    svc_pipeline = Pipeline(pipeline_estimators)

    if param_search:
        if verbose: print('param search is ONNN!')
        C_search = sorted( list(np.logspace(-5,0,10)) + [0.1,5,10,20,50,100] )
        param_grid = dict( scale=[None], svm__C=C_search )

        sk_folds = StratifiedKFold(n_splits=dev_split, shuffle=False,
                                   random_state=1964)
        grid_search = GridSearchCV(svc_pipeline, param_grid=param_grid,
                                   n_jobs=n_jobs_, cv=sk_folds.split(X,y),
                                   verbose=verbose)
        grid_search.fit(X, y)
        # find the best C value
        which_C = np.argmax(grid_search.cv_results_['mean_test_score'])
        best_C = C_search[which_C]
    else:
        best_C = C_

    svc_pipeline.named_steps['svm'].C = best_C
    print('estimated the best C for svm to be', best_C)
    # after param search retrain with all data!
    svc_pipeline.fit(X, y)
    y_pred = svc_pipeline.predict(X_test)
    perf_report = classification_report(y_test, y_pred)
    print(perf_report)
    return svc_pipeline, y_test, y_pred, perf_report

def rcycle(iterable):
        #http://davidaventimiglia.com/python_generators.html
        # this is itertools.cycle but shuffle from the second cycle onwards
    saved = []                 # In-memory cache
    for element in iterable:
        yield element
        saved.append(element)
    while saved:
        random.shuffle(saved)  # Shuffle every batch
        for element in saved:
            yield element

# read wavfile - and rearrange (n_modalities, batch_size, channel, frames, num_mels)
WavLoad = lambda f: \
            np.asarray([[np.load(j.replace('.wav', '.npy')) for j in i] for i in f]).swapaxes(0,1)
            #np.asarray( [[wavfile_to_examples(j) for j in i] for i in f] ).swapaxes(0,1)

# read wavfile - in form (batch_size, channel, frames, num_mels)
WavLoad_OneBranch = lambda f: \
            np.asarray([ np.load(j.replace('.wav', '.npy')) for j in f ])


