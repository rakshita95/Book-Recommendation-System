# -*- coding: utf-8 -*-
"""
Created on Sun Nov 05 22:00:03 2017

@author: Deepak Maran
"""
from __future__ import division

'''
Collected by Cai-Nicolas Ziegler in a 4-week crawl (August / September 2004) from
the Book-Crossing community. Contains 278,858 users (anonymized but with demographic 
information) providing 1,149,780 ratings (explicit / implicit) about 271,379 books.
'''
import pandas as pd
import numpy as np
import os
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
import surprise
from collections import defaultdict
import random
'''
BX-Users
Contains the users. Note that user IDs (`User-ID`) have been anonymized 
and map to integers. Demographic data is provided (`Location`, `Age`) if 
available. Otherwise, these fields contain NULL-values.

BX-Books
Books are identified by their respective ISBN. Invalid ISBNs have already 
been removed from the dataset. Moreover, some content-based information is
given (`Book-Title`, `Book-Author`, `Year-Of-Publication`, `Publisher`), 
obtained from Amazon Web Services. Note that in case of several authors, 
only the first is provided. URLs linking to cover images are also given, 
appearing in three different flavours (`Image-URL-S`, `Image-URL-M`, 
`Image-URL-L`), i.e., small, medium, large. These URLs point to the Amazon 
web site.

BX-Book-Ratings
Contains the book rating information. Ratings (`Book-Rating`) are either
 explicit, expressed on a scale from 1-10 (higher values denoting higher 
 appreciation), or implicit, expressed by 0.
'''

## User defined
def ndcg_at_k(predictions, k=10):
    dcgs = dict()
    idcgs = dict()
    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))
        
    for uid, user_ratings in user_est_true.items():
        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        
        #estimated rank
        rank_est = np.arange(1, len(user_ratings[:k])+1)
        discount_est = np.log2(rank_est+1)
        
        #Relevance 
        rel = [np.power(2,true_r)-1 for (_, true_r) in user_ratings[:k]]
        
        dcgs[uid] = sum(rel/discount_est)
        
        # Sort user ratings by true value
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        
        #estimated rank
        rank_true = np.arange(1, len(user_ratings[:k])+1)
        discount_true = np.log2(rank_true+1)
        
        #Relevance 
        rel_true = [np.power(2,true_r)-1 for (_, true_r) in user_ratings[:k]]
        
        idcgs[uid] = sum(rel_true/discount_true)
        
    dcg = sum(dcgu for (_,dcgu) in dcgs.items())
    idcg = sum(idcgu for (_,idcgu) in idcgs.items())
    return dcg/idcg

def ap_at_k(predictions, k=10, threshold=7):

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    aps = dict()
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        
        y_score = []
        y_true = []
        rec_items = user_ratings[:k]
        for item in rec_items:
            if item[0] >= threshold:
                y_score.append(1)
            else:
                y_score.append(0)
            
            if item[1] >= threshold:
                y_true.append(1)
            else:
                y_true.append(0)
            
        ap = average_precision_score(y_true, y_score)
        aps[uid] = ap
    
    return aps

def get_top_n(predictions, n=10):
    '''Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    '''

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

def precision_recall_at_k(predictions, k=10, threshold=7):
    '''Return precision and recall at k metrics for each user.'''

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        # Precision@K: Proportion of recommended items that are relevant
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1

        # Recall@K: Proportion of relevant items that are recommended
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

    return precisions, recalls

## Input variables depending on domain experience
# Threshold value: Recommend if ratings are above this value
thresh = 7
# Maximum number of recommendations (top-N)
N_rec = 10
# Values of k for tuning in cross-validation
ks = [1,3,5,7,9,12,20]

train = pd.read_csv('train.csv')
reader = surprise.Reader(rating_scale=(1, 10))
data = surprise.Dataset.load_from_df(train[['User-ID', 'ISBN', 'Book-Rating']], reader)
random.seed()
data.split(n_folds=4)




####################################################################
###### Best kNN Algorithm: kNN Basic, Item-item, Pearson similarity
####################################################################

#The following combinations were tried to find the best kNN algorithm:
#kNN basic, with offset by mean, with Z-score
#Similarity: Cosine, Pearson, Pearson baseline
#Item and user-based similarities

# Tuning hyper-parameter: k (number of neighbors)
sim_options = {'name': 'pearson',
       'user_based': False  # compute  similarities between items
       }
mean_ap = []
precision = []
recall = []
fscore = []
normalized_DCG = []
mean_ap_train = []
precision_train = []
recall_train = []
fscore_train = []
normalized_DCG_train = []

for k_val in ks:
    print(k_val)
    algo = surprise.KNNBasic(k=k_val, sim_options=sim_options)
    pr = 0
    re = 0
    fs = 0
    ap = 0
    nd = 0
    pr_train = 0
    re_train = 0
    fs_train = 0
    ap_train = 0
    nd_train = 0
    for trainset, testset in data.folds():
        algo.train(trainset)
        predictions_on_test = algo.test(testset)
    
        precisions_test, recalls_test = precision_recall_at_k(predictions_on_test, k=N_rec, threshold=thresh)
        aps = ap_at_k(predictions_on_test, k=N_rec, threshold=thresh)
        nDCG = ndcg_at_k(predictions_on_test,k=N_rec)
        p = sum(prec for prec in precisions_test.values()) / len(precisions_test)
        r = sum(rec for rec in recalls_test.values()) / len(recalls_test)
        f_score = 2/(1/p+1/r)        
        pr = pr + p
        re = re + r
        fs = fs + f_score
        nd = nd + nDCG
        ap = ap + np.nanmean(aps.values())
        
        trainset_to_test = trainset.build_testset()
        predictions_on_train = algo.test(trainset_to_test)
        precisions_train, recalls_train = precision_recall_at_k(predictions_on_train, k=N_rec, threshold=thresh)
    
        aps_train = ap_at_k(predictions_on_train, k=N_rec, threshold=thresh)
        nDCG_train = ndcg_at_k(predictions_on_train,k=N_rec)
        p_train = sum(prec for prec in precisions_train.values()) / len(precisions_train)
        r_train = sum(rec for rec in recalls_train.values()) / len(recalls_train)
        f_score_train = 2/(1/p_train+1/r_train)        
        pr_train = pr_train + p_train
        re_train = re_train + r_train
        fs_train = fs_train + f_score_train
        nd_train = nd_train + nDCG_train
        ap_train = ap_train + np.nanmean(aps_train.values())        

    precision.append(pr/4)
    recall.append(re/4)
    fscore.append(fs/4)
    mean_ap.append(ap/4) # MAP
    normalized_DCG.append(nd/4) # NDCG

    precision_train.append(pr_train/4)
    recall_train.append(re_train/4)
    fscore_train.append(fs_train/4)
    mean_ap_train.append(ap_train/4) # MAP
    normalized_DCG_train.append(nd_train/4) # NDCG

plt.figure(1)

plt.subplot(211)
B_P_N, = plt.plot(ks, normalized_DCG, marker = 'o', linestyle = '-', color = 'b')
#plt.plot(ks, normalized_DCG_train, 'bo')
plt.ylabel('NDCG')
plt.xlabel('k')

plt.subplot(212)
B_P_F, = plt.plot(ks, fscore, marker = 'o', linestyle = '-', color = 'b')
#plt.plot(ks, fscore_train, 'bo')
plt.ylabel('F-score')
plt.xlabel('k')



#############################
############ kNN Exploration
#############################

###### kNN Baseline

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 22}

plt.rc('font', **font)


sim_options = {'name': 'pearson_baseline',
       'user_based': False  # compute  similarities between items
       }

mean_ap = []
precision = []
recall = []
fscore = []
normalized_DCG = []
mean_ap_train = []
precision_train = []
recall_train = []
fscore_train = []
normalized_DCG_train = []

for k_val in ks:
    print(k_val)
    algo = surprise.KNNBaseline(k=k_val,sim_options=sim_options)
    pr = 0
    re = 0
    fs = 0
    ap = 0
    nd = 0
    pr_train = 0
    re_train = 0
    fs_train = 0
    ap_train = 0
    nd_train = 0
    for trainset, testset in data.folds():
        algo.train(trainset)
        predictions_on_test = algo.test(testset)
    
        precisions_test, recalls_test = precision_recall_at_k(predictions_on_test, k=N_rec, threshold=thresh)
        aps = ap_at_k(predictions_on_test, k=N_rec, threshold=thresh)
        nDCG = ndcg_at_k(predictions_on_test,k=N_rec)
        p = sum(prec for prec in precisions_test.values()) / len(precisions_test)
        r = sum(rec for rec in recalls_test.values()) / len(recalls_test)
        f_score = 2/(1/p+1/r)        
        pr = pr + p
        re = re + r
        fs = fs + f_score
        nd = nd + nDCG
        ap = ap + np.nanmean(aps.values())
        
        trainset_to_test = trainset.build_testset()
        predictions_on_train = algo.test(trainset_to_test)
        precisions_train, recalls_train = precision_recall_at_k(predictions_on_train, k=N_rec, threshold=thresh)
    
        aps_train = ap_at_k(predictions_on_train, k=N_rec, threshold=thresh)
        nDCG_train = ndcg_at_k(predictions_on_train,k=N_rec)
        p_train = sum(prec for prec in precisions_train.values()) / len(precisions_train)
        r_train = sum(rec for rec in recalls_train.values()) / len(recalls_train)
        f_score_train = 2/(1/p_train+1/r_train)        
        pr_train = pr_train + p_train
        re_train = re_train + r_train
        fs_train = fs_train + f_score_train
        nd_train = nd_train + nDCG_train
        ap_train = ap_train + np.nanmean(aps_train.values())        

    precision.append(pr/4)
    recall.append(re/4)
    fscore.append(fs/4)
    mean_ap.append(ap/4) # MAP
    normalized_DCG.append(nd/4) # NDCG

    precision_train.append(pr_train/4)
    recall_train.append(re_train/4)
    fscore_train.append(fs_train/4) 
    mean_ap_train.append(ap_train/4) # MAP
    normalized_DCG_train.append(nd_train/4) # NDCG

plt.figure(1)

plt.subplot(211)
BL_PB_N, = plt.plot(ks, normalized_DCG, marker = 'o', linestyle = '-', color = 'r', label = 'kNNBaseline: Pearson_baseline')
plt.ylabel('NDCG')
plt.xlabel('k')

plt.title('Evaluation on Validation set')

plt.subplot(212)
BL_PB_F, = plt.plot(ks, fscore, marker = 'o', linestyle = '-', color =  'r', label = 'kNNBaseline: Pearson_baseline')
plt.ylabel('F-score')
plt.xlabel('k')


## Plot precision, recall, MAP
#plt.subplot(221)
#plt.plot(ks, precision, 'ro')
#plt.plot(ks, precision_train, 'bo')
#plt.ylabel('precision')
#plt.xlabel('k')
#
#plt.subplot(222)
#plt.plot(range(1,len(recall)+1), recall, 'ro')
#plt.plot(range(1,len(recall_train)+1), recall_train, 'bo')
#plt.ylabel('recall')
#plt.xlabel('k')

#plt.subplot(223)
#plt.plot(range(1,len(mean_ap)+1), mean_ap, 'ro')
#plt.plot(range(1,len(mean_ap_train)+1), mean_ap_train, 'bo')
#plt.ylabel('MAP')
#plt.xlabel('k')


# kNN with Z-score
sim_options = {'name': 'pearson',
       'user_based': False  # compute  similarities between items
       }

mean_ap = []
precision = []
recall = []
fscore = []
normalized_DCG = []
mean_ap_train = []
precision_train = []
recall_train = []
fscore_train = []
normalized_DCG_train = []

for k_val in ks:
    print(k_val)
    algo = surprise.KNNWithZScore(k=k_val,sim_options=sim_options)
    pr = 0
    re = 0
    fs = 0
    ap = 0
    nd = 0
    pr_train = 0
    re_train = 0
    fs_train = 0
    ap_train = 0
    nd_train = 0
    for trainset, testset in data.folds():
        algo.train(trainset)
        predictions_on_test = algo.test(testset)
    
        precisions_test, recalls_test = precision_recall_at_k(predictions_on_test, k=N_rec, threshold=thresh)
        aps = ap_at_k(predictions_on_test, k=N_rec, threshold=thresh)
        nDCG = ndcg_at_k(predictions_on_test,k=N_rec)
        p = sum(prec for prec in precisions_test.values()) / len(precisions_test)
        r = sum(rec for rec in recalls_test.values()) / len(recalls_test)
        f_score = 2/(1/p+1/r)        
        pr = pr + p
        re = re + r
        fs = fs + f_score
        nd = nd + nDCG
        ap = ap + np.nanmean(aps.values())
        
        trainset_to_test = trainset.build_testset()
        predictions_on_train = algo.test(trainset_to_test)
        precisions_train, recalls_train = precision_recall_at_k(predictions_on_train, k=N_rec, threshold=thresh)
    
        aps_train = ap_at_k(predictions_on_train, k=N_rec, threshold=thresh)
        nDCG_train = ndcg_at_k(predictions_on_train,k=N_rec)
        p_train = sum(prec for prec in precisions_train.values()) / len(precisions_train)
        r_train = sum(rec for rec in recalls_train.values()) / len(recalls_train)
        f_score_train = 2/(1/p_train+1/r_train)        
        pr_train = pr_train + p_train
        re_train = re_train + r_train
        fs_train = fs_train + f_score_train
        nd_train = nd_train + nDCG_train
        ap_train = ap_train + np.nanmean(aps_train.values())        

    precision.append(pr/4)
    recall.append(re/4)
    fscore.append(fs/4)
    mean_ap.append(ap/4) # MAP
    normalized_DCG.append(nd/4) # NDCG

    precision_train.append(pr_train/4)
    recall_train.append(re_train/4)
    fscore_train.append(fs_train/4) 
    mean_ap_train.append(ap_train/4) # MAP
    normalized_DCG_train.append(nd_train/4) # NDCG

plt.figure(1)

plt.subplot(211)
Z_P_N, = plt.plot(ks, normalized_DCG, marker = 'o', linestyle = '-', color = 'y')
plt.ylabel('NDCG')
plt.xlabel('k')

plt.subplot(212)
Z_P_F, = plt.plot(ks, fscore, marker = 'o', linestyle = '-', color = 'y')
plt.ylabel('F-score')
plt.xlabel('k')



## kNN with mean
sim_options = {'name': 'pearson',
       'user_based': False  # compute  similarities between items
       }

mean_ap = []
precision = []
recall = []
fscore = []
normalized_DCG = []
mean_ap_train = []
precision_train = []
recall_train = []
fscore_train = []
normalized_DCG_train = []

for k_val in ks:
    print(k_val)
    algo = surprise.KNNWithMeans(k=k_val,sim_options=sim_options)
    pr = 0
    re = 0
    fs = 0
    ap = 0
    nd = 0
    pr_train = 0
    re_train = 0
    fs_train = 0
    ap_train = 0
    nd_train = 0
    for trainset, testset in data.folds():
        algo.train(trainset)
        predictions_on_test = algo.test(testset)
    
        precisions_test, recalls_test = precision_recall_at_k(predictions_on_test, k=N_rec, threshold=thresh)
        aps = ap_at_k(predictions_on_test, k=N_rec, threshold=thresh)
        nDCG = ndcg_at_k(predictions_on_test,k=N_rec)
        p = sum(prec for prec in precisions_test.values()) / len(precisions_test)
        r = sum(rec for rec in recalls_test.values()) / len(recalls_test)
        f_score = 2/(1/p+1/r)        
        pr = pr + p
        re = re + r
        fs = fs + f_score
        nd = nd + nDCG
        ap = ap + np.nanmean(aps.values())
        
        trainset_to_test = trainset.build_testset()
        predictions_on_train = algo.test(trainset_to_test)
        precisions_train, recalls_train = precision_recall_at_k(predictions_on_train, k=N_rec, threshold=thresh)
    
        aps_train = ap_at_k(predictions_on_train, k=N_rec, threshold=thresh)
        nDCG_train = ndcg_at_k(predictions_on_train,k=N_rec)
        p_train = sum(prec for prec in precisions_train.values()) / len(precisions_train)
        r_train = sum(rec for rec in recalls_train.values()) / len(recalls_train)
        f_score_train = 2/(1/p_train+1/r_train)        
        pr_train = pr_train + p_train
        re_train = re_train + r_train
        fs_train = fs_train + f_score_train
        nd_train = nd_train + nDCG_train
        ap_train = ap_train + np.nanmean(aps_train.values())        

    precision.append(pr/4)
    recall.append(re/4)
    fscore.append(fs/4)
    mean_ap.append(ap/4) # MAP
    normalized_DCG.append(nd/4) # NDCG

    precision_train.append(pr_train/4)
    recall_train.append(re_train/4)
    fscore_train.append(fs_train/4) 
    mean_ap_train.append(ap_train/4) # MAP
    normalized_DCG_train.append(nd_train/4) # NDCG

plt.figure(1)

plt.subplot(211)
M_P_N, = plt.plot(ks, normalized_DCG, marker = 'o', linestyle = '-', color = 'g')
plt.ylabel('NDCG')
plt.xlabel('k')

plt.subplot(212)
M_P_F, = plt.plot(ks, fscore, marker = 'o', linestyle = '-', color = 'g')
plt.ylabel('F-score')
plt.xlabel('k')


plt.legend([M_P_F, Z_P_F, BL_PB_F, B_P_F], ['kNN Mean, Pearson', 'kNN Z-score, Pearson', 'kNN Baseline, Pearson baseline', 'kNN, Pearson'])
plt.show()
