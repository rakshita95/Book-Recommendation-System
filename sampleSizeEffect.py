#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 11:40:41 2017

@author: rakshitanagalla
"""
from __future__ import division
import pandas as pd
import numpy as np
import surprise
import matplotlib.pyplot as plt
from collections import defaultdict

def precision_recall_at_k(predictions, k, threshold):
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
        precision = sum(prec for prec in precisions.values()) / len(precisions)

        # Recall@K: Proportion of relevant items that are recommended
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1
        recall = sum(rec for rec in recalls.values()) / len(recalls)

    return precision, recall

def ndcg_at_k(predictions, k):
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

data = pd.read_csv('sampled.csv')
print "Users: "+str(len(np.unique(data['User-ID'])))+ " items: "+str(len(np.unique(data['ISBN'])))
print "No. of ratings: "+str(len(data))

sim_options = {'name': 'pearson',
               'user_based': False
               }

algo_knn = surprise.KNNBasic(k=5, sim_options=sim_options)
algo_svd = surprise.SVD(n_factors = 10, lr_all= 0.001, reg_all =1)

#Around 80% train data for each of these splits
sample_sizes = [0.4, 0.2, 0.1, 0.01]

ndcg_knn = []
ndcg_svd = []
f_knn = []
f_svd = []
for s in sample_sizes:
    a = data.sample(frac = s, random_state = 111)
    print "s= "+str(len(a))
    
    print("Removing users with less than 20 ratings....")
    b = a.groupby('User-ID').filter(lambda x: len(x) >= 20)
    densityu = (float(len(b))/(len(np.unique(b['User-ID']))*len(np.unique(b['ISBN']))))*100
    print "Density after filtering users: "+str(densityu) #0.061
    print "Users: "+str(len(np.unique(b['User-ID'])))+ " items: "+str(len(np.unique(b['ISBN'])))

    print("Splitting into train and test....")
    #randomly assigned 10 ratings to the test set (so that we have a fair number of items to rank)
    grouped = b.groupby('User-ID')
    test = grouped.apply(lambda x: x.sample(10))
    test = test.reset_index(drop=True)
    print "Users: "+str(len(np.unique(test['User-ID'])))+ " items: "+str(len(np.unique(test['ISBN'])))

    keys = ['User-ID', 'ISBN']
    i1 = b.set_index(keys).index
    i2 = test.set_index(keys).index
    train = b[~i1.isin(i2)] 
    print "Users: "+str(len(np.unique(train['User-ID'])))+ " items: "+str(len(np.unique(train['ISBN'])))
    print(len(train)/len(b))
    
    print("Loading train and test sets....")
    reader = surprise.Reader(rating_scale=(1, 10))
    dta = surprise.Dataset.load_from_df(train[['User-ID', 'ISBN', 'Book-Rating']], reader)
    trainset = dta.build_full_trainset()
    t = [tuple(x) for x in test[['User-ID', 'ISBN', 'Book-Rating']].values]
    
    print("Training....")
    algo_knn.train(trainset)
    algo_svd.train(trainset)
    
    print('Testing.....')
    predictions_knn = algo_knn.test(t)
    ndcg_knn.append(ndcg_at_k(predictions_knn, k=10))
    p_knn, r_knn = precision_recall_at_k(predictions_knn, k=10, threshold=7)
    f_knn.append(2*p_knn*r_knn/(p_knn+r_knn))

    predictions_svd = algo_svd.test(t)
    ndcg_svd.append(ndcg_at_k(predictions_svd, k=10))
    p_svd, r_svd = precision_recall_at_k(predictions_svd, k=10, threshold=7)
    f_svd.append(2*p_svd*r_svd/(p_svd+r_svd))
    
plt.plot(sample_sizes, f_svd, 'r', label = 'svd')
plt.plot(sample_sizes, f_knn, 'g', label = 'knn')
plt.xlabel('Fraction of data sampled')
plt.ylabel('F-Score')
plt.legend()
plt.show()

plt.plot(sample_sizes, ndcg_svd, 'r', label = 'svd')
plt.plot(sample_sizes, ndcg_knn, 'g', label = 'knn')
plt.xlabel('Fraction of data sampled')
plt.ylabel('NDCG')
plt.legend()
plt.show()


