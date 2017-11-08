#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 17:48:51 2017

@author: rakshitanagalla
"""

#BaselineOnly
from __future__ import division
import pandas as pd
import numpy as np
from collections import defaultdict
import surprise

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


def user_space_coverage(predictions, k, n_user, threshold):
	# First map the predictions to each user.
    user_est = defaultdict(list)
    for uid, _, _, est, _ in predictions:
        if est >= threshold:
            user_est[uid].append(est)
    n_user_k = sum((len(n_est) >= k ) for n_est in user_est.values())
    a = n_user_k/n_user
    return a

def get_top_n(predictions, n=10):

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

def item_space_coverage(predictions, k, n_items, threshold):
    top_n = get_top_n(predictions, k)
    items = []
    for uid, user_ratings in top_n.items():
        for (iid,rtg) in user_ratings:
            if rtg>=threshold:
                items.append(iid)
    
    return(len(set(items))/n_items)


train = pd.read_csv('train.csv') 
n_items_train = len(np.unique(train['ISBN']))
n_users_train = len(np.unique(train['User-ID']))

reader = surprise.Reader(rating_scale=(1, 10))
data = surprise.Dataset.load_from_df(train[['User-ID', 'ISBN', 'Book-Rating']], reader)

test = pd.read_csv('test.csv')
n_items_test = len(np.unique(test['ISBN']))
n_users_test = len(np.unique(train['User-ID']))
t = [tuple(x) for x in test[['User-ID', 'ISBN', 'Book-Rating']].values]
   
'''
Training
'''
#algo = surprise.BaselineOnly()
algo = surprise.NormalPredictor()
sim_options = {'name': 'pearson',
               'user_based': False
               }
algo_knn = surprise.KNNBasic(k=5, sim_options=sim_options)
#algo_svd = surprise.SVD(n_factors = 5, lr_all= 0.01, reg_all =1.3)#n_factors = , lr_all =, reg_all =
algo_svd = surprise.SVD(n_factors = 10, lr_all= 0.001, reg_all =1)
'''
Baseline
''' 
print "\n Baseline\n"  
# retrain on the whole train set
trainset = data.build_full_trainset()
algo.train(trainset)

# Compute biased accuracy on train set
predictions = algo.test(trainset.build_testset())
precision, recall = precision_recall_at_k(predictions, k=10, threshold=7)
ndcg = ndcg_at_k(predictions, k=10)
print "\n Training Set accuracy:"
print "Precision: "+str(precision)
print "Recall: "+str(recall)
f = (2*precision*recall)/(precision+recall)
print "F-Score: "+str(f)
print "NDCG: "+str(ndcg)
print "User-space coverage: "+str(user_space_coverage(predictions, 10, n_users_train, 7))
print "Item-space coverage: "+str(item_space_coverage(predictions, 10, n_items_train,7))

# Compute unbiased accuracy on test set
predictions = algo.test(t)
precision, recall = precision_recall_at_k(predictions, k=10, threshold=7)
idcg = ndcg_at_k(predictions, k=10)
print "\n Test Set accuracy:"
print "Precision: "+str(precision)
print "Recall: "+str(recall)
f = (2*precision*recall)/(precision+recall)
print "F-Score: "+str(f)
print "NDCG: "+str(idcg)
print "Item-space coverage: "+str(item_space_coverage(predictions, 10, n_items_test,7))
print "User-space coverage: "+str(user_space_coverage(predictions, 10, n_users_test,7))

'''
kNN
''' 
print "\n kNN\n"  
# retrain on the whole train set
trainset_knn = data.build_full_trainset()
algo_knn.train(trainset_knn)

# Compute biased accuracy on train set
predictions_knn = algo_knn.test(trainset_knn.build_testset())
precision_knn, recall_knn = precision_recall_at_k(predictions_knn, k=10, threshold=7)
ndcg_knn = ndcg_at_k(predictions_knn, k=10)
print "\n Training Set accuracy:"
print "Precision: "+str(precision_knn)
print "Recall: "+str(recall_knn)
f_knn = (2*precision_knn*recall_knn)/(precision_knn+recall_knn)
print "F-Score: "+str(f_knn)
print "NDCG: "+str(ndcg_knn)
print "Item-space coverage: "+str(item_space_coverage(predictions_knn, 10, n_items_train,7))
print "User-space coverage: "+str(user_space_coverage(predictions_knn, 10, n_users_train,7))


# Compute unbiased accuracy on test set
predictions_knn = algo_knn.test(t)
precision_knn, recall_knn = precision_recall_at_k(predictions_knn, k=10, threshold=7)
idcg_knn = ndcg_at_k(predictions_knn, k=10)
print "\n Test Set accuracy:"
print "Precision: "+str(precision_knn)
print "Recall: "+str(recall_knn)
f_knn = (2*precision_knn*recall_knn)/(precision_knn+recall_knn)
print "F-Score: "+str(f_knn)
print "NDCG: "+str(idcg_knn)
print "Item-space coverage: "+str(item_space_coverage(predictions_knn, 10, n_items_test,7))
print "User-space coverage: "+str(user_space_coverage(predictions_knn, 10, n_users_test,7))


'''
SVD
''' 
print "\n SVD\n"  
# retrain on the whole train set
trainset_svd = data.build_full_trainset()
algo_svd.train(trainset_svd)

# Compute biased accuracy on train set
predictions_svd = algo_svd.test(trainset_svd.build_testset())
precision_svd, recall_svd = precision_recall_at_k(predictions_svd, k=10, threshold=7)
ndcg_svd = ndcg_at_k(predictions_svd, k=10)
print "\n Training Set accuracy:"
print "Precision: "+str(precision_svd)
print "Recall: "+str(recall_svd)
f_svd = (2*precision_svd*recall_svd)/(precision_svd+recall_svd)
print "F-Score: "+str(f_svd)
print "NDCG: "+str(ndcg_svd)
print "Item-space coverage: "+str(item_space_coverage(predictions_svd, 10, n_items_train,7))
print "User-space coverage: "+str(user_space_coverage(predictions_svd, 10, n_users_train,7))


# Compute unbiased accuracy on test set
predictions_svd = algo_svd.test(t)
precision_svd, recall_svd = precision_recall_at_k(predictions_svd, k=10, threshold=7)
idcg_svd = ndcg_at_k(predictions_svd, k=10)
print "\n Test Set accuracy:"
print "Precision: "+str(precision_svd)
print "Recall: "+str(recall_svd)
f_svd = (2*precision_svd*recall_svd)/(precision_svd+recall_svd)
print "F-Score: "+str(f_svd)
print "NDCG: "+str(idcg_svd)
print "Item-space coverage: "+str(item_space_coverage(predictions_svd, 10, n_items_test, 7))
print "User-space coverage: "+str(user_space_coverage(predictions_svd, 10, n_users_test,7))


'''
Precision-Recall
'''
pb=[]
rb=[]
pk=[]
rk=[]
ps=[]
rs=[]
for k in np.arange(1,11):
    precision_b, recall_b = precision_recall_at_k(predictions, k, threshold=7)
    precision_k, recall_k = precision_recall_at_k(predictions_knn, k, threshold=7)
    precision_s, recall_s = precision_recall_at_k(predictions_svd, k, threshold=7)

    pb.append(precision_b)
    rb.append(recall_b)
    pk.append(precision_k)
    rk.append(recall_k)
    ps.append(precision_s)
    rs.append(recall_s)

import matplotlib.pyplot as plt 
plt.plot(rb,pb,'b',label = 'Baseline' )
plt.plot(rk,pk,'r',label = 'kNN' )
plt.plot(rs,ps,'g',label = 'SVD' )
#plt.plot(rc, pc, 'y', label = 'CD')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.show()

