
# coding: utf-8
#!/usr/bin/env python2


import pandas as pd
import numpy as np

users = pd.read_csv('BX-Users.csv',sep=';') #278858 users
items = pd.read_csv('BX-Books.csv',sep=';',error_bad_lines=False) #271360 items
ratings = pd.read_csv('BX-Book-Ratings.csv',sep=';') #105283 users/ 340556 items. More items in ratings than in items.
ratings = ratings[ratings['Book-Rating'] > 0] # Users: 77805 items: 185973

##Clean
#users in ratings.csv are all included in users.csv
#But items in ratings.csv more than items.csv --> invalid ISBNs in ratings.csv. We first filter them out
ratings = ratings[ratings['ISBN'].isin(items['ISBN'])] # Users: 68091 items: 149836
#twice as many users as items
density = (float(len(ratings))/(len(np.unique(ratings['User-ID']))*len(np.unique(ratings['ISBN']))))*100
print "Density in percent: "+str(density) 
print "Users: "+str(len(np.unique(ratings['User-ID'])))+ " items: "+str(len(np.unique(ratings['ISBN'])))

#Remove items which were rated less than 10 times
a = ratings.groupby('ISBN').filter(lambda x: len(x) >= 10)
densityi = (float(len(a))/(len(np.unique(a['User-ID']))*len(np.unique(a['ISBN']))))*100
print "Density after filtering items: "+str(densityi) #0.061
print "Users: "+str(len(np.unique(a['User-ID'])))+ " items: "+str(len(np.unique(a['ISBN'])))

#Remove users who gave less than 20 ratings
b = a.groupby('User-ID').filter(lambda x: len(x) >= 20)
densityu = (float(len(b))/(len(np.unique(b['User-ID']))*len(np.unique(b['ISBN']))))*100
print "Density after filtering users: "+str(densityu) #0.061
print "Users: "+str(len(np.unique(b['User-ID'])))+ " items: "+str(len(np.unique(b['ISBN'])))

#randomly assigned 10 ratings to the test set (so that we have a fair number of items to rank)
grouped = b.groupby('User-ID')
test = grouped.apply(lambda x: x.sample(10))
test = test.reset_index(drop=True)

keys = ['User-ID', 'ISBN']
i1 = b.set_index(keys).index
i2 = test.set_index(keys).index
train = b[~i1.isin(i2)]

test_perc = float(len(test))/float(len(train)+len(test))*100
print "Percentage of test set: "+str(test_perc)

train.to_csv('train.csv')
test.to_csv('test.csv')
b.to_csv('sampled.csv')

