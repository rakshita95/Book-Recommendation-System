#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 15:02:47 2017

@author: Xiaohui(Eartha) Guo
"""

## this code is to recommend books by implementing Matrix Factorization MAP
## inference coordinate ascent algorithm.

import numpy as np
import pandas as pd
import math
from numpy.linalg import inv
import pickle
from collections import defaultdict


# Import dataset

train = pd.read_csv('train.csv')   # 37548 rows × 4 columns

# This test_raw dataset is the origiral data set used by the whold group.
# It contains the predictions values with 0s.
test_raw = pd.read_csv('test.csv')    # 4555126 rows × 4 columns

# Subset the test_raw data set and only select the rows with non 0 rows.
test = test_raw.loc[test_raw['Book-Rating'] != 0]  # 11170 rows × 4 columns

# merge the train and test data set to get all users and items in the dataset. 
whole = pd.concat([train,test])   # 48718 rows × 4 columns


## Start to implement the algorithm

## First of all, create a function to initialize V matrix 
## Here V matrix represents the matrix of books, Later on will introduce 
## matrix for users, which is U matrix. 

def initialize_V(V):
    import numpy as np
    import pandas as pd
    import math

    mean = [0,0,0,0,0,0,0,0,0,0]
    cov = [[1,0,0,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],
       [0,0,0,1,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0],
       [0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,1,0],
       [0,0,0,0,0,0,0,0,0,1]]
    
    
    V = np.zeros((len(np.unique(whole['ISBN'])),10))
    V = np.matrix(V)
    
    for i in range(0,len(np.unique(whole['ISBN']))):
        V[i] = np.random.multivariate_normal(mean, cov, 1)
    
    return V



# Create a V matrix frame with all cells 0 temporately. Set the parameter rank 
# to be 10. The number of columns of the matrix is the number of unique ISBN in 
# the merged dataset "whole".
# The shape of the matrix is (5472, 10)

V = np.zeros((len(np.unique(whole['ISBN'])),10))
V = np.matrix(V)

# Initialize V, prepare for the algorithm.
# initialize V for each run of the 10 runs
# Here run the algorithm for 10 times then select the best run, which has the 
# highest objective value. 

run_1_time_V = initialize_V(V)
run_2_time_V = initialize_V(V)
run_3_time_V = initialize_V(V)
run_4_time_V = initialize_V(V)
run_5_time_V = initialize_V(V)
run_6_time_V = initialize_V(V)
run_7_time_V = initialize_V(V)
run_8_time_V = initialize_V(V)
run_9_time_V = initialize_V(V)
run_10_time_V = initialize_V(V)



# Create the traning matrix. 

#Create column name
column_name = np.unique(whole['ISBN'])
#Create row name
row_name = np.unique(whole['User-ID'])

#Create empty data frame
empty = np.empty((len(row_name),len(column_name)),dtype=np.float)
empty[:] = np.nan

output_data = pd.DataFrame(empty)
#Insert column names
output_data.columns = column_name
    
#Insert row names
index = pd.Index(row_name)
output_data = pd.DataFrame(output_data, index=index)

M_train = output_data

# Fill in training observations

for i in range(0,len(train)):
    M_train.loc[train.loc[i,'User-ID'],train.loc[i,'ISBN']] = train.loc[i,'Book-Rating']
    

# save  M_train  dataframe
# M_train.to_pickle('M_train_df_with_null.pickle')
# load pickle
# df2 = pd.read_pickle('M_train_df_with_null.pickle')


# Create M_train with 0s.


#Create column name
column_name1 = np.unique(whole['ISBN'])
#Create row name
row_name1 = np.unique(whole['User-ID'])


#Create empty data frame
zeros = np.empty((len(row_name1),len(column_name1)),dtype=np.float)
zeros[:] = 0

output_data1 = pd.DataFrame(zeros)
#Insert column names
output_data1.columns = column_name1
    
#Insert row names
index1 = pd.Index(row_name1)
output_data1 = pd.DataFrame(output_data1, index=index1)
M_train_with_zeros = output_data1

M_train_with_zeros[np.isnan(M_train_with_zeros)] = 0

# Fill in training observations

for i in range(0,len(train)):
    M_train_with_zeros.loc[train.loc[i,'User-ID'],train.loc[i,'ISBN']] = train.loc[i,'Book-Rating']





# save M_train_with_zeros

# M_train_with_zeros.to_pickle('M_train_df_with_zeros.pickle')
    
i_j_list = []
for i in range(0,len(np.unique(whole['User-ID']))):
    j_list = []
    for j in range(0,len(np.unique(whole['ISBN']))):
        if math.isnan(M_train.iloc[i,j]) is False:
            j_list.append(j)
    i_j_list.append(j_list)
    

# with open("i_j_list.pickle", "wb") as fp:   #Pickling
#      pickle.dump(i_j_list, fp)
 
    
# with open("i_j_list.pickle", "rb") as fp2:   # Unpickling
#     b = pickle.load(fp2)

j_i_list = []
for j in range(0,len(np.unique(whole['ISBN']))):
    i_list = []
    for i in range(0,len(np.unique(whole['User-ID']))):
        if math.isnan(M_train.iloc[i,j]) is False:
            i_list.append(i)
    j_i_list.append(i_list)
    
# with open("j_i_list.pickle", "wb") as fp_j_i_list:   #Pickling
#      pickle.dump(j_i_list, fp_j_i_list)
 
    

# Function to update for multiple iterations
# Add L value


def matrix_factorization(T, initial_V):
    import numpy as np
    import pandas as pd
    import math
    from numpy.linalg import inv
    
    U_list = []
    V_list = []
    L_list = []
    
    for t in range(0,T):
        if t == 0:

                
            U = np.zeros((1117,10))
            U = np.matrix(U)
            I = np.identity(10)
            I = np.matrix(I)
            first_part = 0.25*I

            for i in range(0,1117):
                middle = np.dot(np.transpose(initial_V[i_j_list[i]]),initial_V[i_j_list[i]])
                last_part = np.dot(np.matrix(M_train_with_zeros.iloc[i,:]),initial_V)
                U[i] = np.dot(last_part,inv(first_part + middle))
    
   
            U_list.append(U)
            
        
        
        
            V = run_1_time_V.copy()
            for j in range(0,5472):
                
                middle = np.dot(np.transpose(U[j_i_list[j]]),U[j_i_list[j]])
                last_part = np.dot(np.matrix(M_train_with_zeros.iloc[:,j]),U)
                V[j] = np.dot(last_part,inv(first_part + middle))
        
        
            V_list.append(V)
            
            
            # Calculate L
            
            middle = (np.linalg.norm(U)**2)*0.5
            last = (np.linalg.norm(V)**2)*0.5
            
            aa = M_train - np.dot(U,np.transpose(V))
            bb = (np.square(aa))/(0.25*2)
            first = np.nansum(bb)
            
            L = (-1)*(first + middle + last)
            L_list.append(L)
            
            
        else:
            I = np.identity(10)
            I = np.matrix(I)
            first_part = 0.25*I

            for i in range(0,1117):
                middle = np.dot(np.transpose(V[i_j_list[i]]),V[i_j_list[i]])
                last_part = np.dot(np.matrix(M_train_with_zeros.iloc[i,:]),V)
                U[i] = np.dot(last_part,inv(first_part + middle))
    
   
            U_list.append(U)
            

            for j in range(0,5472):
                
                middle = np.dot(np.transpose(U[j_i_list[j]]),U[j_i_list[j]])
                last_part = np.dot(np.matrix(M_train_with_zeros.iloc[:,j]),U)
                V[j] = np.dot(last_part,inv(first_part + middle))
        
        
            V_list.append(V)
            
            
            # Calculate L
            
            middle = (np.linalg.norm(U)**2)*0.5
            last = (np.linalg.norm(V)**2)*0.5
            
            aa = M_train - np.dot(U,np.transpose(V))
            bb = (np.square(aa))/(0.25*2)
            first = np.nansum(bb)
            
            L = (-1)*(first + middle + last)
            L_list.append(L)
            

    
    return    U_list, V_list, L_list
    
    

run1 = matrix_factorization(100, run_1_time_V)
run2 = matrix_factorization(100, run_2_time_V)
run3 = matrix_factorization(100, run_3_time_V)
run4 = matrix_factorization(100, run_4_time_V)
run5 = matrix_factorization(100, run_5_time_V)
run6 = matrix_factorization(100, run_6_time_V)
run7 = matrix_factorization(100, run_7_time_V)
run8 = matrix_factorization(100, run_8_time_V)
run9 = matrix_factorization(100, run_9_time_V)
run10 = matrix_factorization(100, run_10_time_V)



# with open("run1.pickle", "wb") as run1_pkl:   #Pickling
#       pickle.dump(run1, run1_pkl)
 

# with open("run2.pickle", "wb") as run2_pkl:   #Pickling
#      pickle.dump(run2, run2_pkl)
 

# with open("run3.pickle", "wb") as run3_pkl:   #Pickling
#      pickle.dump(run3, run3_pkl)
 


# with open("run4.pickle", "wb") as run4_pkl:   #Pickling
#      pickle.dump(run4, run4_pkl)
 
# with open("run5.pickle", "wb") as run5_pkl:   #Pickling
#      pickle.dump(run5, run5_pkl)
 
# with open("run6.pickle", "wb") as run6_pkl:   #Pickling
#      pickle.dump(run6, run6_pkl)
 

# with open("run7.pickle", "wb") as run7_pkl:   #Pickling
#      pickle.dump(run7, run7_pkl)
 
# with open("run8.pickle", "wb") as run8_pkl:   #Pickling
#      pickle.dump(run8, run8_pkl)
 
# with open("run9.pickle", "wb") as run9_pkl:   #Pickling
#      pickle.dump(run9, run9_pkl)
 
# with open("run10.pickle", "wb") as run10_pkl:   #Pickling
#      pickle.dump(run10, run10_pkl)
 



# with open("M_train_df_with_null.pickle", "rb") as M_train_load:   # Unpickling
      #M_train = pickle.load(M_train_load)
      
# with open("M_train_df_with_zeros.pickle", "rb") as M_train_with_zeros_load:   # Unpickling
#      M_train_with_zeros = pickle.load(M_train_with_zeros_load)
      
# with open("i_j_list.pickle", "rb") as i_j_list_load:   # Unpickling
#      i_j_list = pickle.load(i_j_list_load)

# with open("j_i_list.pickle", "rb") as j_i_list_load:   # Unpickling
 #     j_i_list = pickle.load(j_i_list_load)

# with open("run1.pickle", "rb") as run1_load:   # Unpickling
#      run1 = pickle.load(run1_load)
 
 
with open("run2.pickle", "rb") as run2_load:   # Unpickling
      run2 = pickle.load(run2_load)

with open("run3.pickle", "rb") as run3_load:   # Unpickling
      run3 = pickle.load(run3_load)
      
with open("run4.pickle", "rb") as run4_load:   # Unpickling
      run4 = pickle.load(run4_load)

with open("run5.pickle", "rb") as run5_load:   # Unpickling
      run5 = pickle.load(run5_load)

with open("run6.pickle", "rb") as run6_load:   # Unpickling
      run6 = pickle.load(run6_load)
      
with open("run7.pickle", "rb") as run7_load:   # Unpickling
      run7 = pickle.load(run7_load)
      
with open("run8.pickle", "rb") as run8_load:   # Unpickling
      run8 = pickle.load(run8_load)

with open("run9.pickle", "rb") as run9_load:   # Unpickling
      run9 = pickle.load(run9_load)

with open("run10.pickle", "rb") as run10_load:   # Unpickling
      run10 = pickle.load(run10_load)

run1_data = run1[2]
run2_data = run2[2]
run3_data = run3[2]
run4_data = run4[2]
run5_data = run5[2]
run6_data = run6[2]
run7_data = run7[2]
run8_data = run8[2]
run9_data = run9[2]
run10_data = run10[2]

# select the value needed.

run1_data_yes = run1_data[1:101]
run2_data_yes = run2_data[1:101]
run3_data_yes = run3_data[1:101]
run4_data_yes = run4_data[1:101]
run5_data_yes = run5_data[1:101]
run6_data_yes = run6_data[1:101]
run7_data_yes = run7_data[1:101]
run8_data_yes = run8_data[1:101]
run9_data_yes = run9_data[1:101]
run10_data_yes = run10_data[1:101]


# Plot log joint likelihood for the iterations 2 to 100 for each 10 run.
import matplotlib.pyplot as plt
from matplotlib.pyplot import *


t = []
for i in range(2,101):
    t.append(i)
    
plot(t, run1_data_yes,label='1st Run')
plot(t, run2_data_yes,label='2nd Run') 
plot(t, run3_data_yes,label='3rd Run')
plot(t, run4_data_yes,label='4th Run')
plot(t, run5_data_yes,label='5th Run')
plot(t, run6_data_yes,label='6th Run') 
plot(t, run7_data_yes,label='7th Run')
plot(t, run8_data_yes,label='8th Run')
plot(t, run9_data_yes,label='9th Run')
plot(t, run10_data_yes,label='10th Run')


plt.title('Show the log joint likelihood for the iterations 2 to 100 for each run')
plt.xlabel('t')
plt.ylabel('L')


plt.legend()

plt.show()





# First Run to final run
Run1_U = run1[0][99]
Run1_U.shape

Run2_U = run2[0][99]
Run2_U.shape

Run3_U = run3[0][99]
Run3_U.shape

Run4_U = run4[0][99]
Run4_U.shape

Run5_U = run5[0][99]
Run5_U.shape

Run6_U = run6[0][99]
Run6_U.shape

Run7_U = run7[0][99]
Run7_U.shape

Run8_U = run8[0][99]
Run8_U.shape

Run9_U = run9[0][99]
Run9_U.shape

Run10_U = run10[0][99]
Run10_U.shape


Run1_V = run1[1][99]
Run1_V.shape

Run2_V = run2[1][99]
Run2_V.shape

Run3_V = run3[1][99]
Run3_V.shape

Run4_V = run4[1][99]
Run4_V.shape

Run5_V = run5[1][99]
Run5_V.shape

Run6_V = run6[1][99]
Run6_V.shape

Run7_V = run7[1][99]
Run7_V.shape

Run8_V = run8[1][99]
Run8_V.shape

Run9_V = run9[1][99]
Run9_V.shape

Run10_V = run10[1][99]
Run10_V.shape


# M train matrix columns names amd rows names
M_train_column_names_list = list(M_train.columns)
M_train_row_names_list = list(M_train.index)

# reindex the data frame
test_add_index = test.copy()


index_reset = []
for i in range(0,len(test)):
    index_reset.append(i)
    
    
add_index_column = test_add_index.assign(new_index = index_reset)

after_new_index = add_index_column.set_index('new_index')


ratings_test_run1 = after_new_index.copy()
# ratings_test_run1 


ratings_test_run1['1st_run_Obj'] = np.nan
# ratings_test_run1

for i in range(0,11170):
    
    user_id = list(ratings_test_run1['User-ID'])[i]
    row_number = M_train_row_names_list.index(user_id)
    
    ISBN_id = list(ratings_test_run1['ISBN'])[i]
    column_number = M_train_column_names_list.index(ISBN_id)
    
    #ratings_test_run1['1st_run_Obj'][i] = np.dot(Run1_U[row_number],np.transpose(Run1_V[column_number]))   
    ratings_test_run1.loc[i,'1st_run_Obj'] = np.dot(Run1_U[row_number],np.transpose(Run1_V[column_number]))[0,0] 
    

ratings_test_run1_sorted  = ratings_test_run1.sort(['1st_run_Obj'], ascending=[False])





# Run 2 
ratings_test_run2 = after_new_index.copy()
ratings_test_run2

ratings_test_run2['2nd_run_Obj'] = np.nan


for i in range(0,11170):
    
    user_id = list(ratings_test_run2['User-ID'])[i]
    row_number = M_train_row_names_list.index(user_id)
    
    ISBN_id = list(ratings_test_run2['ISBN'])[i]
    column_number = M_train_column_names_list.index(ISBN_id)
     
    ratings_test_run2.loc[i,'2nd_run_Obj'] = np.dot(Run2_U[row_number],np.transpose(Run2_V[column_number]))[0,0] 
    
ratings_test_run2_sorted = ratings_test_run2.sort(['2nd_run_Obj'], ascending=[False])



# Run 3 
ratings_test_run3 = after_new_index.copy()

ratings_test_run3['3rd_run_Obj'] = np.nan


for i in range(0,11170):
    
    user_id = list(ratings_test_run3['User-ID'])[i]
    row_number = M_train_row_names_list.index(user_id)
    
    ISBN_id = list(ratings_test_run3['ISBN'])[i]
    column_number = M_train_column_names_list.index(ISBN_id)
     
    ratings_test_run3.loc[i,'3rd_run_Obj'] = np.dot(Run3_U[row_number],np.transpose(Run3_V[column_number]))[0,0] 
    


ratings_test_run3_sorted = ratings_test_run3.sort(['3rd_run_Obj'], ascending=[False])


# Run 4 
ratings_test_run4 = after_new_index.copy()

ratings_test_run4['4th_run_Obj'] = np.nan


for i in range(0,11170):
    
    user_id = list(ratings_test_run4['User-ID'])[i]
    row_number = M_train_row_names_list.index(user_id)
    
    ISBN_id = list(ratings_test_run4['ISBN'])[i]
    column_number = M_train_column_names_list.index(ISBN_id)
     
    ratings_test_run4.loc[i,'4th_run_Obj'] = np.dot(Run4_U[row_number],np.transpose(Run4_V[column_number]))[0,0] 
    


ratings_test_run4_sorted = ratings_test_run4.sort(['4th_run_Obj'], ascending=[False])



# Run 5
ratings_test_run5 = after_new_index.copy()

ratings_test_run5['5th_run_Obj'] = np.nan


for i in range(0,11170):
    
    user_id = list(ratings_test_run5['User-ID'])[i]
    row_number = M_train_row_names_list.index(user_id)
    
    ISBN_id = list(ratings_test_run5['ISBN'])[i]
    column_number = M_train_column_names_list.index(ISBN_id)
     
    ratings_test_run5.loc[i,'5th_run_Obj'] = np.dot(Run5_U[row_number],np.transpose(Run5_V[column_number]))[0,0] 
    


ratings_test_run5_sorted = ratings_test_run5.sort(['5th_run_Obj'], ascending=[False])



# Run 6
ratings_test_run6 = after_new_index.copy()

ratings_test_run6['6th_run_Obj'] = np.nan


for i in range(0,11170):
    
    user_id = list(ratings_test_run6['User-ID'])[i]
    row_number = M_train_row_names_list.index(user_id)
    
    ISBN_id = list(ratings_test_run6['ISBN'])[i]
    column_number = M_train_column_names_list.index(ISBN_id)
     
    ratings_test_run6.loc[i,'6th_run_Obj'] = np.dot(Run6_U[row_number],np.transpose(Run6_V[column_number]))[0,0] 
    


ratings_test_run6_sorted = ratings_test_run6.sort(['6th_run_Obj'], ascending=[False])



# Run 7
ratings_test_run7 = after_new_index.copy()

ratings_test_run7['7th_run_Obj'] = np.nan


for i in range(0,11170):
    
    user_id = list(ratings_test_run7['User-ID'])[i]
    row_number = M_train_row_names_list.index(user_id)
    
    ISBN_id = list(ratings_test_run7['ISBN'])[i]
    column_number = M_train_column_names_list.index(ISBN_id)
     
    ratings_test_run7.loc[i,'7th_run_Obj'] = np.dot(Run7_U[row_number],np.transpose(Run7_V[column_number]))[0,0] 
    


ratings_test_run7_sorted = ratings_test_run7.sort(['7th_run_Obj'], ascending=[False])



# Run 8
ratings_test_run8 = after_new_index.copy()

ratings_test_run8['8th_run_Obj'] = np.nan


for i in range(0,11170):
    
    user_id = list(ratings_test_run8['User-ID'])[i]
    row_number = M_train_row_names_list.index(user_id)
    
    ISBN_id = list(ratings_test_run8['ISBN'])[i]
    column_number = M_train_column_names_list.index(ISBN_id)
     
    ratings_test_run8.loc[i,'8th_run_Obj'] = np.dot(Run8_U[row_number],np.transpose(Run8_V[column_number]))[0,0] 
    


ratings_test_run8_sorted = ratings_test_run8.sort(['8th_run_Obj'], ascending=[False])



# Run 9
ratings_test_run9 = after_new_index.copy()

ratings_test_run9['9th_run_Obj'] = np.nan


for i in range(0,11170):
    
    user_id = list(ratings_test_run9['User-ID'])[i]
    row_number = M_train_row_names_list.index(user_id)
    
    ISBN_id = list(ratings_test_run9['ISBN'])[i]
    column_number = M_train_column_names_list.index(ISBN_id)
     
    ratings_test_run9.loc[i,'9th_run_Obj'] = np.dot(Run9_U[row_number],np.transpose(Run9_V[column_number]))[0,0] 
    


ratings_test_run9_sorted = ratings_test_run9.sort(['9th_run_Obj'], ascending=[False])





# Run 10
ratings_test_run10 = after_new_index.copy()

ratings_test_run10['10th_run_Obj'] = np.nan


for i in range(0,11170):
    
    user_id = list(ratings_test_run10['User-ID'])[i]
    row_number = M_train_row_names_list.index(user_id)
    
    ISBN_id = list(ratings_test_run10['ISBN'])[i]
    column_number = M_train_column_names_list.index(ISBN_id)
     
    ratings_test_run10.loc[i,'10th_run_Obj'] = np.dot(Run10_U[row_number],np.transpose(Run10_V[column_number]))[0,0] 
    


ratings_test_run10_sorted = ratings_test_run10.sort(['10th_run_Obj'], ascending=[False])



# Calculate RMSE
# Run 1

from sklearn.metrics import mean_squared_error
real_run1 = np.array(ratings_test_run1['Book-Rating'])
prediction_run1 = np.array(ratings_test_run1['1st_run_Obj'])
RMSE_run1 = mean_squared_error(real_run1, prediction_run1)**0.5
RMSE_run1


# Run 2

real_run2 = np.array(ratings_test_run2['Book-Rating'])
prediction_run2 = np.array(ratings_test_run2['2nd_run_Obj'])
RMSE_run2 = mean_squared_error(real_run2, prediction_run2)**0.5
RMSE_run2



# Run 3

real_run3 = np.array(ratings_test_run3['Book-Rating'])
prediction_run3 = np.array(ratings_test_run3['3rd_run_Obj'])
RMSE_run3 = mean_squared_error(real_run3, prediction_run3)**0.5
RMSE_run3


# Run 4

real_run4 = np.array(ratings_test_run4['Book-Rating'])
prediction_run4 = np.array(ratings_test_run4['4th_run_Obj'])
RMSE_run4 = mean_squared_error(real_run4, prediction_run4)**0.5
RMSE_run4


# Run 5

real_run5 = np.array(ratings_test_run5['Book-Rating'])
prediction_run5 = np.array(ratings_test_run5['5th_run_Obj'])
RMSE_run5 = mean_squared_error(real_run5, prediction_run5)**0.5
RMSE_run5


# Run 6

real_run6 = np.array(ratings_test_run6['Book-Rating'])
prediction_run6 = np.array(ratings_test_run6['6th_run_Obj'])
RMSE_run6 = mean_squared_error(real_run6, prediction_run6)**0.5
RMSE_run6


# Run 7

real_run7 = np.array(ratings_test_run7['Book-Rating'])
prediction_run7 = np.array(ratings_test_run7['7th_run_Obj'])
RMSE_run7 = mean_squared_error(real_run7, prediction_run7)**0.5
RMSE_run7


# Run 8

real_run8 = np.array(ratings_test_run8['Book-Rating'])
prediction_run8 = np.array(ratings_test_run8['8th_run_Obj'])
RMSE_run8 = mean_squared_error(real_run8, prediction_run8)**0.5
RMSE_run8


# Run 9

real_run9 = np.array(ratings_test_run9['Book-Rating'])
prediction_run9 = np.array(ratings_test_run9['9th_run_Obj'])
RMSE_run9 = mean_squared_error(real_run9, prediction_run9)**0.5
RMSE_run9



# Run 10

real_run10 = np.array(ratings_test_run10['Book-Rating'])
prediction_run10 = np.array(ratings_test_run10['10th_run_Obj'])
RMSE_run10 = mean_squared_error(real_run10, prediction_run10)**0.5
RMSE_run10


RMSE_Value = [RMSE_run1,RMSE_run2,RMSE_run3,RMSE_run4,RMSE_run5,
             RMSE_run6,RMSE_run7,RMSE_run8,RMSE_run9,RMSE_run10]
RMSE_Value = pd.Series(RMSE_Value)


objective_value = [run1[2][99],run2[2][99],run3[2][99],run4[2][99],run5[2][99],run6[2][99],run7[2][99],run8[2][99],
                  run9[2][99],run10[2][99]]
objective_value = pd.Series(objective_value)


run_index = [1,2,3,4,5,6,7,8,9,10]
run_index = pd.Series(run_index)


table_2a = pd.concat([run_index, RMSE_Value, objective_value], axis=1)
table_2a.columns = ['Number of Run', 'RMSE', 'Value of Training Objective Function']

table_2a_sorted = table_2a.sort(['Value of Training Objective Function'],ascending=[False])

table_2a_sorted_complete = table_2a_sorted.reset_index(drop=True)
table_2a_sorted_complete.index = table_2a_sorted_complete.index + 1


table_2a1 = pd.concat([run_index,  objective_value], axis=1)
table_2a1.columns = ['Number of Run', 'Objective Function']
table_2a1_sorted = table_2a1.sort(['Objective Function'],ascending=[False])


Find_The_highest_objective_value  = pd.Series(objective_value, index=run_index)
Find_The_highest_objective_value = pd.DataFrame(Find_The_highest_objective_value)
Find_The_highest_objective_value.columns = ['objective value']
Find_The_highest_objective_value_sorted = Find_The_highest_objective_value.sort(['objective value'], ascending=[False])



test_add_index = test.copy()

index_reset = []
for i in range(0,len(test)):
    index_reset.append(i)
    
add_index_column = test_add_index.assign(new_index = index_reset)

after_new_index = add_index_column.set_index('new_index')

test_prediction = after_new_index.copy()

test_prediction['Prediction'] = np.nan


# select U and V matrix created at the 3rd run's final iteration

Run3_U = run3[0][99]
Run3_U.shape

Run3_V = run3[1][99]
Run3_V.shape

M_train_column_names_list = list(M_train.columns)
M_train_row_names_list = list(M_train.index)


# Make prediction on the test data set.
for i in range(0, 11170):
    
    user_id = test_prediction.loc[i, 'User-ID']
    row_number = M_train_row_names_list.index(user_id)
    
    ISBN_id = test_prediction.loc[i, 'ISBN']
    column_number = M_train_column_names_list.index(ISBN_id)
    
    test_prediction.loc[i,'Prediction'] = np.dot(Run3_U[row_number], np.transpose(Run3_V[column_number]))[0,0]
    

# this list is used for the function "precision_recall_at_k_coordinate_ascent(predictions, k, threshold)"
predictions_list = []
for i in range(0,11170):
    user_id = test_prediction.loc[i,'User-ID']
    real_rating =  test_prediction.loc[i,'Book-Rating']
    estimation = test_prediction.loc[i,'Prediction']
    sublist = [user_id,real_rating,estimation]
    predictions_list.append(sublist)


def precision_recall_at_k_coordinate_ascent(predictions, k, threshold):
    '''Return precision and recall at k metrics for each user.'''

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, true_r, est in predictions:
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

    return precision,recall


# Calcualte the precisiona and recall ont the test data set.

precision_recall_at_k_coordinate_ascent(predictions_list, 10, 7)


# Calculate F value on the test data set. 
2*0.85224879566866951*0.50356112603203029

0.85224879566866951+0.50356112603203029

0.8583187264127139/1.3558099217006998


# Calculat the ndcg 

def ndcg_at_k(predictions, k):
    dcgs = dict()
    idcgs = dict()
    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, true_r, est,in predictions:
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




ndcg_at_k(predictions_list, 10)





# test on train data set
train_prediction = train.copy()

train_prediction['Prediction'] = np.nan

for i in range(0, 37548):
    
    user_id = train_prediction.loc[i, 'User-ID']
    row_number = M_train_row_names_list.index(user_id)
    
    ISBN_id = train_prediction.loc[i, 'ISBN']
    column_number = M_train_column_names_list.index(ISBN_id)
    
    train_prediction.loc[i,'Prediction'] = np.dot(Run3_U[row_number], np.transpose(Run3_V[column_number]))[0,0]
    

# this list is used for the function "precision_recall_at_k_coordinate_ascent(predictions, k, threshold)"
train_predictions_list = []
for i in range(0,37548):
    user_id = train_prediction.loc[i,'User-ID']
    real_rating =  train_prediction.loc[i,'Book-Rating']
    estimation = train_prediction.loc[i,'Prediction']
    sublist = [user_id,real_rating,estimation]
    train_predictions_list.append(sublist)

# calcualte the precision and recall on train data set
precision_recall_at_k_coordinate_ascent(train_predictions_list, 10, 7)

# Calculate the F scroe on train data set
2*1*0.54254125751071769

1+0.54254125751071769

1.0850825150214354/1.5425412575107176

# Calculate the ndcg on train data set
ndcg_at_k(train_predictions_list, 10)

# Calcualte the coverage

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
    for uid, iid, true_r, est, in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n



# this list is used for the function "get_top_n(predictions_knn, n=10)
predictions_coordinate_ascend_test = []
for i in range(0,11170):
    user_id = test_prediction.loc[i,'User-ID']
    ISBN_id = test_prediction.loc[i, 'ISBN']
    real_rating =  test_prediction.loc[i,'Book-Rating']
    estimation = test_prediction.loc[i,'Prediction']
    sublist = [user_id, ISBN_id, real_rating, estimation]
    predictions_coordinate_ascend_test.append(sublist)
    

n_items_test = len(np.unique(test['ISBN']))



# Calculate test coverage

top_n = get_top_n(predictions_coordinate_ascend_test, n=10)
items = []
for uid, user_ratings in top_n.items():
#    print(len([iid for (iid, _) in user_ratings]))
    for (iid,rtg) in user_ratings:
        if rtg >= 7:
            items.append(iid)

len(set(items))/n_items_test


n_items_train = len(np.unique(train['ISBN']))


# this list is used for the function "get_top_n(predictions_knn, n=10)
predictions_coordinate_ascend_train = []
for i in range(0,37548):
    user_id = train_prediction.loc[i,'User-ID']
    ISBN_id = train_prediction.loc[i, 'ISBN']
    real_rating =  train_prediction.loc[i,'Book-Rating']
    estimation = train_prediction.loc[i,'Prediction']
    sublist = [user_id, ISBN_id, real_rating, estimation]
    predictions_coordinate_ascend_train.append(sublist)
    
    
# Calculate train coverage

top_n_train = get_top_n(predictions_coordinate_ascend_train, n=10)
items_train = []
for uid, user_ratings in top_n_train.items():
#    print(len([iid for (iid, _) in user_ratings]))
    for (iid,rtg) in user_ratings:
        if rtg >= 7:
            items_train.append(iid)

len(set(items_train))/n_items_train



# Calculate user space coverage

def user_space_coverage(predictions, k, n_user, threshold):
    # First map the predictions to each user.
    user_est = defaultdict(list)
    for uid, est, in predictions:
        if est >= threshold:
            user_est[uid].append(est)
    n_user_k = sum((len(n_est) >= k ) for n_est in user_est.values())
    a = n_user_k/n_user
    return a


# this list is used for the function "user_space_coverage(predictions, k, n_user, threshold)"
predictions_train_user_space_coverage = []
for i in range(0,37548):
    user_id = train_prediction.loc[i,'User-ID']
    estimation = train_prediction.loc[i,'Prediction']
    sublist = [user_id,estimation]
    predictions_train_user_space_coverage.append(sublist)
    
n_users_train = len(np.unique(train['User-ID']))

user_space_coverage(predictions_train_user_space_coverage, 10, n_users_train, 7)


# this list is used for the function "get_top_n(predictions_knn, n=10)
predictions_test_user_space_coverage = []
for i in range(0,11170):
    user_id = test_prediction.loc[i,'User-ID']
    estimation = test_prediction.loc[i,'Prediction']
    sublist = [user_id, estimation]
    predictions_test_user_space_coverage.append(sublist)
    
    
    
n_users_test = len(np.unique(test['User-ID']))

user_space_coverage(predictions_test_user_space_coverage, 10, n_users_test, 7)






























