
# Book Recommendation System

## Contents:

BX-users.csv: Data file with information about the users  
train.csv: Data used for validation and training  
test.csv: Data used for testing 
image: Folder containing all the plots

Data Cleaning and Splitting.py: Cleans the BX-Book-ratings.csv file and splits into train and test files  
Hyperparameter Tunning_1.ipynb: Cross-validation code for tuning latent factors and learning rate hyperparameters for SGD  
Hyperparameter Tunning_2.ipynb: Cross-validation code for tuning regularization term hyperparameter for SGD  
MAP_inference_Coordinat_Ascent.py: Code for implementation of MAP inference algorithm  
kNN_v5.py: Cross-validation code for tuning hyperparameters for kNN  
evaluation.py: Training and test set accuracy and coverage for SGD, kNN and baseline models. Precision-recall curves also plotted    
SampleSizeTime.py: Plots running time as a function of sample size  
sampleSizeAccuracy.py: Plots accuracy as a function of sample size  
