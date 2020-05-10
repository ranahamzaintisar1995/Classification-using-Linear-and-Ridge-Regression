#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 14:26:14 2019

@author: ranahamzaintisar
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.linalg import svd
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import accuracy_score as acc
import random
from sklearn.utils import shuffle

'''functions built'''

## Function to split into training and test dataset and create target vector z:

def training_test_split(dataset):

    split_data = np.array_split(dataset,10)
    training =[]
    test = []
    for i in range(len(split_data)):
        np.random.shuffle(split_data[i])
        train_test_split = np.array_split(split_data[i],2)
        for item in train_test_split[0]:
            if i == 0:
                new = np.append(item,10) #class label 10 for digit 0
                training.append(new)
            else:
                new = np.append(item,i) # class labels for other digits
                training.append(new)
        for item in train_test_split[1]:
             if i == 0:
                new = np.append(item,10)
                test.append(new)
             else:
                new = np.append(item,i)
                test.append(new)

    # Training dataset with target vector Z
    training_dataset = pd.DataFrame(training)
    training_dataset[240] = training_dataset[240].astype('category') # make class label as category
    ##create dummy variables for the categorical variable i.e target vectors
    training_dataset = pd.get_dummies(training_dataset, dummy_na=True, prefix_sep='_' )
    ## drop nan dummy columns if created
    training_dataset = training_dataset.loc[:, training_dataset.nunique(axis=0) > 1]

    # Test dataset with target vector Z
    test_dataset = pd.DataFrame(test)
    test_dataset[240] = test_dataset[240].astype('category') # make class label as category
    ##create dummy variables for the categorical variable i.e target vectors
    test_dataset = pd.get_dummies(test_dataset, dummy_na=True, prefix_sep='_' )
    ## drop nan dummy columns if created
    test_dataset = test_dataset.loc[:, test_dataset.nunique(axis=0) > 1]


    return training_dataset , test_dataset

## function to seperate feature vectors from binary target vectors
def split_features_labels(data):
    label_col = [x for x in data.columns if isinstance(x, str)]

    return (data.drop(label_col, axis=1),
            data[label_col])


def split_features_labels_cv(data):
    label_col = [x for x in data.columns if x>239]

    return (data.drop(label_col, axis=1),
            data[label_col])


## function to center the data
def center(df):
    cols = df.columns
    for field in cols:
        mean_field = df[field].mean()
        # account for constant columns
        if np.all(df[field] - mean_field != 0):
            df.loc[:, field] = (df[field] - mean_field)

    return df

## Function to find coorelation matrix of the centered data point:

def coor_c(df):
    df_matrix = df.as_matrix()
    df_matrix_transpose = df_matrix.transpose()
    coor_matrix = np.dot(df_matrix_transpose,df_matrix)
    n = coor_matrix.shape[1]
    normal_coor_matrix = np.multiply(coor_matrix,1/n)
    return normal_coor_matrix

##Function Computing the eigenvalues and right eigenvectors of coorelation matrix.
#and returning them in decending order

def eigen(coor_matrix):
    #compute the eigen vector and values
    eig_val_cov, eig_vec_cov = np.linalg.eig(coorelation_matrix_train )
    ## sort eigen vector and eigen values from high to low
    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_val_cov[i]), eig_vec_cov[:,i]) for i in range(len(eig_val_cov))]
    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    #seperate the sorted pair
    eigen_val_decending =[]
    for i in eig_pairs:
        eigen_val_decending.append(i[0])
    eigen_vec_decending = []
    for i in eig_pairs:
        eigen_vec_decending.append(i[1])

    return eigen_val_decending,eigen_vec_decending

## function to reaturn number of desiered PC features and padded with Bias

def pc_features(eigen_vec,eigen_val,centered_data,num_pc):
    s_pc = num_pc
    pc_vectors = np.stack(eigen_vec[0:s_pc],axis=0)
    pc_eigen_val = np.stack(eigen_val[0:s_pc],axis=0)
    pc_features = np.dot(pc_vectors,centered_data.as_matrix().transpose()).transpose()
    #add bias to the features:
    feat_df= pd.DataFrame(pc_features)
    bias = np.full(pc_features.shape[0],1)
    feat_df['bias']=bias
    features = feat_df.as_matrix()
    return features,pc_eigen_val

## Ridge regression function using formula 39 ML notes
def ridge_reg(features,target,a):

    ##computing the SVD
    semi_def_matrix = np.dot(features,features.transpose())
    target_matrix = target.as_matrix()
    num_data=semi_def_matrix.shape[0]
    identity_matrix = np.identity(num_data)
    alpha = a
    alpha_sq= alpha**2
    r_mat = alpha_sq*identity_matrix
    ridge_matrix = semi_def_matrix+r_mat
    ridge_matrix_inv = np.linalg.inv(ridge_matrix)
    wopt_inv= np.matmul(np.matmul(ridge_matrix_inv,features).transpose(),target_matrix)
    wopt = wopt_inv.transpose()
    ## use the wopt to find hypothesis vectors
    hypothesis_matrix = np.matmul(wopt,features.transpose()).transpose()
    ## use hypothesis vectors to find prediction
    prediction = []
    for row in hypothesis_matrix:
        pred = np.zeros_like(row,dtype='int')
        index = np.argmax(row)
        pred[index]=1
        prediction.append(pred)
        df_pred = pd.DataFrame(prediction)
        pred_matrix = df_pred.as_matrix()

    return pred_matrix , target_matrix

def misclass_rate(pred,actual):
    return 1-((sum(np.array([np.argmax(a) for a in pred])==np.array([np.argmax(a) for a in actual]))).astype("float")/len(actual))

def meansq_error(pred,actual):
    return np.mean((pred - actual)**2)

##cross validation with alpha
def cv_ridge(dataset,no_fold,tune_grid,numdim):

    #take the training dataframe with the target vectors
    cv_df = dataset.copy()
    # make k fold splits
    a = []
    mse_tr_a = []
    m_tr_a=[]
    mse_val_a = []
    m_val_a =[]
    for alpha in a:
        k=5
        num_dig = int(cv_df.shape[0])
        size = int(num_dig/k)
        mse_tr =[]
        m_tr=[]
        mse_val = []
        m_val = []

        for i in range (k):
            cv_new = shuffle(cv_df.values)
            test_indices = [x for x in range(i*size,size+(i*size))]
            train_indices = range(0,num_dig)
            #remove the test indices from the train set
            train_indices = [x for x in train_indices if x not in test_indices]
            train_cv = pd.DataFrame(cv_new[train_indices])
            test_cv = pd.DataFrame(cv_new[test_indices])


            ##fit the model on training data

            #split into intitial fetures and target vectors

            feature_train,target_train = split_features_labels_cv(train_cv)
            feature_val,target_val= split_features_labels_cv(test_cv)

            #center the feature vectors for PCA
            centered_train = center(feature_train)
            centered_test = center(feature_val)

            #find the coorelation matrix (240,240) matrix size
            coorelation_matrix_train = coor_c(centered_train)

            # Find the eigenvectors and eigen values of the coorelation matrix
            eig_val,eig_vec = eigen(coorelation_matrix_train)



            # number of PCA features selected=20
            # compute the projections of original image vectors in the selected PC directions

            feat,pc_eigen_val = pc_features(eig_vec,eig_val,centered_train,numdim)
            feat_val,pc_eig_v = pc_features(eig_vec,eig_val,centered_test,numdim)



            ## run the ridge regression on and compute MSEtrain and MISStrain

            # ridge regression
            reg_pred_train, reg_target_train = ridge_reg(feat,target_train,alpha)
            #MSE
            mse_train = meansq_error(reg_pred_train,reg_target_train)
            mse_tr.append(mse_train)
            #MISS
            miss_train = misclass_rate(reg_pred_train,reg_target_train)
            m_tr.append(miss_train)



            #Predict for validation set

            #fit the ridge reg model
            reg_pred_val, reg_target_val = ridge_reg(feat_val,target_val,alpha)

            #MSE
            ms_val = meansq_error(reg_pred_val,reg_target_val)
            mse_val.append(ms_val)
            #MISS
            miss_val = misclass_rate(reg_pred_val,reg_target_val)
            m_val.append(miss_val)


        mse_tr_a.append(np.mean(mse_tr))
        m_tr_a.append(np.mean(m_tr))
        mse_val_a.append(np.mean(mse_val))
        m_val_a.append(np.mean(m_val))

    return mse_tr_a,m_tr_a,mse_val_a,m_val_a


def cv_ridge_kmeans(dataset,no_fold,tune_grid,cnum):

    cv_df = dataset.copy()
    # make k fold splits
    a = tune_grid
    mse_tr_a = []
    m_tr_a=[]
    mse_val_a = []
    m_val_a =[]
    for alpha in a:
        k = no_fold
        num_dig = int(cv_df.shape[0])
        size = int(num_dig/k)
        mse_tr =[]
        m_tr=[]
        mse_val = []
        m_val = []

        for i in range (k):
            cv_new = shuffle(cv_df.values)
            test_indices = [x for x in range(i*size,size+(i*size))]
            train_indices = range(0,num_dig)
            #remove the test indices from the train set
            train_indices = [x for x in train_indices if x not in test_indices]
            train_cv = pd.DataFrame(cv_new[train_indices])
            test_cv = pd.DataFrame(cv_new[test_indices])


            ##fit the model on training data

            #split into intitial fetures and target vectors

            feature_train,target_train = split_features_labels_cv(train_cv)
            feature_val,target_val= split_features_labels_cv(test_cv)

            # use the Kmeans for feature selection
            new_feat = kmeans_algorithm(feature_train.as_matrix(),cnum)
            new_feat_v = kmeans_algorithm(feature_val.as_matrix(),cnum)

            ## run the ridge regression on and compute MSEtrain and MISStrain

            # ridge regression
            reg_pred_train, reg_target_train = ridge_reg(new_feat,target_train,alpha)
            #MSE
            mse_train = meansq_error(reg_pred_train,reg_target_train)
            mse_tr.append(mse_train)
            #MISS
            miss_train = misclass_rate(reg_pred_train,reg_target_train)
            m_tr.append(miss_train)



            #Predict for validation set

            #fit the ridge reg model
            reg_pred_val, reg_target_val = ridge_reg(new_feat_v,target_val,alpha)

            #MSE
            ms_val = meansq_error(reg_pred_val,reg_target_val)
            mse_val.append(ms_val)
            #MISS
            miss_val = misclass_rate(reg_pred_val,reg_target_val)
            m_val.append(miss_val)


        mse_tr_a.append(np.mean(mse_tr))
        m_tr_a.append(np.mean(m_tr))
        mse_val_a.append(np.mean(mse_val))
        m_val_a.append(np.mean(m_val))

    return mse_tr_a,m_tr_a,mse_val_a,m_val_a

### crossvalidation with feature number(PCA features)
def cv_features(dataset,no_fold,tune_grid,alpha):

    cv_df = dataset.copy()
    a = tune_grid
    mse_tr_a = []
    m_tr_a=[]
    mse_val_a = []
    m_val_a =[]
    for dimnum in a:
        k = no_fold
        num_dig = int(cv_df.shape[0])
        size = int(num_dig/k)
        mse_tr =[]
        m_tr=[]
        mse_val = []
        m_val = []

        for i in range (k):
            cv_new = shuffle(cv_df.values)
            test_indices = [x for x in range(i*size,size+(i*size))]
            train_indices = range(0,num_dig)
            #remove the test indices from the train set
            train_indices = [x for x in train_indices if x not in test_indices]
            train_cv = pd.DataFrame(cv_new[train_indices])
            test_cv = pd.DataFrame(cv_new[test_indices])


            ##fit the model on training data

            #split into intitial fetures and target vectors

            feature_train,target_train = split_features_labels_cv(train_cv)
            feature_val,target_val= split_features_labels_cv(test_cv)

            #center the feature vectors for PCA
            centered_train = center(feature_train)
            centered_test = center(feature_val)

            #find the coorelation matrix (240,240) matrix size
            coorelation_matrix_train = coor_c(centered_train)

            # Find the eigenvectors and eigen values of the coorelation matrix
            eig_val,eig_vec = eigen(coorelation_matrix_train)


            # number of PCA features selected=20
            # compute the projections of original image vectors in the selected PC directions

            feat,pc_eigen_val = pc_features(eig_vec,eig_val,centered_train,dimnum)
            feat_val,pc_eig_v = pc_features(eig_vec,eig_val,centered_test,dimnum)



            ## run the ridge regression on and compute MSEtrain and MISStrain

            # ridge regression
            reg_pred_train, reg_target_train = ridge_reg(feat,target_train,alpha)
            #MSE
            mse_train = meansq_error(reg_pred_train,reg_target_train)
            mse_tr.append(mse_train)
            #MISS
            miss_train = misclass_rate(reg_pred_train,reg_target_train)
            m_tr.append(miss_train)



            #Predict for validation set

            #fit the ridge reg model
            reg_pred_val, reg_target_val = ridge_reg(feat_val,target_val,alpha)

            #MSE
            ms_val = meansq_error(reg_pred_val,reg_target_val)
            mse_val.append(ms_val)
            #MISS
            miss_val = misclass_rate(reg_pred_val,reg_target_val)
            m_val.append(miss_val)


        mse_tr_a.append(np.mean(mse_tr))
        m_tr_a.append(np.mean(m_tr))
        mse_val_a.append(np.mean(mse_val))
        m_val_a.append(np.mean(m_val))

    return mse_tr_a,m_tr_a,mse_val_a,m_val_a

### exploring with K-means feature

def kmeans_algorithm (dataframe, n):

    # create a copy of the  2-d image vecotr array
    data_copy = dataframe.copy()

    # shuffe 2-d image vector arrays along the first axis(row)
    np.random.shuffle(data_copy)

    # take the first n image vector arrays, from the shuffeld 2-d image vector array, as initial random codebook vector assignments
    codebook = data_copy[:n]

    # Compute the eucledian disance between vector arrays in dataset and the randomly selected codebook vectors

    # substract each codebook vector from the dataset image vectors.
    # numpy broadcasting allows to substract all dataset image vectors form the codebook vector array even if their shape dont match.
    # Step 1: extend the codebook vector array by adding a new dimension in between the two existing dimensions
    # extending a new dimension allows us to use the rule of broadcasting- array of unequal dimension are when compatable if one the array dimension is 1 here the extended dimension is 1.
    extend_codebook = codebook[:,np.newaxis]
    # Step 2: substract extended codebook vector array form image vector array
    difference = dataset - extend_codebook


    #find the absolute distance from the difference, abs distance = ||difference||
    abs_dist_extended = np.sqrt((difference)**2)

    #reduce the 3-d absolute distance array back into a 2-d array
    abs_dist = abs_dist_extended.sum(axis=2)

    # compute an array of index for each vector in the dataset; the index value will be the nearest index of the nearest codebook vector from the data image vector.
    nearest_codebook = np.argmin(abs_dist,axis=0)


    #assign new codebook vectors, as mean of the dataset image vectors that lie closest to a particular codebook vector assigned above
    new_codebook = np.array([dataset[nearest_codebook==i].mean(axis=0) for i in range(codebook.shape[0])])

    #distance of points from new coodebook vectors taken as features
    extend_new_codebook = new_codebook[:,np.newaxis]
    diff_new_codebook =  dataframe - extend_new_codebook
    abs_new_codebook = np.sqrt((diff_new_codebook)**2)
    abs_new = abs_new_codebook.sum(axis=2).T
    feat_df= pd.DataFrame(abs_new)
    bias = np.full(abs_new.shape[0],1)
    feat_df['bias']=bias
    features = feat_df.as_matrix()
    return features

def cv_features_kmeans(dataset,no_fold,tune_grid,alpha):

    cv_df = dataset.copy()
    # make k fold splits
    a = tune_grid
    mse_tr_a = []
    m_tr_a=[]
    mse_val_a = []
    m_val_a =[]
    for cnum in a:
        k = no_fold
        num_dig = int(cv_df.shape[0])
        size = int(num_dig/k)
        mse_tr =[]
        m_tr=[]
        mse_val = []
        m_val = []

        for i in range (k):
            cv_new = shuffle(cv_df.values)
            test_indices = [x for x in range(i*size,size+(i*size))]
            train_indices = range(0,num_dig)
            #remove the test indices from the train set
            train_indices = [x for x in train_indices if x not in test_indices]
            train_cv = pd.DataFrame(cv_new[train_indices])
            test_cv = pd.DataFrame(cv_new[test_indices])


            ##fit the model on training data

            #split into intitial fetures and target vectors

            feature_train,target_train = split_features_labels_cv(train_cv)
            feature_val,target_val= split_features_labels_cv(test_cv)

            # use the Kmeans for feature selection
            new_feat = kmeans_algorithm(feature_train.as_matrix(),cnum)
            new_feat_v = kmeans_algorithm(feature_val.as_matrix(),cnum)

            ## run the ridge regression on and compute MSEtrain and MISStrain

            # ridge regression
            reg_pred_train, reg_target_train = ridge_reg(new_feat,target_train,alpha)
            #MSE
            mse_train = meansq_error(reg_pred_train,reg_target_train)
            mse_tr.append(mse_train)
            #MISS
            miss_train = misclass_rate(reg_pred_train,reg_target_train)
            m_tr.append(miss_train)



            #Predict for validation set

            #fit the ridge reg model
            reg_pred_val, reg_target_val = ridge_reg(new_feat_v,target_val,alpha)

            #MSE
            ms_val = meansq_error(reg_pred_val,reg_target_val)
            mse_val.append(ms_val)
            #MISS
            miss_val = misclass_rate(reg_pred_val,reg_target_val)
            m_val.append(miss_val)


        mse_tr_a.append(np.mean(mse_tr))
        m_tr_a.append(np.mean(m_tr))
        mse_val_a.append(np.mean(mse_val))
        m_val_a.append(np.mean(m_val))

    return mse_tr_a,m_tr_a,mse_val_a,m_val_a



''' Function calls and regression models'''

### Data-preperation:
random.seed(1)
filepath = "mfeat-pix.txt"
dataset = np.loadtxt(filepath)
data_copy = dataset.copy()

training, test = training_test_split(data_copy)
feature_train,target_train = split_features_labels(training)
feature_test,target_test =split_features_labels(test)

### PCA features and linear regression

#center the feature vectors for PCA
centered_train = center(feature_train)
centered_test = center(feature_test)

#find the coorelation matrix (240,240) matrix size
coorelation_matrix_train = coor_c(centered_train)

# Find the eigenvectors and eigen values of the coorelation matrix
eig_val,eig_vec = eigen(coorelation_matrix_train)


# number of PCA features selected=240
# compute the projections of original image vectors in the selected PC directions

feat,pc_eigen_val = pc_features(eig_vec,eig_val,centered_train,240)
feat_val,pc_eig_v = pc_features(eig_vec,eig_val,centered_test,240)

## run the regression on and compute MSEtrain and MISStrain

# ridge regression
reg_pred_train, reg_target_train = ridge_reg(feat,target_train,0)
#MSE
mse_train = meansq_error(reg_pred_train,reg_target_train)

#MISS
miss_train = misclass_rate(reg_pred_train,reg_target_train)

#Predict for validation set
#fit the reg model

reg_pred_val, reg_target_val = ridge_reg(feat_val,target_test,0)
#MSE

ms_val = meansq_error(reg_pred_val,reg_target_val)
#MISS

miss_val = misclass_rate(reg_pred_val,reg_target_val)



print('mse_train:',mse_train)
print('miss_train:',miss_train)
print('mse_test:',ms_val)
print('miss_test:',miss_val)


#explore the eigen values graphically to see eigen value spectrum againt number of PC
plt.plot(eig_val)
plt.plot([50, 50], [-100, 800], '--', lw=1 ,color ='r')
plt.xlabel('Principle Component dimensions')
plt.ylabel('eigen value')
plt.title('Eigen value spectrum of PC dimensions')

#explore the PC below 50
plt.plot(eig_val[0:50])
plt.plot([20, 20], [-100, 800], '--', lw=1 ,color ='r')
plt.xlabel('Principle Component dimensions')
plt.ylabel('eigen value')
plt.title('Eigen value spectrum of PC dimensions')


# linear regression with 20 PCA features

# number of PCA features selected=20
# compute the projections of original image vectors in the selected PC directions

feat,pc_eigen_val = pc_features(eig_vec,eig_val,centered_train,20)
feat_val,pc_eig_v = pc_features(eig_vec,eig_val,centered_test,20)

## run the regression on and compute MSEtrain and MISStrain

# ridge regression
reg_pred_train, reg_target_train = ridge_reg(feat,target_train,0)
#MSE
mse_train = meansq_error(reg_pred_train,reg_target_train)

#MISS
miss_train = misclass_rate(reg_pred_train,reg_target_train)

#Predict for validation set
#fit the reg model

reg_pred_val, reg_target_val = ridge_reg(feat_val,target_test,0)
#MSE

ms_val = meansq_error(reg_pred_val,reg_target_val)
#MISS

miss_val = misclass_rate(reg_pred_val,reg_target_val)

print('mse_train:',mse_train)
print('miss_train:',miss_train)
print('mse_test:',ms_val)
print('miss_test:',miss_val)


#cv for optimal number of PCA features
mse_tr_a,m_tr_a,mse_val_a,m_val_a  = cv_features(training,5,[x for x in range(10,240,20)],0)

#plotting the CV evaluation metrics
a = [x for x in range(10,240,20)]
i_mse = np.argmin(mse_val_a)
i_miss = np.argmin(m_val_a)
l1,=plt.plot(a,m_tr_a,color='blue',label="l1")
l2,=plt.plot(a,m_val_a,color='red',label="l2")
l3,=plt.plot(a,mse_tr_a,"--",color='blue',label="l3")
l4,=plt.plot(a,mse_val_a,"--",color = 'red',label="l4")
plt.plot(a[i_mse],mse_tr_a[i_mse],'o',color='k')
plt.plot(a[i_miss],m_tr_a[i_miss],'x',color='k')
plt.xlabel('number of features')
plt.ylabel('error rates')
plt.title('Train and Validate error rates for number of features')
plt.legend([l1,l2,l3,l4], ["MISS(validate)", "MISS(train)","MSE(validate)","MSE(train)"])
plt.show()

#cv for optimal alpha value
mse_tr_a,m_tr_a,mse_val_a,m_val_a = cv_ridge(training,5,np.arange(0.0, 1.0, 0.1) ,90)

#plotting the CV evaluation metrics
a = np.arange(0.0, 1.0, 0.1)
i_mse = np.argmin(mse_tr_a)
i_miss = np.argmin(m_tr_a)
l1,=plt.plot(a,m_tr_a,color='blue',label="l1")
l2,=plt.plot(a,m_val_a,color='red',label="l2")
l3,=plt.plot(a,mse_tr_a,"--",color='blue',label="l3")
l4,=plt.plot(a,mse_val_a,"--",color = 'red',label="l4")
plt.plot(a[i_mse],mse_tr_a[i_mse],'o',color='k')
plt.plot(a[i_miss],m_tr_a[i_miss],'x',color='k')
plt.xlabel('ridge regression tuning parameter(alpha)')
plt.ylabel('error rates')
plt.title('Train and Validate error rates for choice of alpha')
plt.legend([l1,l2,l3,l4], ["MISS(validate)", "MISS(train)","MSE(validate)","MSE(train)"])
plt.show()

#model with optimal features and alpha
feat,pc_eigen_val = pc_features(eig_vec,eig_val,centered_train,90)
feat_val,pc_eig_v = pc_features(eig_vec,eig_val,centered_test,90)

## run the regression on and compute MSEtrain and MISStrain

# ridge regression
reg_pred_train, reg_target_train = ridge_reg(feat,target_train,0.3)
#MSE
mse_train = meansq_error(reg_pred_train,reg_target_train)

#MISS
miss_train = misclass_rate(reg_pred_train,reg_target_train)

#Predict for validation set
#fit the reg model

reg_pred_val, reg_target_val = ridge_reg(feat_val,target_test,0.3)
#MSE

ms_val = meansq_error(reg_pred_val,reg_target_val)
#MISS

miss_val = misclass_rate(reg_pred_val,reg_target_val)

print('mse_train:',mse_train)
print('miss_train:',miss_train)
print('mse_test:',ms_val)
print('miss_test:',miss_val)




## linear regression with k means feature


# linear regression with k=40
new_feat = kmeans_algorithm(feature_train.as_matrix(),100)
new_feat_v = kmeans_algorithm(feature_test.as_matrix(),100)

## run the ridge regression on and compute MSEtrain and MISStrain

# ridge regression
reg_pred_train, reg_target_train = ridge_reg(new_feat,target_train,0)
#MSE
mse_train = meansq_error(reg_pred_train,reg_target_train)

#MISS
miss_train = misclass_rate(reg_pred_train,reg_target_train)

#Predict for test set

#fit the ridge reg model
reg_pred_val, reg_target_val = ridge_reg(new_feat_v,target_test,0)

#MSE
ms_val = meansq_error(reg_pred_val,reg_target_val)

#MISS
miss_val = misclass_rate(reg_pred_val,reg_target_val)

print('mse_train:',mse_train)
print('miss_train:',miss_train)
print('mse_test:',ms_val)
print('miss_test:',miss_val)

#cross validation for optimal number of centroids
mse_tr_a,m_tr_a,mse_val_a,m_val_a  = cv_features_kmeans(training,5,[x for x in range(50,1000,50)],0)

#plotting the CV evaluation metrics
a = [x for x in range(50,1000,50)]
i_mse = np.argmin(mse_tr_a)
i_miss = np.argmin(m_tr_a)
l1,=plt.plot(a,m_tr_a,color='blue',label="l1")
l2,=plt.plot(a,m_val_a,color='red',label="l2")
l3,=plt.plot(a,mse_tr_a,"--",color='blue',label="l3")
l4,=plt.plot(a,mse_val_a,"--",color = 'red',label="l4")
plt.plot(a[i_mse],mse_tr_a[i_mse],'o',color='k')
plt.plot(a[i_miss],m_tr_a[i_miss],'x',color='k')
plt.xlabel('number of features')
plt.ylabel('error rates')
plt.title('Train and Validate error rates for number of features using Kmeans')
plt.legend([l1,l2,l3,l4], ["MISS(validate)", "MISS(train)","MSE(validate)","MSE(train)"])
plt.show()

#crossvalidation for optimal alpha
mse_tr_a,m_tr_a,mse_val_a,m_val_a = cv_ridge_kmeans(training,5,np.arange(0.0, 1.0, 0.1),400)

#plotting the CV evaluation metrics
a = np.arange(0.0, 1.0, 0.1)
i_mse = np.argmin(mse_tr_a)
i_miss = np.argmin(m_tr_a)
l1,=plt.plot(a,m_tr_a,color='blue',label="l1")
l2,=plt.plot(a,m_val_a,color='red',label="l2")
l3,=plt.plot(a,mse_tr_a,"--",color='blue',label="l3")
l4,=plt.plot(a,mse_val_a,"--",color = 'red',label="l4")
plt.plot(a[i_mse],mse_tr_a[i_mse],'o',color='k')
plt.plot(a[i_miss],m_tr_a[i_miss],'x',color='k')
plt.xlabel('ridge regression tuning parameter(alpha)')
plt.ylabel('error rates')
plt.title('Train and Validate error rates for choice of alpha')
plt.legend([l1,l2,l3,l4], ["MISS(train)", "MISS(validate)","MSE(train)","MSE(validate)"])
plt.show()
