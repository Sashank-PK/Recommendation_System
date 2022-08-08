# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 11:55:40 2022

@author: sasha
"""
from numba import jit, cuda


import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from surprise import SVD, Dataset


class Recommend:
    
    def __init__(self):
        format_ = np.loadtxt('format.dat')
        train_ = np.loadtxt('train.dat')
        test_ = np.loadtxt('test.dat')
        self.UserConsistency = False
        self.ItemConsistency = False
        
        TestUsers = set(np.unique(test_[:,0]))
        TrainUsers = set(np.unique(train_[:,0]))
        TestItems = set(np.unique(test_[:,1]))
        TrainItems = set(np.unique(train_[:,1]))
        
        
        if len(TrainUsers.intersection(TestUsers)) == len(TestUsers):
            self.UserConsistency = True #print('All test set users are available in training set')
        else :
            #print('Missing Users found. Use Mis_Users variable')
            self.Mis_Users = TestUsers.difference(TrainUsers)
        
        if len(TrainItems.intersection(TestItems)) == len(TestItems):
            self.ItemConsistency = True #print('All test set items are available in training set')
        else :
            #print('Missing Items found. Use Mis_Items variable')   
            self.Mis_Items = TestItems.difference(TrainItems)
        
        
        train_Head = ['UserID','ItemID','Ratings','Timestamp']
        test_Head = ['UserID','ItemID']

        #self.Training = pd.DataFrame(train_,columns=train_Head)
        #self.Testing = pd.DataFrame(test_,columns=test_Head)
        
        self.Training = pd.DataFrame(train_[0:80000,:],columns=train_Head)
        self.Testing = pd.DataFrame(train_[80001:,0:2],columns=test_Head)
        self.answers = pd.DataFrame(train_[80001:,2],columns=['ans'])
        self.UvI = 0
        
        return
    
    def MakeUserVsItem(self):
        Tests = self.Testing.copy()
        Tests['Ratings']=0
        Combine = Tests.append(self.Training.iloc[:,0:-1])
        self.UvI = Combine.pivot('UserID','ItemID','Ratings')
        self.UvI.fillna(0,inplace=True)
        self.Users = self.UvI.index.to_list()
        self.Items = self.UvI.columns.to_list()
        return
    
    def verifyUvI(self):
        crossverify = []
        for i in range(0,self.UvI.shape[0]):
            if self.UvI[self.Testing.iloc[i,1]][self.Testing.iloc[i,0]] == 0:
                crossverify.append(True)
            else:
                crossverify.append(False)
        
        if not all(crossverify):
            print('All testing segments are proper')
        
        return
        
    def GetUvIMatrix(self):
        if not self.UvI ==0 :
            self.MakeUserVsItem()
        
        self.verifyUvI()
        return self.UvI
    
    def CosineSimilarity(self):
        sparsematrix = sp.csr_matrix(self.UvI)
        self.UUSim = cosine_similarity(sparsematrix)
        self.IISim = cosine_similarity(sparsematrix.T)
        return
    #@jit(target="cuda")
    #@cuda.jit
    def JaccardSimilarity(self):
        self.UUSim = np.zeros((self.UvI.shape[0],self.UvI.shape[0]))
        self.IISim = np.zeros((self.UvI.shape[1],self.UvI.shape[1]))
        # UU Matrix
        for i in range(0,self.UvI.shape[0]):
            for j in range (i,self.UvI.shape[0]):
                A = set(self.UvI.T[self.UvI.T.iloc[:,i]>0].index.to_list())
                B = set(self.UvI.T[self.UvI.T.iloc[:,j]>0].index.to_list())
                self.UUSim[i][j] = len(A & B) / len(A | B)
        return
    #@jit(target="cuda")
    #@cuda.jit
    def PearsonSimilarity(self):
        #user user matrix
        self.UUSim = np.zeros((self.UvI.shape[0],self.UvI.shape[0]))
        self.IISim = np.zeros((self.UvI.shape[1],self.UvI.shape[1]))
        #print('ok-1')
        for i in range(0,self.UvI.shape[0]):
            for j in range (i,self.UvI.shape[0]):
                self.UUSim[i][j] = pearsonr(self.UvI.iloc[i,:],self.UvI.iloc[j,:])[0]
                #self.UUSim[j][i] = self.UUSim[i][j]
        #print('ok-2')
        for i in range(0,self.UvI.shape[1]):
            for j in range(i,self.UvI.shape[1]):
                self.IISim[i][j] = pearsonr(self.UvI.iloc[:,i],self.UvI.iloc[:,j])[0]
                #self.IISim[j][i] = self.IISim[i][j]
        return
    
    def PredictRating(self):
        RatingPredict = []
        u_ = self.UvI.mean().mean()
        
        for i in range(0,5):
            # bxi = u + bx + bi  == user ratings avg + item rating avg - u
            x_ = self.Testing.iloc[i,0] #x
            i_ = self.Testing.iloc[i,1] # i 
            bxi = self.UvI.T.mean().loc[x_]-u_ + self.UvI.mean().loc[i_]-u_ + u_
            
            #print(bxi)
            
            Rxj = np.array(self.UvI.loc[x_,:]) # Ratings of user to all movies
            
            i_ind = self.UvI.columns.to_list().index(i_) 
            Sij = self.IISim[i_ind] #Similarity scores of Item v Item for given Item I 
            
            bx = np.array(self.UvI.mean())+self.UvI.T.mean().loc[x_]
            bx = bx - u_
            #print(np.max(bx))
            
            Numer = np.sum(Sij * (Rxj - bx))
            Denom = np.sum(Sij)
            #print(Numer/Denom)
            
            RatingPredict.append(bxi+(Numer/Denom))
            
        return RatingPredict
    
    """
    def Predict2Rating(self,user,n_users=10):
        UserSim = pd.DataFrame(self.UUSim,index=self.Users,columns=self.Users)
        ItemSim = pd.DataFrame(self.UUSim,index=self.Items,columns=self.Items)
        for i in range(0,len(self.Testing))
        SimUsers = UserSim.sort_values(by=user,ascending=False).index[1:n_users+1]
        
        
        
        return


Sys = Recommend()
Sys.MakeUserVsItem()
Sys.CosineSimilarity()
#ans = Sys.Predict2Rating() """



#Checking if all the test users and movies are available in Training data


"""
del TestUsers,TrainUsers, TestItems, TrainItems



del test_, train_, train_Head, test_Head
#Training.to_csv('Training.csv')
#Testing.to_csv('Testing.csv')

UvI_=Training.pivot('UserID','ItemID','Ratings')
UvI_.fillna(0,inplace=True)

sparse = sp.csr_matrix(UvI_)

user_similarity = cosine_similarity(sparse)
Item_similarity = cosine_similarity(sparse.T)

import datetime

#datetime.datetime.utcfromtimestamp()"""