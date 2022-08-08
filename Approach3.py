# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 15:35:30 2022

@author: sasha
"""

# Approach3 
from surprise import Reader, SVD, Dataset, SVDpp
import numpy as np
import pandas as pd

from surprise.model_selection import cross_validate

train_ = np.loadtxt('train.dat')
test_ = np.loadtxt('test.dat')

read = Reader(rating_scale = (1.0,5.0))
Training = pd.DataFrame(train_ ,columns=['UserID','ItemID','Ratings','Timestamp'])
Testing = pd.DataFrame(test_ ,columns=['UserID','ItemID'])
#Training = pd.DataFrame(train_[0:80000,:],columns=['UserID','ItemID','Ratings','Timestamp'])
#Testing = pd.DataFrame(train_[80001:,:],columns=['UserID','ItemID','Ratings','Timestamp'])
Tr = Dataset.load_from_df(Training[['UserID','ItemID','Ratings']],read)

# Training = pd.DataFrame(train_[0:80000,:],columns=['UserID','ItemID','Ratings','Timestamp'])
# Tr = Dataset.load_from_df(Training,read)
# Testing = pd.DataFrame(train_[80001:,:],columns=['UserID','ItemID','Ratings','Timestamp'])
# Ts = Dataset.load_from_df(Testing,read)



# Used = Training.append(Testing.iloc[:,0:2])
# Dat = Dataset.load_from_df(Used.iloc[:,0:3], read)

model = SVD(n_factors=20,verbose=False, n_epochs=30)
cross_validate(model, Tr, measures = ['RMSE'], cv=10, verbose=False)

model.fit(Tr.build_full_trainset())


Predictions=[]
Str = ''
#model.predict(247,950)
for i in range(0,len(Testing)):
    Predictions.append(model.predict(Testing.iloc[i,0],Testing.iloc[i,1]).est)
    Str = Str + str(model.predict(Testing.iloc[i,0],Testing.iloc[i,1]).est)+'\n'



"""
Name= 'Prediction2.dat'
F=open(Name,'w')
F.write(Str)
F.close()
del F
"""

