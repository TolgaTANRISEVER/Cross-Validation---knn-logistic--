# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 16:29:51 2022

@author: Tolga
"""

from sklearn.datasets import load_iris
import pandas as pd 
import numpy as np 
#%% 
iris=load_iris()

x=iris.data
y=iris.target

#%%
x=(x-np.min(x))/(np.max(x)-np.min(x))
#%%
from sklearn.model_selection import train_test_split
x_train , x_test, y_train, y_test=train_test_split(x,y,test_size=0.3)
#%%knn
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=9)

#%% k fold cv k =10
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=knn,X=x_train,y=y_train,cv=10)
print("avarage accuracy : ",np.mean(accuracies))
print("avarage std : ",np.std(accuracies))
#%%
knn.fit(x_train,y_train)
print("test accuracy : ",knn.score(x_test,y_test))
#%% grid seach cross validation 
from sklearn.model_selection import GridSearchCV
grid={"n_neighbors":np.arange(1,50)}

knn=KNeighborsClassifier()

knn_cv=GridSearchCV(knn,grid,cv=10)
knn_cv.fit(x_train,y_train)
#%% print hyperparameter 
print("tunedd hyperparameter K: ",knn_cv.best_params_)
print("for tuned prameter :best accuracy K: ",knn_cv.best_score_)
#%% grid search CV with logistic regresiions 
from sklearn.linear_model import LogisticRegression
x1=x[:100,:]
y1=y[:100]

grid ={'C':np.logspace(-3,3,7),'penalty':['l1','l2']}# if C very high=overfitting else underfitting

logreg=LogisticRegression()
logreg_cv=GridSearchCV(logreg,grid,cv=10)
logreg_cv.fit(x1,y1)

#%%print hyperparemeter
print("hyperparameters",logreg_cv.best_params_)
print("hyperparameters",logreg_cv.best_score_)

#%% logistic reg new ex
x1=(x1-np.min(x1))/(np.max(x1)-np.min(x1))
x1_train,x1_test,y1_train,y1_test=train_test_split(x1,y1,test_size=0.3)

grid ={'C':np.logspace(-3,3,7),'penalty':['l1','l2']}# if C very high=overfitting else underfitting

logreg=LogisticRegression()
logreg_cv=GridSearchCV(logreg,grid,cv=10)
logreg_cv.fit(x1_train,y1_train)
print("hyperparameters",logreg_cv.best_params_)
print("hyperparameters",logreg_cv.best_score_)
#%%
      
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1, penalty='l1', solver='liblinear')
lr.fit(x1_train,y1_train)
print("test accuracy {}".format(lr.score(x1_test,y1_test)))





