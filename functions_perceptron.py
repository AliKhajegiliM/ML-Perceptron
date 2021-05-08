# -*- coding: utf-8 -*-
"""
Created on Wed May  5 19:37:04 2021

@author: user
"""

import numpy as np
import pandas as pd

def load_data(file_path):
    X,y = None,None
    df = pd.read_csv(file_path)
    X = df.drop('label',axis=1)
    y = df['label']
    return X, y




#Model Details:
class PerceptronModel:
    def __init__(self, alpha = 0.1, total_iter = 5):
        self.alpha = alpha
        self.total_iter = total_iter
###################################################################################################        
    # multi-class perceptron algorithm using stochastic gradient descent (packet approach)
    def Perceptron_sgd(self, X, y):
        n, d = len(X), len(X[0])
        y = y.reshape(n,1)
        W = np.random.randn(d,10)
        y_hat = None
        lowest_cost = float('inf')
        Hist_cost=np.zeros((self.total_iter,1))
        # repeat the procedure:
        for n_iter in range(self.total_iter):
            cost = 0.0
            for i_smpl in range(n):
                #find the estimated class for the i-th sample
                y_hat = np.argmax(np.dot(W.T,X[i_smpl,:].T))
                # if it is missclassified update the weight matrix using SGD
                if y_hat != y[i_smpl]:
                    X_i = X[[i_smpl] ,:].T
                    W[:, y[i_smpl]] = W[:, y[i_smpl]] + self.alpha*X_i
                    W[: , [y_hat]] = W[:, [y_hat]] - self.alpha*X_i
                    cost=cost+np.dot(X_i.T , W[:, y[i_smpl]]-W[: , [y_hat]])
            #find the Weight matrix corrospding to the lowest achieved cost: 
            Hist_cost[n_iter]=cost
            if cost <= lowest_cost:
                    lowest_cost = cost
                    W_best = W
                    
            print('iteration:', n_iter+1, ', cost:', cost[0]/n)
                
        return W_best, Hist_cost
##################################################################################################
    # multi-class (batch) perceptron algorithm using gradient descent (packet approach)
    def Perceptron_batch(self, X, y):
        n, d = len(X), len(X[0])
        y = y.reshape(n,1)
        W = np.random.randn(d,10)
        lowest_cost = float('inf')
        Hist_cost=np.zeros((self.total_iter,1))
        # repeat the procedure
        for n_iter in range(self.total_iter):
            cost = 0.0
            gradient = np.zeros((d,1))
            y_hat = np.argmax(np.dot(W.T,X.T), axis=0)
            # compute the gradient
            for i_smpl in range(n):
                if y_hat[i_smpl] != y[i_smpl]:
                    gradient = gradient + X[[i_smpl] ,:].T
            # update the weight matrix and cost:                
            for i_smpl in range(n):
                if y_hat[i_smpl] != y[i_smpl]:
                    W[: , y[i_smpl]]= W[: , y[i_smpl]] + self.alpha*gradient
                    W[:, y_hat[i_smpl]]= W[:, y_hat[i_smpl]] - self.alpha*gradient.T
                    cost=cost+abs(np.dot(X[[i_smpl], :] , W[:, y[i_smpl]]-W[: , [y_hat[i_smpl]]]))
            #find the Weight matrix corrospding to the lowest achieved cost:    
            Hist_cost[n_iter]=cost
            if cost <= lowest_cost:
                    lowest_cost = cost
                    W_best = W
                    
            print('iteration:', n_iter+1, ', cost:', cost[0]/n)
            
        return W_best, Hist_cost
####################################################################################################    
