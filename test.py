# -*- coding: utf-8 -*-
"""
Created on Sat May  8 14:43:43 2021

@author: user
"""
import functions_perceptron as fp
import matplotlib.pyplot as plt
import numpy as np

#data path
mnist_train_path = "mnist_train.csv"
mnist_test_path = "mnist_test.csv"

#load training and test data
X, y = fp.load_data(mnist_train_path)
X = X.to_numpy()/256.0
y = y.to_numpy()
X_test, y_test=fp.load_data(mnist_test_path)
X_test=X_test.to_numpy()
#evaluations:
total_iter=150 #number of iterations
alpha = 0.01 #learning rate
model=fp.PerceptronModel(alpha = alpha, total_iter = total_iter)
# perceptron algorithm using stochastic gradient descent:
#W_best, Hist_cost = model.Perceptron_sgd(X, y)
# perceptron algorithm using gradient descent:
W_best, Hist_cost = model.Perceptron_batch(X, y)

#plot the cost vs iteration curve
plt.figure(figsize=(8, 6), dpi=96)
plt.plot(Hist_cost)
plt.ylabel('Cost')
plt.xlabel('Iteration')
plt.grid(linestyle='-', axis='both')
plt.show()

#accuracy evaulation:
y_train_hat=np.argmax(np.dot(W_best.T,X.T), axis=0)
y_test_hat=np.argmax(np.dot(W_best.T,X_test.T), axis=0)
print('test accurcy:', sum(y_test_hat==y_test.T)/len(y_test))
print('train accuracy:', sum(y_train_hat==y.T)/len(y))