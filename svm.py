# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import math
import cvxopt
from statistics import mean
import time

train_data=np.loadtxt("train_data.txt")
test_data=np.loadtxt("test_data.txt")
train_label=np.loadtxt("train_label.txt")
test_label=np.loadtxt("test_label.txt")

#Normalizing the train data
train_mean = np.mean(train_data,axis = 0)
train_std = np.std(train_data,axis = 0)
train_std.shape
train_normalized = [(train_data[:,i] - train_mean[i])/train_std[i] for i in range(train_data.shape[1])]
train_normalized = np.array(train_normalized).T

#Normalizing the test data
test_normalized = [(test_data[:,i] - train_mean[i])/train_std[i] for i in range(test_data.shape[1])]
test_normalized = np.array(test_normalized).T

print(f"Shape of the normalized train and test data is : {train_normalized.shape} and {test_normalized.shape}\n")
print(f"Mean and standard deviation of the 3rd feature are: {train_mean[2]} and {train_std[2]}")
print(f"Mean and standard deviation of the 10th feature are: {train_mean[9]} and {train_std[9]}")

from cvxopt import matrix
from cvxopt.solvers import qp,options
train_label = train_label.reshape(-1,1)
train_data_new = train_label*train_data
H = (train_data_new@train_data_new.T).astype(float)

P = matrix(H)
q = matrix(-np.ones((train_normalized.shape[0],1)))
G = matrix(-np.eye(train_normalized.shape[0]))
h = matrix(np.zeros(train_normalized.shape[0]))
A = matrix(train_label.reshape(1,1000))
b = matrix(np.zeros((1,1)))

options['show_progress'] = False
alpha = qp(P,q,G,h,A,b)['x']
W = ((train_label*alpha).T@train_normalized).reshape(-1,1)
bias = train_label - train_normalized@W

from cvxopt import matrix
from cvxopt.solvers import qp,options

#Using the dual form of the SVM
def train_svm(train_data, train_label, C):
  """Train linear SVM (primal form)

  Argument:
    train_data: N*D matrix, each row as a sample and each column as a feature
    train_label: N*1 vector, each row as a label
    C: tradeoff parameter (on slack variable side)

  Return:
    w: feature vector (column vector)
    b: bias term
  """
  train_label = train_label.reshape(-1,1)
  train_data_new = train_label*train_data
  H = (train_data_new @ train_data_new.T).astype(float)

  P = matrix(H)
  q = matrix(-np.ones((train_data.shape[0],1)))
  G = matrix(np.vstack((-np.eye(train_data.shape[0]),np.eye(train_data.shape[0]))))
  h = matrix(np.hstack((np.zeros(train_data.shape[0]),C*np.ones(train_data.shape[0]))))
  A = matrix(train_label.reshape(1,-1))
  b = matrix(np.zeros(1))

  alpha = qp(P,q,G,h,A,b)['x']
  W = ((train_label*alpha).T @ train_data).reshape(-1,1)
  bias = train_label - train_data @ W
  return W,bias

def test_svm(test_data, test_label, w, b):
  """Test linear SVM

  Argument:
    test_data: M*D matrix, each row as a sample and each column as a feature
    test_label: M*1 vector, each row as a label
    w: feature vector
    b: bias term

  Return:
    test_accuracy: a float between [0, 1] representing the test accuracy
  """
  b = np.tile(b[0],len(test_label)).reshape(-1,1)
  test_label = test_label.reshape(-1,1)
  Y_pred = test_data @ w + b
  thre = np.percentile(Y_pred,50)
  Y_pred_binary = np.where(Y_pred>thre,1,-1)
  correct = [i==j for i,j in zip(test_label,Y_pred_binary)]
  accuracy = np.count_nonzero(np.array(correct))/len(correct)
  return accuracy

regularizer = [4**i for i in range(-6,7)]

for i,C in enumerate(regularizer):
  fold_accuracy,training_time,test_accuracy = [],[],[]
  print(f"{i+1}. C = {C}")
  for i in range(1,6):
    #Preparing training and testing data for Cross-validation
    traindata = np.concatenate((train_normalized[:200*(i-1),:], train_normalized[200*i:,:]),axis = 0)
    trainlabel = np.concatenate((train_label[:200*(i-1),:], train_label[200*i:,:]),axis = 0)
    testdata = train_normalized[200*(i-1):200*i,:]
    testlabel = train_label[200*(i-1):200*(i)]    
    #Time taken for Training
    start = time.time()
    W,bias = train_svm(traindata,trainlabel,C)
    duration = time.time() - start

    #Cross-validation accuracy
    acc = test_svm(testdata,testlabel,W,bias)
    #Test accuracy
    test_acc = test_svm(test_normalized,test_label,W,bias)

    fold_accuracy.append(acc)
    test_accuracy.append(test_acc)
    training_time.append(duration)
  mean_train_accuracy = mean(fold_accuracy)
  mean_test_accuracy = mean(test_accuracy)
  mean_training_time = mean(training_time)
  print(f"Avg. Training Accuracy: {mean_train_accuracy} Avg. Test Accuracy: {mean_test_accuracy}\n")
  print(f"Avg. Training Time = {mean_training_time} seconds")
  print('********************************************************')

Pending - Look into the time taken at higher values of C

x_neg = np.array([[3,4],[1,4],[2,3]])
y_neg = np.array([-1,-1,-1])
x_pos = np.array([[6,-1],[7,-1],[5,-3]])
y_pos = np.array([1,1,1])
x1 = np.linspace(-10,10)
x = np.vstack((np.linspace(-10,10),np.linspace(-10,10)))
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers

#Data for the next section
X = np.vstack((x_pos, x_neg))
y = np.concatenate((y_pos,y_neg))

#Parameters guessed by inspection
w = np.array([1,-1]).reshape(-1,1)
b = -3

m,n = X.shape
y = y.reshape(-1,1) * 1.
X_dash = y * X
H = np.dot(X_dash , X_dash.T) * 1.

#Converting into cvxopt format
P = cvxopt_matrix(H)
q = cvxopt_matrix(-np.ones((m, 1)))
G = cvxopt_matrix(-np.eye(m))
h = cvxopt_matrix(np.zeros(m))
A = cvxopt_matrix(y.reshape(1, -1))
b = cvxopt_matrix(np.zeros(1))

#Setting solver parameters (change default to decrease tolerance) 
cvxopt_solvers.options['show_progress'] = False
cvxopt_solvers.options['abstol'] = 1e-10
cvxopt_solvers.options['reltol'] = 1e-10
cvxopt_solvers.options['feastol'] = 1e-10

#Run solver
sol = cvxopt_solvers.qp(P, q, G, h, A, b)
alphas = np.array(sol['x'])

w = ((y * alphas).T @ X).reshape(-1,1)

#Selecting the set of indices S corresponding to non zero parameters
S = (alphas > 1e-4).flatten()

#Computing b
b = y[S] - np.dot(X[S], w)

