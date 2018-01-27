import numpy as np
from sklearn import datasets
import math
import matplotlib.pyplot as plt

# Load dataset
dataset = datasets.load_boston()

features = dataset.data
labels = dataset.target

Nsplit = 50
# Training set
X_train, y_train = features[:-Nsplit], labels[:-Nsplit]
# Test set
X_test, y_test = features[-Nsplit:], labels[-Nsplit:]

n = round(len(X_train)*0.9)
X_v, y_v = X_train[n:], y_train[n:]
X_train, y_train = X_train[:n], y_train[:n]

mean_X = X_train.mean(axis=0)
std_X = X_train.std(axis=0)
X_train_n = (X_train - mean_X) / std_X
X_v_n = (X_v - mean_X) / std_X
X_test_n = (X_test - mean_X) / std_X

nv = len(X_v)
nt = len(X_test)
X_train_a = np.ones((n, 1))
X_v_a = np.ones((nv, 1))
X_test_a = np.ones((nt, 1))
X_train_a = np.concatenate((X_train_a, np.asarray(X_train_n)), axis=1)
X_v_a = np.concatenate((X_v_a, np.asarray(X_v_n)), axis=1)
X_test_a = np.concatenate((X_test_a, np.asarray(X_test_n)), axis=1)

RMSE = []
wlist = []
for i in range(6):
    q = i/10
    w = np.dot(np.linalg.inv(np.dot(np.transpose(X_train_a), X_train_a) + n * q * np.eye(len(X_train[0])+1)),
               np.dot(np.transpose(X_train_a), y_train))
    wlist.append(w)
    RMSE.append(math.sqrt(np.mean((X_v_a.dot(w) - [y_v]) ** 2)))
lowest = np.min(RMSE)
num = list.index(RMSE,lowest)
test_error = math.sqrt(np.mean((X_test_a.dot(wlist[num]) - [y_test]) ** 2))
print('The lowest RMSE error is', lowest)
print('The corresponding value for λ is', num/10)
print('The test error is', test_error)
