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



#a
n = len(X_train)
nt = len(X_test)
RMSE = []
RMSEt = []
X_train_a0 = np.ones((n, 1))
X_test_a0 = np.ones((nt, 1))

for j in range(5):
    X_train_a = X_train_a0
    X_test_a = X_test_a0
    for i in range(1,j+1):
        train_add = np.asarray(X_train) ** i
        test_add = np.asarray(X_test) ** i
        mean_X = train_add.mean(axis=0)
        std_X = train_add.std(axis=0)
        train_add = (train_add - mean_X) / std_X
        test_add = (test_add - mean_X) / std_X
        X_train_a = np.concatenate((X_train_a, train_add), axis=1)
        X_test_a = np.concatenate((X_test_a, test_add), axis=1)
    w = np.dot(np.linalg.pinv(np.dot(np.transpose(X_train_a), X_train_a)),
                np.dot(np.transpose(X_train_a), y_train))
    train_error = math.sqrt(np.mean((X_train_a.dot(w) - [y_train]) ** 2))
    test_error = math.sqrt(np.mean((X_test_a.dot(w) - [y_test]) ** 2))
    RMSE.append(train_error)
    RMSEt.append(test_error)
    # print('Order', j,':')
    # print('The weight vector is ', w)
    # print('The bias term is ', w[0])
    # print('The train error is ', train_error)
    # print('The test error is ', test_error)
print('train', RMSE)
print('test', RMSEt)
x1 = np.linspace(0,4,5)
plt.subplot(2,1,1)
plt.plot(x1, RMSE, label="train")
plt.plot(x1, RMSEt, label="test")
plt.xlabel("Order")
plt.ylabel("RMSE")
plt.title("RMSE for different orders")
plt.legend()
plt.show()

#b
RMSE = []
RMSEt = []
for i in range(1,6):
    nn = round(n*i/5)
    X_train, y_train = features[:-Nsplit], labels[:-Nsplit]
    X_test, y_test = features[-Nsplit:], labels[-Nsplit:]
    X_train, y_train = X_train[:nn], y_train[:nn]
    mean_X = X_train.mean(axis=0)
    std_X = X_train.std(axis=0)
    for i in range(len(std_X)):
        if std_X[i] == 0:
            std_X[i] = 1
    X_train = (X_train - mean_X) / std_X
    X_test = (X_test - mean_X) / std_X
    X_train_a = np.ones((nn, 1))
    X_train_a = np.concatenate((X_train_a, X_train), axis=1)
    X_test_a = np.concatenate((X_test_a0, X_test), axis=1)
    w = np.dot(np.linalg.pinv(np.dot(np.transpose(X_train_a), X_train_a)), np.dot(np.transpose(X_train_a), y_train))
    train_error = math.sqrt(np.mean((X_train_a.dot(w) - [y_train]) ** 2))
    test_error = math.sqrt(np.mean((X_test_a.dot(w) - [y_test]) ** 2))
    RMSE.append(train_error)
    RMSEt.append(test_error)
print('train', RMSE)
print('test', RMSEt)
x1 = np.linspace(0.2,1,5)
plt.subplot(2,1,2)
plt.plot(x1, RMSE, label = "train")
plt.plot(x1, RMSEt, label = "test")
plt.xlabel("Proportion")
plt.ylabel("RMSE")
plt.title("RMSE for different training sizes")
plt.legend()
plt.show()
