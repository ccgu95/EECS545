import numpy as np
from sklearn import datasets
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
mean_X = X_train.mean(axis=0)
std_X = X_train.std(axis=0)
X_train_n = (X_train - mean_X) / std_X
X_test_n = (X_test - mean_X) / std_X

#b
learning_rate = 5e-4
Epoch=500
n = len(X_train)
m = len(X_train[0])
MSE = np.zeros(Epoch)
#initiate w
w = np.random.uniform(-0.1,0.1,size=(1,m+1))
#append 1 to feature vector
X_train_a = np.ones((n, 1))
X_train_a = np.concatenate((X_train_a,X_train_n),axis=1)
for j in range(Epoch):
    Randsh = list(range(n))
    np.random.shuffle(Randsh)
    for k in range(n):
        i = int(Randsh[k])
        w = w - learning_rate * (w.dot(np.transpose(X_train_a[i])) - y_train[i]) * X_train_a[i]
    MSE[j] = np.mean((X_train_a.dot(np.transpose(w)) - np.transpose([y_train])) ** 2)
nt = len(X_test)
X_test_a = np.ones((nt, 1))
X_test_a = np.concatenate((X_test_a,X_test_n),axis=1)
test_error = np.mean((X_test_a.dot(np.transpose(w)) - np.transpose([y_test])) ** 2)
print('The learnt weight vector is ', w[0])
print('The bias term is ', w[0][0])
print('The train error is ', MSE[-1])
print('The test error is ', test_error)
plt.subplot(211)
plt.plot(MSE)
plt.xlabel("epochs")
plt.ylabel("MSE")

#c
learning_rate = 5e-4
Epoch=500
MSE2 = np.zeros(Epoch)
#initiate w
w2 = np.random.uniform(-0.1,0.1,size=(1,m+1))
#append 1 to feature vector
for j in range(Epoch):
    grad = sum((w2.dot(np.transpose(X_train_a[i])) - y_train[i]) * X_train_a[i] for i in range(n))
    w2 = w2 - learning_rate * grad
    MSE2[j] = np.mean((X_train_a.dot(np.transpose(w2)) - np.transpose([y_train])) ** 2)
test_error2 = np.mean((X_test_a.dot(np.transpose(w2)) - np.transpose([y_test])) ** 2)
print('The learnt weight vector is ', w2[0])
print('The bias term is ', w2[0][0])
print('The train error is ', MSE2[-1])
print('The test error is ', test_error2)
plt.subplot(212)
plt.plot(MSE2)
plt.xlabel("epochs")
plt.ylabel("MSE")
plt.show()

#d
w_c = np.dot(np.linalg.inv(np.dot(np.transpose(X_train_a),X_train_a)),np.dot(np.transpose(X_train_a),y_train))
train_error_c = np.mean((X_train_a.dot(w_c) - [y_train]) ** 2)
test_error_c = np.mean((X_test_a.dot(w_c) - [y_test]) ** 2)
print('The weight vector is ', w_c)
print('The bias term is ', w_c[0])
print('The train error is ', train_error_c)
print('The test error is ', test_error_c)

#e
# Original features
features_orig = dataset.data
labels_orig = dataset.target
Ndata = len(features_orig)

train_errs = []
test_errs = []

for k in range(100):
  # Shuffle data
  rand_perm = np.random.permutation(Ndata)
  features = [features_orig[ind] for ind in rand_perm]
  labels = [labels_orig[ind] for ind in rand_perm]
  # Train/test split
  Nsplit = 50
  X_train, y_train = features[:-Nsplit], labels[:-Nsplit]
  X_test, y_test = features[-Nsplit:], labels[-Nsplit:]
  # Preprocess your data - Normalization, adding a constant feature
  mean_X = np.mean(X_train)
  std_X = np.std(X_train)
  X_train_n = (X_train - mean_X) / std_X
  X_test_n = (X_test - mean_X) / std_X
  X_train_a = np.ones((n, 1))
  X_train_a = np.concatenate((X_train_a, X_train_n), axis=1)
  X_test_a = np.ones((nt, 1))
  X_test_a = np.concatenate((X_test_a, X_test_n), axis=1)
  # Solve for optimal w
  # Use your solver function
  w = np.dot(np.linalg.inv(np.dot(np.transpose(X_train_a), X_train_a)), np.dot(np.transpose(X_train_a), y_train))
  # Collect train and test errors
  train_error = np.mean((X_train_a.dot(w) - [y_train]) ** 2)
  test_error = np.mean((X_test_a.dot(w) - [y_test]) ** 2)
  # Use your implementation of the mse function
  train_errs.append(train_error)
  test_errs.append(test_error)

print('Mean training error: ', np.mean(train_errs))
print('Mean test error: ', np.mean(test_errs))