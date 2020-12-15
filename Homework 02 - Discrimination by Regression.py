import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# getting data from csv files
X = np.genfromtxt('hw02_data_set_images.csv', delimiter=',')
Y = np.genfromtxt('hw02_data_set_labels.csv', usecols=0, dtype=str)

X_train = np.concatenate((X[0:25], X[39:64], X[78:103], X[117:142], X[156:181]))
X_test = np.concatenate((X[25:39], X[64:78], X[103:117], X[142:156], X[181:195]))

Y_tr = np.concatenate((Y[0:25], Y[39:64], Y[78:103], Y[117:142], Y[156:181]))
Y_tst = np.concatenate((Y[25:39], Y[64:78], Y[103:117], Y[142:156], Y[181:195]))

y_truth = []
y_test_truth = []

# converting class labels from string to integer
for i in Y_tr:
    if i == '\"A\"':
        y_truth.append(1)
    elif i == '\"B\"':
        y_truth.append(2)
    elif i == '\"C\"':
        y_truth.append(3)
    elif i == '\"D\"':
        y_truth.append(4)
    elif i == '\"E\"':
        y_truth.append(5)

for i in Y_tst:
    if i == '\"A\"':
        y_test_truth.append(1)
    elif i == '\"B\"':
        y_test_truth.append(2)
    elif i == '\"C\"':
        y_test_truth.append(3)
    elif i == '\"D\"':
        y_test_truth.append(4)
    elif i == '\"E\"':
        y_test_truth.append(5)

K = 5
N = 125

y_truth = np.array(y_truth)
y_test_truth = np.array(y_test_truth)

Y_truth = np.zeros((N, K)).astype(int)
Y_truth[range(N), y_truth - 1] = 1
Y_test_truth = np.zeros((N, K)).astype(int)
Y_test_truth[range(N), y_truth - 1] = 1


# defining safelog, softmax and gradient functions
def safelog(x):
    return np.log(x + 1e-100)


def gradient_W(X, y_truth, y_predicted):
    return (np.asarray(
        [-np.sum(np.repeat((y_truth[:, c] - y_predicted[:, c])[:, None], X.shape[1], axis=1) * X, axis=0) for c in
         range(K)]).transpose())


def gradient_w0(Y_truth, Y_predicted):
    return -np.sum(Y_truth - Y_predicted, axis=0)


# define the sigmoid function
def sigmoid(X_, w_, w0_):
    return 1 / (1 + np.exp(-(np.matmul(X_, w_) + w0_)))

# setting learning parameters
eta = 0.01
epsilon = 1e-3

# randomly initalizing W and w0
np.random.seed(421)
W = np.random.uniform(low=0, high=1, size=(X.shape[1], K))
w0 = np.random.uniform(low=0, high=1, size=(1, K))
print(W)
# learning W and w0 using gradient descent
iteration = 1
objective_values = []
while 1:
    Y_predicted = sigmoid(X_train, W, w0)

    objective_values = np.append(objective_values, -np.sum(Y_truth * safelog(Y_predicted)))

    W_old = W
    w0_old = w0

    W = W - eta * gradient_W(X_train, Y_truth, Y_predicted)
    w0 = w0 - eta * gradient_w0(Y_truth, Y_predicted)

    if np.sqrt(np.sum((w0 - w0_old)) ** 2 + np.sum((W - W_old) ** 2)) < epsilon:
        break

    iteration = iteration + 1


# plot objective function during iterations
plt.figure(figsize=(10, 6))
plt.plot(range(1, iteration + 1), objective_values, "k-")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.show()

y_predicted = np.argmax(Y_predicted, axis=1) + 1
confusion_matrix = pd.crosstab(y_predicted, y_truth, rownames=['y_pred'], colnames=['y_truth'])
print(confusion_matrix)