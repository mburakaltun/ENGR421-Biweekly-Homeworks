import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.neighbors.kde import KernelDensity

# extracting data from the csv file
df = pd.read_csv('hw04_data_set.csv', usecols=["x", "y"])

X = np.array(df["x"])
Y = np.array(df["y"])

# dividing the data set into two parts by assigning the first 100 data points to the training set and the remaining
# 33 data points to the test set.
X_train = X[0:100]
X_test = X[100:133]

Y_train = Y[0:100]
Y_test = Y[100:133]

# setting constants
N_train = len(X_train)
N_test = len(X_test)
minimum_value = 0
maximum_value = 60
bin_width = 3
data_interval = np.linspace(minimum_value, maximum_value, 1201)

left_borders = np.arange(minimum_value, maximum_value, bin_width)
right_borders = np.arange(minimum_value + bin_width, maximum_value + bin_width, bin_width)


# REGRESSOGRAM
# function for getting probability of each data in data set based on and comparing the training data set with given
# binwidth
def get_p_hat(data_set):
    p_hat = []
    for i in range(len(left_borders)):
        current_sum = 0
        cnt = 0
        for j in range(len(data_set)):
            if left_borders[i] < data_set[j] <= right_borders[i]:
                current_sum += Y_train[j]
                cnt += 1
        if current_sum == 0:
            p_hat.append(0)
        else:
            p_hat.append(round(current_sum / cnt, 4))
    return p_hat


# using the function above y_hat values are generated based on X_train
y_hat = get_p_hat(X_train)

# drawing training data points, test data points, and regressogram in the same figure with bin width 3
plt.figure(figsize=(8, 4))
plt.xlabel("x")
plt.ylabel("y")
plt.title("h=3")
plt.plot(X_train, Y_train, "b.", markersize=3)
plt.plot(X_test, Y_test, "r.", markersize=3)
for b in range(len(left_borders)):
    plt.plot([left_borders[b], right_borders[b]], [y_hat[b], y_hat[b]], "k-")
for b in range(len(left_borders) - 1):
    plt.plot([right_borders[b], right_borders[b]], [y_hat[b], y_hat[b + 1]], "k-")
plt.show()

# calculating RMSE of regressogram for test data points
rmse_regressogram_sum = 0
for a in range(len(left_borders)):
    for b in range(len(X_test)):
        if left_borders[a] < X_test[b] <= right_borders[a]:
            rmse_regressogram_sum += (Y_test[b] - y_hat[a]) ** 2

rmse_regressogram = math.sqrt(rmse_regressogram_sum / N_test)
print("Regressogram => RMSE is " + str(rmse_regressogram) + " when h is " + str(bin_width))


# RUNNING MEAN SMOOTHER
# function for calculating Running Mean Smoother of a given data set using given bin width based on training set
def rms_dataset(data_set, bw):
    rms = []
    for i in data_set:
        cur_sum = 0
        cnt = 0
        for j in range(len(X_train)):
            if i - bw / 2 < X_train[j] <= i + bw / 2:
                cur_sum += Y_train[j]
                cnt += 1
        if cnt == 0:
            rms.append(0)
        else:
            rms.append(cur_sum / cnt)
    return rms


# Running Mean Smoother for data interval
data_interval_rms = rms_dataset(data_interval, bin_width)

# drawing training data points, test data points, and running mean smoother in the same figure
plt.figure(figsize=(8, 4))
plt.xlabel("x")
plt.ylabel("y")
plt.title("h=3")
plt.plot(X_train, Y_train, "b.", markersize=3)
plt.plot(X_test, Y_test, "r.", markersize=3)
plt.plot(data_interval, data_interval_rms, "k-", markersize=3)
plt.show()

# calculating RMSE of running mean smoother for test data points
rmse_rms_sum = 0
for a in range(len(data_interval_rms) - 1):
    for b in range(len(X_test)):
        if data_interval[a] < X_test[b] <= data_interval[a + 1]:
            rmse_rms_sum += (Y_test[b] - data_interval_rms[a]) ** 2

rmse_rms = math.sqrt(rmse_rms_sum / N_test)
print("Running Mean Smoother => RMSE is " + str(rmse_rms) + " when h is " + str(bin_width))

# KERNEL SMOOTHER
bin_width_kernel = 1
data_interval_kernel = []
X_train_sorted = sorted(X_train)
Y_train_sorted = sorted(Y_train)


def K(x):
    return math.exp(-(x ** 2) / 2) / math.sqrt(2 * math.pi)


# Kernel Smoother for data interval
for i in data_interval:
    cur_sum = 0
    general_sum = 0
    for j in range(len(X_train)):
        cur_sum += (K((i - X_train[j]) / bin_width_kernel) * Y_train[j])
        general_sum += K((i - X_train[j]) / bin_width_kernel)
    data_interval_kernel.append(cur_sum / general_sum)

# drawing training data points, test data points, and running mean smoother in the same figure
plt.figure(figsize=(8, 4))
plt.xlabel("x")
plt.ylabel("y")
plt.title("h=3")
plt.plot(X_train, Y_train, "b.", markersize=3)
plt.plot(X_test, Y_test, "r.", markersize=3)
plt.plot(data_interval, data_interval_kernel, "k-", markersize=3)
plt.show()

# calculating the RMSE of kernel smoother for test data points
rmse_kernel_sum = 0
for a in range(len(data_interval_kernel) - 1):
    for b in range(len(X_test)):
        if data_interval[a] < X_test[b] <= data_interval[a + 1]:
            rmse_kernel_sum += (Y_test[b] - data_interval_kernel[a]) ** 2

rmse_kernel_rms = math.sqrt(rmse_kernel_sum / N_test)
print("Kernel Smoother => RMSE is " + str(rmse_kernel_rms) + " when h is " + str(bin_width_kernel))
