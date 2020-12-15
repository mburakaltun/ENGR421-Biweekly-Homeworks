import matplotlib.pyplot as plt
from math import sqrt, pow
import pandas as pd
import numpy as np
from queue import Queue

# extracting data from the csv file
df = pd.read_csv('hw05_data_set.csv')
df = df.reindex(np.random.permutation(df.index))

# dividing the data set into two parts by assigning the first 100 data points to the training set and the remaining
# 33 data points to the test set.
train_data = df[0:100]
test_data = df[100:133]

train_values = []
for index, row in train_data.iterrows():
    train_values.append([row['x'], row['y']])

test_values = []
for index, row in test_data.iterrows():
    test_values.append([row['x'], row['y']])

# setting constants
N_train = len(train_data)
N_test = len(test_data)


# Decision Tree Regression Algorithm
def DecisionTreeRegression(P, data, indices, queue):
    if len(data) > P:
        sum = 0
        for i in data:
            sum += i[0]
        index = sum * 1.0 / len(data)
        indices.append(index)

        S1 = []
        S2 = []

        for i in data:
            if i[0] <= index:
                S1.append([i[0], i[1]])
            else:
                S2.append([i[0], i[1]])
        if len(S2) == 0:
            S1 = np.array_split(S1, 2)[0]
            S2 = np.array_split(S1, 2)[1]
        if len(S1) > P:
            queue.put(S1)
        if len(S2) > P:
            queue.put(S2)


P = 15
globalQueue = Queue()
globalQueue.put(train_values)
globalIndices = []
Ps = [P]
while not globalQueue.empty():
    popped = globalQueue.get()
    DecisionTreeRegression(P, popped, globalIndices, globalQueue)
globalIndices.append(0)
globals = sorted(globalIndices)
dic = {}

for i in range(0, len(globals)):
    counter = 0
    sum = 0
    if i == len(globals) - 1:
        for index, data in train_data.iterrows():
            if 60 > data['x'] >= globals[i]:
                sum += data['y']
                counter += 1
    else:
        for index, data in train_data.iterrows():
            if globals[i + 1] > data['x'] >= globals[i]:
                sum += data['y']
                counter += 1
    if counter != 0:
        dic[globals[i]] = sum * 1.0 / counter
    else:
        dic[globals[i]] = dic[globals[i - 1]]

xs = [0]
ys = []
for i in sorted(dic):
    if i != 0:
        xs.append(i)
        xs.append(i)
    ys.append(dic[i])
    ys.append(dic[i])

xs.append(60)
plt.plot(train_data['x'], train_data['y'], 'b.', label='training')
plt.plot(test_data['x'], test_data['y'], 'r.', label='test')
plt.plot(xs, ys, 'k-')
plt.xlabel('x')
plt.ylabel('y')
plt.title('P = 15')
plt.show()

RMSE_sum = 0
for indexer, row in test_data.iterrows():
    for i in range(0, len(globals)):
        averageY = dic[globals[i]]
        if i == len(globals) - 1:
            if 60 > row['x'] >= globals[i]:
                RMSE_sum += pow((averageY - row['y']), 2)
        else:
            if globals[i + 1] > row['x'] >= globals[i]:
                RMSE_sum += pow((averageY - row['y']), 2)

RMSE = sqrt(RMSE_sum / len(test_values))
print("RMSE is " + str(RMSE) + " when P is 15")
