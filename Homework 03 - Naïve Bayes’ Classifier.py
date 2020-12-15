import numpy as np
import matplotlib.pyplot as plt

# getting data from csv files
X = np.genfromtxt('hw03_data_set_images.csv', delimiter=',')
Y = np.genfromtxt('hw03_data_set_labels.csv', usecols=0, dtype=str)

# dividing the data set into two parts by assigning the first 25 images from each class to the training set and the
# remaining 14 images to the test set
X_train = (X[0:25], X[39:64], X[78:103], X[117:142], X[156:181])
X_test = (X[25:39], X[64:78], X[103:117], X[142:156], X[181:195])

"""I didn't divide class labels because I know class labels from index of train and test data set i.e. if I want to 
predict X_train[0], I know that its class label is "A" or if I predict X_test[4], I know that its class label is "E" 
etc. """

# number of features, classes and data sets for each class
d = len(X[0])
N = 5
num_train = 25
num_test = 14

# initializing parameters to empty lists to be filled during estimation process
p = [[], [], [], [], []]

# estimating parameters of each training sets by getting means of each feature from 25 data
for i in range(N):
    train = X_train[i]
    for j in range(d):
        p[i].append(np.mean(train[:, j]))
    print("\np" + str(i + 1), " -> ", p[i])

# plotting estimation parameters
for param in p:
    param = np.array(param).reshape(16, 20)
    plt.imshow(1 - np.rot90(np.flip(param, 1), 1), cmap='gray')
    plt.show()


# function for predicting given data based on 5 estimation parameters, returns maximum likelihood of that data
def predict(data):
    m = [1] * N
    for feature in range(d):
        for ind in range(N):
            if data[feature] == 1:
                m[ind] = m[ind] * p[ind][feature]
            else:
                m[ind] = m[ind] * (1 - (p[ind][feature]))
    return np.argmax(np.array(m))


# function for printing confusion matrix for given data set as a parameter
def print_confusion_matrix(data_set):
    conf_matrix = np.array([0] * N * N).reshape(5, 5)
    for ind in range(len(data_set)):
        for train_data in data_set[ind]:
            prediction = predict(train_data)
            conf_matrix[prediction, ind] += 1
    print(conf_matrix)


print("\nConfusion Matrix for Training Data Set")
print_confusion_matrix(X_train)
print("\nConfusion Matrix for Test Data Set")
print_confusion_matrix(X_test)
