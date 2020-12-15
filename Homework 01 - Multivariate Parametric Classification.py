from math import log

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv, det

# initializing class parameters
class_means = np.array([[0, 2.5], [-2.5, -2], [2.5, -2]])
class_deviations = np.array([[[3.2, 0], [0, 1.2]], [[1.2, -0.8], [-0.8, 1.2]], [[1.2, 0.8], [0.8, 1.2]]])
class_sizes = [120, 90, 90]

# generating random data points by given class parameters
points1 = np.random.multivariate_normal(class_means[0], class_deviations[0], class_sizes[0])
x1 = points1[:, 0]
y1 = points1[:, 1]

points2 = np.random.multivariate_normal(class_means[1], class_deviations[1], class_sizes[1])
x2 = points2[:, 0]
y2 = points2[:, 1]

points3 = np.random.multivariate_normal(class_means[2], class_deviations[2], class_sizes[2])
x3 = points3[:, 0]
y3 = points3[:, 1]

# plotting randomly generated data points
# plt.plot(x1, y1, "r.")
# plt.plot(x2, y2, "g.")
# plt.plot(x3, y3, "b.")
# plt.show()

# calculating and printing sample means (8 digits after decimal)
print("\n\t\t[,1]\t\t[,2]\t\t[,3]")
print("[1,]", "{:.8f}".format(np.mean(x1)), "" + "{:.8f}".format(np.mean(x2)), "" + "{:.8f}".format(np.mean(x3)))
print("[2,]", "{:.8f}".format(np.mean(y1)), "" + "{:.8f}".format(np.mean(y2)), "" + "{:.8f}".format(np.mean(y3)))

# calculating sample covariances
cov1 = np.cov(x1, y1)
cov2 = np.cov(x2, y2)
cov3 = np.cov(x3, y3)

# printing sample covariances (8 digits after decimal)
print("\n, , 1")
print("\n\t\t[,1]\t\t[,2]")
print("[1,]", "{:.8f}".format(cov1[0][0]), "" + "{:.8f}".format(cov1[0][1]))
print("[1,]", "{:.8f}".format(cov1[1][0]), "" + "{:.8f}".format(cov1[1][1]))

print("\n, , 2")
print("\n\t\t[,1]\t\t[,2]")
print("[1,]", "{:.8f}".format(cov2[0][0]), "" + "{:.8f}".format(cov2[0][1]))
print("[1,]", "{:.8f}".format(cov2[1][0]), "" + "{:.8f}".format(cov2[1][1]))

print("\n, , 3")
print("\n\t\t[,1]\t\t[,2]")
print("[1,]", "{:.8f}".format(cov3[0][0]), "" + "{:.8f}".format(cov3[0][1]))
print("[1,]", "{:.8f}".format(cov3[1][0]), "" + "{:.8f}".format(cov3[1][1]))

# calculating and printing class priors
total_size = np.sum(class_sizes)
prior1 = class_sizes[0] / total_size
prior2 = class_sizes[1] / total_size
prior3 = class_sizes[2] / total_size
print("\nClass priors")
print(prior1, prior2, prior3)

X = ['1', '2', '3']

# organizing sample means, sample covariances, priors and points
sample_means = [[np.mean(x1), np.mean(x2), np.mean(x3)], [np.mean(y1), np.mean(y2), np.mean(y3)]]
sample_covariances = [cov1, cov2, cov3]
priors = [prior1, prior2, prior3]
points = [x1, y1, x2, y2, x3, y3]

# calculating means
means = np.matrix(sample_means).T

# calculating ws, Ws and w0s for each class in order to use while calculating confusion matrix
W1 = (-0.5) * inv(np.matrix(cov1))
w1 = inv(np.matrix(cov1)) * means[0].T
w01 = (-0.5) * means[0] * inv(np.matrix(cov1)) * means[0].T \
      - 0.5 * log(det(np.matrix(cov1))) + log(prior1)

W2 = (-0.5) * inv(np.matrix(cov2))
w2 = inv(np.matrix(cov2)) * means[1].T
w02 = (-0.5) * means[1] * inv(np.matrix(cov2)) * means[1].T \
      - 0.5 * log(det(np.matrix(cov2))) + log(prior2)

W3 = (-0.5) * inv(np.matrix(cov3))
w3 = inv(np.matrix(cov3)) * means[2].T
w03 = (-0.5) * means[2] * inv(np.matrix(cov3)) * means[2].T \
      - 0.5 * log(det(np.matrix(cov3))) + log(prior3)

# combining w, W and w0 of each class
W = [W1, W2, W3]
w = [w1, w2, w3]
w0 = [w01, w02, w03]


# quadratic discriminant function
def g(x, i):
    return np.matrix(x).T * W[i] * x + w[i].T * x + w0[i]


# initializing confussion matrix
confusion_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

# calculating confusion matrix and plotting each point on the graph
for i in range(0, 3):
    for k in range(0, class_sizes[i]):
        a = []
        for j in range(0, 3):
            a.append(g([[points[2 * i][k]], [points[2 * i + 1][k]]], j))
        print(a)
        s = np.argmax(a)
        if s != i:
            if i == 0:
                plt.plot([points[2 * i][k]], [points[2 * i + 1][k]], 'ro')
            if i == 1:
                plt.plot([points[2 * i][k]], [points[2 * i + 1][k]], 'go')
            if i == 2:
                plt.plot([points[2 * i][k]], [points[2 * i + 1][k]], 'bo')
        confusion_matrix[s][i] = confusion_matrix[s][i] + 1

# printing confusion matrix
print("\nConfusion matrix")
for i in range(0, 3):
    print(confusion_matrix[i])

xx = np.arange(-6.0, 6.0, 0.05)
yy = np.arange(-6.0, 6.0, 0.05)


def class_function(x):
    if g(x, 0) > g(x, 1) and g(x, 0) > g(x, 2):
        return 0
    if g(x, 1) > g(x, 0) and g(x, 1) > g(x, 2):
        return 1
    if g(x, 2) > g(x, 0) and g(x, 2) > g(x, 1):
        return 2


# plotting the graph by showing curves and false predictions indicated
Z = []
for a in yy:
    for b in xx:
        Z.append(class_function([[a], [b]]))

Z = np.array(Z)
Z = Z.reshape((xx.size, yy.size))
Z = Z.T

X, Y = np.meshgrid(xx, yy)

levels = [-0.5, 0.5, 1.5, 2.5]
cp = plt.contour(X, Y, Z, colors='k', linewidths=0.5)
cp = plt.contourf(X, Y, Z, levels=levels, colors=['#ff9898', '#98ffa1', '#98cfff'])

plt.show()