import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as spa
from scipy.stats import multivariate_normal
from math import e, pow

# initializing class parameters
class_means = np.array([[2.5, 2.5], [-2.5, 2.5], [-2.5, -2.5], [2.5, -2.5], [0, 0]])
class_deviations = np.array(
    [[[0.8, -0.6], [-0.6, 0.8]],
     [[0.8, 0.6], [0.6, 0.8]],
     [[0.8, -0.6], [-0.6, 0.8]],
     [[0.8, 0.6], [0.6, 0.8]],
     [[1.6, 0], [0, 1.6]]])
class_sizes = [50, 50, 50, 50, 100]

N = sum(class_sizes)
K = len(class_sizes)

# generating random data points by given class parameters
points1 = np.random.multivariate_normal(class_means[0], class_deviations[0], class_sizes[0])
points2 = np.random.multivariate_normal(class_means[1], class_deviations[1], class_sizes[1])
points3 = np.random.multivariate_normal(class_means[2], class_deviations[2], class_sizes[2])
points4 = np.random.multivariate_normal(class_means[3], class_deviations[3], class_sizes[3])
points5 = np.random.multivariate_normal(class_means[4], class_deviations[4], class_sizes[4])
X = np.vstack((points1, points2, points3, points4, points5))

# plotting randomly generated data points by given parameters
plt.plot(X[:, 0], X[:, 1], "k.")
plt.xlim(-6, 6)
plt.ylim(-6, 6)
plt.show()

# Three useful functions which are taken from the lab session
def update_centroids(memberships, X):
    if memberships is None:
        # initialize centroids
        centroids = X[np.random.choice(range(N), K), :]
    else:
        # update centroids
        centroids = np.vstack([np.mean(X[memberships == k,], axis=0) for k in range(K)])
    return centroids


def update_memberships(centroids, X):
    # calculate distances between centroids and data points
    D = spa.distance_matrix(centroids, X)
    # find the nearest centroid for each data point
    memberships = np.argmin(D, axis=0)
    return memberships


def plot_current_state(centroids, memberships, X):
    cluster_colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#b15928",
                               "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6", "#ffff99"])
    if memberships is None:
        plt.plot(X[:, 0], X[:, 1], ".", markersize=10, color="black")
    else:
        for c in range(K):
            plt.plot(X[memberships == c, 0], X[memberships == c, 1], ".", markersize=10,
                     color=cluster_colors[c])
    plt.xlabel("x1")
    plt.ylabel("x2")


# Two iterations for initializing covariances and prior probabilities
centroids = None
memberships = None
for _ in range(2):
    centroids = update_centroids(memberships, X)
    memberships = update_memberships(centroids, X)

points = [[], [], [], [], []]
for i in range(len(memberships)):
    points[memberships[i]].append(X[i])

initial_priors = []
initial_covariances = []
for group in points:
    initial_covariances.append(np.cov(np.asmatrix(group).T))
    initial_priors.append(len(group) / len(memberships))

# initializing variables for EM algorithm
means = centroids
h = np.zeros((300, 5))
priors = initial_priors
covariances = initial_covariances
XX = np.asmatrix(X)

# EM algorithm
for i in range(0, 100):
    print("Iteration#{}".format(i))
    for k in range(0, 300):
        tempSum = 0
        for j in range(0, 5):
            xx = np.matrix([X[k] - means[j]])
            mat = xx.dot(np.linalg.inv(covariances[j])).dot(xx.T)
            mat = mat * (-.5)
            h[k][j] = priors[j] * pow(np.linalg.det(covariances[j]), -0.5) * pow(e, mat[0])
            tempSum += h[k][j]
        h[k] /= tempSum

    means = h.T.dot(XX)
    tempHsum = np.sum(h, axis=0)

    means = means / tempHsum[:, None]
    means = np.asarray(means)

    covariances = []

    for j in range(0, 5):
        tempSum = 0
        for k in range(0, 300):
            xx = np.matrix([X[k] - means[j]])
            mat = xx.T.dot(xx)
            res = mat * h[k][j]
            tempSum += res
        tempSum /= tempHsum[j]
        covariances.append(tempSum)

# Mean vectors of EM algorithm
print(means)

# Drawing the clustering result obtained by EM algorithm by coloring each cluster with a different color
memberships = update_memberships(means, X)
plot_current_state(means, memberships, X)

# Drawing the original Gaussian densities and the Gaussian densities that EM algorithm finds with dashed and solid
# lines, respectively.
aa, bb = np.mgrid[-6:6:.05, -6:6:.05]
pos = np.empty(aa.shape + (2,))
pos[:, :, 0] = aa;
pos[:, :, 1] = bb

colors = ['c', 'm', 'y', 'k']
for i in range(0, 5):
    rv = multivariate_normal(class_means[i], class_deviations[i])
    plt.contour(aa, bb, rv.pdf(pos), linestyles='dashed', levels=[0.05], colors='k')
    rv = multivariate_normal(means[i], covariances[i])
    plt.contour(aa, bb, rv.pdf(pos), colors='k', levels=[0.05])


plt.show()
