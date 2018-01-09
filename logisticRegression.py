import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def normal_equation(X, y):
    x0 = np.ones((X.shape[0],1))
    X = np.concatenate((x0, X), axis=1)
    #print(X)
    theta = np.linalg.pinv(X.transpose().dot(X)).dot(X.transpose()).dot(y)
    #print(theta)
    return theta


def gradientDescend(X, y, step= 0.00001, initial_theta = None):
    x0 = np.ones((X.shape[0], 1))
    X = np.concatenate((x0, X), axis=1)
    if initial_theta == None:
        theta = np.ones((X.shape[1],))
    else:
        theta = initial_theta
    for i in range(100000):
        g = X.transpose().dot(1.0 / (1.0 - np.exp(X.dot(theta))) - y)
        #g = X.transpose().dot(X).dot(theta) - X.transpose().dot(y)
        #print(g)
        theta -= g * step
    return theta


def linearPredict():
    X = np.loadtxt("NormalEData/X.txt")
    y = np.loadtxt("NormalEData/y.txt")
    print(normal_equation(X, y))
    print(gradientDescend(X, y, 0.00001))

def xorPredict():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    theta = normal_equation(X, y)
    print(theta)
    netlen = 10
    xs = np.linspace(0, 1, netlen)
    ys = np.linspace(0, 1, netlen)
    xs, ys = np.meshgrid(xs, ys)
    zs = np.zeros((netlen, netlen))
    for i in range(netlen):
        for j in range(netlen):
            zs[i][j] = np.array([1, xs[i][j], ys[i][j]]).dot(theta)
    ax.plot_surface(xs, ys, zs, cmap="rainbow")
    plt.show()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    theta = gradientDescend(X, y)
    print(theta)
    for i in range(netlen):
        for j in range(netlen):
            zs[i][j] = 1.0 / (1.0 - np.exp(np.array([1, xs[i][j], ys[i][j]]).dot(theta)))
    ax.plot_surface(xs, ys, zs, cmap="rainbow")
    plt.show()
def andPredict():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])
    print(normal_equation(X, y))
    print(gradientDescend(X, y))

if __name__ == '__main__':
    xorPredict()
    andPredict()
