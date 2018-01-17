import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def h(x, theta):
    return 1.0 / (1.0 + np.exp(-x.dot(theta)))

def cost(y_predict, y):
    return -y * np.log(y_predict) - (1-y) * np.log(1-y_predict)

def normal_equation(X, y):
    x0 = np.ones((X.shape[0],1))
    X = np.concatenate((x0, X), axis=1)
    #print(X)
    theta = np.linalg.pinv(X.transpose().dot(X)).dot(X.transpose()).dot(y)
    #print(theta)
    return theta


def gradientDescend(X, y, step= 0.01, initial_theta = None):
    x0 = np.ones((X.shape[0], 1))
    X = np.concatenate((x0, X), axis=1)
    if initial_theta == None:
        theta = np.ones((X.shape[1],))
    else:
        theta = initial_theta
    for i in range(100000):
        g = X.transpose().dot(h(X, theta) - y)
        theta -= g /len(y) * step
    return theta


def linearPredict():
    X = np.loadtxt("NormalEData/X.txt")
    y = np.loadtxt("NormalEData/y.txt")
    print(normal_equation(X, y))
    print(gradientDescend(X, y, 0.00001))
netlen = 10
def orPredict():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 1])
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    theta = normal_equation(X, y)
    print(theta)
    xs = np.linspace(0, 1, netlen)
    ys = np.linspace(0, 1, netlen)
    xs, ys = np.meshgrid(xs, ys)
    zs = np.zeros((netlen, netlen))
    theta = gradientDescend(X, y, 1)
    print(theta)
    for i in range(netlen):
        for j in range(netlen):
            zs[i][j] = h(np.array([1, xs[i][j], ys[i][j]]), theta)
    for x in X:
        print(1.0 / (1.0 + np.exp(-np.array([1, x[0], x[1]]).dot(theta))))
    ax.plot_surface(xs, ys, zs, cmap="rainbow")
    plt.show()

def xorPredict():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    theta = normal_equation(X, y)
    print(theta)
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
            zs[i][j] = 1.0 / (1.0 + np.exp(-np.array([1, xs[i][j], ys[i][j]]).dot(theta)))
    ax.plot_surface(xs, ys, zs, cmap="rainbow")
    plt.show()

def andPredict():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])
    print(normal_equation(X, y))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    theta = gradientDescend(X, y, 1, [-1, 1, 1])
    print(theta)
    xs = np.linspace(0, 1, netlen)
    ys = np.linspace(0, 1, netlen)
    xs, ys = np.meshgrid(xs, ys)
    zs = np.zeros((netlen, netlen))
    for i in range(netlen):
        for j in range(netlen):
            zs[i][j] = 1.0 / (1.0 + np.exp(-np.array([1, xs[i][j], ys[i][j]]).dot(theta)))
    for x in X:
        print(1.0 / (1.0 + np.exp(-np.array([1, x[0], x[1]]).dot(theta))))

    ax.plot_surface(xs, ys, zs, cmap="rainbow")
    plt.show()

if __name__ == '__main__':
    orPredict()
