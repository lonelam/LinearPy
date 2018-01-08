import numpy as np


def normal_equation(X, y):
    x0 = np.ones((X.shape[0],1))
    X = np.concatenate((x0, X), axis=1)
    #print(X)
    theta = np.linalg.pinv(X.transpose().dot(X)).dot(X.transpose()).dot(y)
    #print(theta)
    return theta


def gradientDescend(X, y, step= 0.1, initial_theta = None):
    x0 = np.ones((X.shape[0], 1))
    X = np.concatenate((x0, X), axis=1)
    if initial_theta == None:
        theta = np.ones((X.shape[1],))
    else:
        theta = initial_theta
    for i in range(100000):
        g = X.transpose().dot(X).dot(theta) - X.transpose().dot(y)
        #print(g)
        theta -= g * step
    return theta


def linearPredict():
    X = np.loadtxt("NormalEData/X.txt")
    y = np.loadtxt("NormalEData/y.txt")
    print(normal_equation(X, y))
    print(gradientDescend(X, y))

def xorPredict():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])
    print(normal_equation(X, y))
    print(gradientDescend(X, y))
def andPredict():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])
    print(normal_equation(X, y))
    print(gradientDescend(X, y))

if __name__ == '__main__':
    andPredict()