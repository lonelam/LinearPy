import numpy as np
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from kNearCourse import normalize, DataCollect, TestCollect

def fish(X):
    from fisher import m, S, S_w, err_rate
    mean = m(X)
    cov = S(X, mean)
    sw = S_w(cov, X)
    w_star = np.dot(np.linalg.pinv(sw), (mean[0] - mean[1]).T)
    w_0 = (mean[0] + mean[1]).dot(np.linalg.pinv(sw)).dot((mean[0] - mean[1]).T) / 2.0
    xs = []
    ys = []
    ax = plt.subplot(111)
    for y in np.arange(-0.01, 0.01, 0.0001):
        x = (w_0 - y * w_star[1]) / w_star[0]
        xs.append(float(x))
        ys.append(y)
    ax.scatter(np.array(X[0][:, 0]), np.array(X[0][:, 1]), color="red")
    ax.scatter(np.array(X[1][:, 0]), np.array(X[1][:, 1]), color="blue")
    aline = ax.plot(xs, ys, label="fisher split")
    ax.legend()
    plt.show()
    err_rate(X)

if __name__ == '__main__':
    X = DataCollect()
    X = normalize(X)
    mess = np.row_stack(X).T
    mess_cov = np.cov(mess)
    eig_val, eig_vec = np.linalg.eig(mess_cov)
    eig_pairs = [(np.abs(eig_val[i]), eig_vec[:, i]) for i in range(len(eig_val))]
    eig_pairs.sort(key=lambda p: p[0], reverse=True)
    #计算变换矩阵
    feature = []
    for i in range(2):
        feature.append(eig_pairs[i][0] * eig_pairs[i][1])
    feature = np.array(feature)
    for i in range(2):
        X[i] = np.dot(X[i], feature.T)
    fish(X)
