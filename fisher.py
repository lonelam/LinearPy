import numpy as np
import matplotlib.pyplot as plt


def DataCollect():
    femaleDocPath = "data/girl.txt"
    maleDocPath = "data/boy.txt"
    femaleFile = open(femaleDocPath, "r")
    maleFile = open(maleDocPath, "r")
    data = [np.mat(np.loadtxt(femaleFile)[:,:2]), np.mat(np.loadtxt(maleFile)[:,:2])]
    return data

def DataGen():
    data = DataCollect()
    mix = []
    mix.extend(data[0])
    mix.extend(data[1])
    for i in range(3):
        #print(mix)
        a = min(mix, key=lambda p: p[i])[i]
        b = max(mix, key=lambda p: p[i])[i]
        #print(a, b)
        for gender in data:
            for person in gender:
                #print(person[i])
                person[i] -= a
                person[i] /= (b-a)
    return data

#Bayes
def trainHW():
    hwData = DataCollect()
#    hwData = [[[x[0] for x in gender], [x[1] for x in gender]] for gender in rawData]
    P0 = (hwData[0].shape[0]) / ((hwData[0].shape[0]) + (hwData[1].shape[0]))

    theta = [
        [
            np.array([np.average(gender[:,0]),np.average(gender[:,1])]) ,
            np.cov(gender[:, 0].T, gender[:, 1].T)
        ]
        for gender in hwData
    ]
    return theta, P0

def GaussN(X, Mu, Sigma):
    return np.exp(np.dot((X - Mu) ,np.linalg.pinv(Sigma) ).dot((X - Mu).transpose()) * (-0.5))

def classifyN(X, Theta, P0, t):
    #print(GaussN(X, Theta[0][0], Theta[0][1]), GaussN(X, Theta[1][0], Theta[1][1]))
    if (GaussN(X, Theta[0][0], Theta[0][1]) * P0 / (GaussN(X, Theta[1][0], Theta[1][1])) * (1-P0) ) > t:
        return 0
    return 1

#类均值向量
def m(X):
    ret = []
    for xi in X:
        sum = np.zeros(xi[0].shape)
        for xj in xi:
            sum += xj
        sum /= len(xi)
        ret.append(sum)
    return ret

#类内离散度矩阵
def S(X, m):
    ret = []
    for i in range(len(X)):
        # print(X[i][0])
        sum = np.zeros(((X[i][0].shape[1]), (X[i][0].shape[1])))
        for xj in X[i]:
            sum += (xj - m[i]).T.dot((xj - m[i]))

        ret.append(np.mat(sum))
    return ret
#总类内离散度矩阵
def S_w(S, X):
    P0 = (X[0].shape[0]) / ((X[0].shape[0]) + (X[1].shape[0]))
    return (S[0] * P0 + S[1] * (1-P0))



#错误率估计
def err_rate():
    rX = DataCollect()
    err = 0
    tot = rX[0].shape[0] + rX[1].shape[0]
    for i in range(rX[0].shape[0]):
        tar = rX[0][i].T
        X = [np.row_stack([rX[0][0:i], rX[0][i+1:]]), rX[1]]

        mean = m(X)
        # print(mean)

        cov = S(X, mean)
        sw = S_w(cov, X)
        # 投影面
        w_star = np.linalg.pinv(sw).dot((mean[0] - mean[1]).T)

        w_0 = (mean[0] + mean[1]).dot(np.linalg.pinv(sw)).dot((mean[0] - mean[1]).T) / 2.0
        if tar[0] * w_star[0] + tar[1] * w_star[1] <= w_0:
            err += 1
    for i in range(rX[1].shape[0]):
        tar = rX[1][i].T
        X = [rX[0], np.row_stack([rX[1][:i], rX[1][i+1:]])]

        mean = m(X)
        # print(mean)

        cov = S(X, mean)
        sw = S_w(cov, X)
        # 投影面
        w_star = np.linalg.pinv(sw).dot((mean[0] - mean[1]).T)

        w_0 = (mean[0] + mean[1]).dot(np.linalg.pinv(sw)).dot((mean[0] - mean[1]).T) / 2.0
        if tar[0] * w_star[0] + tar[1] * w_star[1] > w_0:
            err += 1
    print("留一法错误率：", err / tot)


if __name__ == '__main__':
    X = DataCollect()
    mean = m(X)
    # print(mean)

    cov = S(X, mean)
    sw = S_w(cov, X)
    #投影面
    w_star = np.linalg.pinv(sw).dot((mean[0] - mean[1]).T)

    # print(w_star)

    w_0 = (mean[0] + mean[1]).dot(np.linalg.pinv(sw)).dot((mean[0] - mean[1]).T) / 2.0
    # rxs = []
    # rys = []
    # bxs = []
    # bys = []
    # for x in np.arange(145, 190, 0.1):
    #     for y in np.arange(35, 90, 0.1):
    #         if x * w_star[0] + y * w_star[1] > w_0:
    #             rxs.append(x)
    #             rys.append(y)
    #         else:
    #             bxs.append(x)
    #             bys.append(y)
    xs = []
    ys = []
    ax = plt.subplot(121)
    for y in np.arange(35, 90, 1):
        x = (w_0 - y * w_star[1]) / w_star[0]
        xs.append(float(x))
        ys.append(y)
    ax.scatter(np.array(X[0][:,0]), np.array(X[0][:,1]), color="red")
    ax.scatter(np.array(X[1][:, 0]), np.array(X[1][:, 1]), color="blue")
    # plt.scatter(rxs, rys, color="red")
    # plt.scatter(bxs, bys, color="blue")
    aline = ax.plot(xs, ys, label="fisher split")
    ax.legend(aline)
    bx = plt.subplot(122)
    theta, P0 = trainHW()
    rxs = []
    rys = []
    bxs = []
    bys = []
    for x in np.arange(145, 190, 1):
        for y in np.arange(35, 90, 0.1):
            if classifyN([x, y], theta, P0, 1) == 0:
                rxs.append(x)
                rys.append(y)
            else:
                bxs.append(x)
                bys.append(y)
    bx.scatter(rxs, rys, color="red")
    bx.scatter(bxs, bys, color="blue")
    plt.show()
    # print(w_0)
    err_rate()

