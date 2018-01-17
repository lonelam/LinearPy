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
        print(X[i][0])
        sum = np.zeros(((X[i][0].shape[1]), (X[i][0].shape[1])))
        for xj in X[i]:
            sum += (xj - m[i]).T.dot((xj - m[i]))

        ret.append(np.mat(sum))
    return ret
#总类内离散度矩阵
def S_w(S):
    return (S[0] * 0.5 + S[1] * 0.5)
if __name__ == '__main__':
    X = DataCollect()
    mean = m(X)
    print(mean)

    cov = S(X, mean)
    sw = S_w(cov)
    #投影面
    w_star = np.linalg.pinv(sw).dot((mean[0] - mean[1]).T)

    print(w_star)

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
    for y in np.arange(35, 90, 1):
        x = (w_0 - y * w_star[1]) / w_star[0]
        xs.append(float(x))
        ys.append(y)
    plt.scatter(np.array(X[0][:,0]), np.array(X[0][:,1]), color="red")
    plt.scatter(np.array(X[1][:, 0]), np.array(X[1][:, 1]), color="blue")
    # plt.scatter(rxs, rys, color="red")
    # plt.scatter(bxs, bys, color="blue")
    plt.plot(xs, ys)
    plt.show()
    print(w_0)