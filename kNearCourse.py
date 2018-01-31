import numpy as np
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def normalize(data):
    minmax = [(min(np.min(data[0].T[i]) ,np.min(data[1].T[i])), max(np.max(data[0].T[i]), np.max(data[1].T[i]))) for i in range(data[0].shape[1])]
    for gender in data:
        for i in range(gender.shape[0]):
            for j in range(gender.shape[1]):
                gender[i, j] =(gender[i, j] - minmax[j][0]) / ( minmax[j][1] - minmax[j][0])
    return data

def DataCollect():
    femaleDocPath = "data/girl.txt"
    maleDocPath = "data/boy.txt"
    femaleFile = open(femaleDocPath, "r")
    maleFile = open(maleDocPath, "r")
    data = [np.mat(np.loadtxt(femaleFile)), np.mat(np.loadtxt(maleFile))]
    femaleFile.close()
    maleFile.close()
    return data


def TestCollect():
    femaleDocPath = "data/female.txt"
    maleDocPath = "data/male.txt"
    femaleFile = open(femaleDocPath, "r")
    maleFile = open(maleDocPath, "r")
    data = [np.mat(np.loadtxt(femaleFile)), np.mat(np.loadtxt(maleFile))]
    femaleFile.close()
    maleFile.close()
    return data
#对X进行剪辑
def split(X):
    X[0]
    Stree = KDTree(np.row_stack([X[0], X[1]]))

if __name__ == '__main__':
    for kkk in [1,3,5]:
        #Train
        X = DataCollect()
        X = normalize(X)
        bx = plt.subplot(122)

        h = 0.01
        xx, yy = np.meshgrid(np.arange(0, 1, h),
                             np.arange(0, 1, h))

        tree2 = KDTree(np.row_stack([X[0][:, 1:], X[1][:, 1:]]))
        Z = np.zeros(xx.shape)
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                dist, ind = tree2.query(np.array([xx[i,j], yy[i,j]]).reshape(1, -1), k=kkk)
                cnt = [0, 0]
                for k in ind.T:
                    if k >= X[0].shape[0]:
                        cnt[1] += 1
                    else:
                        cnt[0] += 1
                if cnt[0] > cnt[1]:
                    tmp = 0
                else:
                    tmp = 1
                Z[i,j] = tmp

        plt.pcolormesh(xx, yy, Z)
        bx.scatter(np.array(X[0][:,1]), np.array(X[0][:,2].T),color="red", label="female")
        bx.scatter(np.array(X[1][:, 1]), np.array(X[1][:, 2].T), color="blue", label="male")

        ax = plt.subplot(121, projection="3d")

        ax.scatter(np.array(X[0][:,0]), np.array(X[0][:,1]), np.array(X[0][:,2].T), color="red", label="female")
        ax.scatter(np.array(X[1][:,0]), np.array(X[1][:, 1]), np.array(X[1][:, 2].T), color="blue", label="male")

        tree = KDTree(np.row_stack([X[0], X[1]]))
        #Test
        x = TestCollect()
        x = normalize(x)
        tot = x[0].shape[0] + x[1].shape[0]
        errcnt = 0
        errpointx = []
        errpointy = []
        errpointz=  []
        for i in range(2):
            for xj in x[i]:
                dist, ind = tree.query(xj, k=kkk)
                cnt = [0, 0]
                for k in ind.T:
                    if k >= X[0].shape[0]:
                        cnt[1] += 1
                    else:
                        cnt[0] += 1
                if cnt[0] > cnt[1]:
                    tmp = 0
                else:
                    tmp = 1
                if tmp != i:
                    errcnt+= 1
                    errpointx.append(xj.T[0])
                    errpointy.append(xj.T[1])
                    errpointz.append(xj.T[2])
        ax.scatter(np.array(errpointx), np.array(errpointy), np.array(errpointz), color="black", label="Error point")
        plt.legend()
        plt.show()
        print("k=",kkk, "错误率: ",errcnt / tot)

