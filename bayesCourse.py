import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def Gauss(x, avg, var):
    return np.exp(-((x-avg) **2) / var / 2.0) / np.sqrt(2.0 * np.pi * var)
def GaussN(X, Mu, Sigma):
    return np.exp(np.dot((X - Mu) ,np.linalg.inv(Sigma) ).dot((X - Mu).transpose()) * (-0.5))
def DataCollect():
    femaleDocPath = "data/girl.txt"
    maleDocPath = "data/boy.txt"
    femaleFile = open(femaleDocPath, "r")
    maleFile = open(maleDocPath, "r")
    data = [np.loadtxt(femaleFile), np.loadtxt(maleFile)]
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

def trainHeight():
    rawData = DataCollect()
    hData = [[x[0] for x in gender ] for gender in rawData]
    #print(hData)
    theta = np.array([(np.average(gender), np.var(gender)) for gender in hData])
    print(theta)
    lF, = plt.plot(range(140, 200), [Gauss(x, theta[0][0], theta[0][1]) for x in range(140, 200)] , label='P(x|Female)')
    lM, = plt.plot(range(140, 200), [Gauss(x, theta[1][0], theta[1][1]) for x in range(140, 200)], label='P(x|Male)')
    plt.legend(handles=[lF,lM])
    plt.show()
    return theta
def classify(x, theta):
    if Gauss(x, theta[0][0], theta[0][1]) > Gauss(x, theta[1][0], theta[1][1]):
        return 0
    return 1
eps = 0.1
def trainHW():
    rawData = DataGen()
    hwData = [[[x[0] for x in gender], [x[1] for x in gender]] for gender in rawData]
    theta = [
        [
            np.array([np.average(gender[0]),np.average(gender[1])]) ,
            np.cov(gender[0], gender[1])
        ]
        for gender in hwData
    ]
    return theta
def drawFace(theta):
    fig = plt.figure(figsize=plt.figaspect(0.5))
    X = np.arange(0, 1, 0.005)
    xlen = len(X)
    Y = np.arange(0, 1, 0.005)
    ylen = len(Y)
    X, Y = np.meshgrid(X, Y)
    Z0 = np.zeros((xlen, ylen))
    Z1 = np.zeros((xlen, ylen))
    for i in range(xlen):
        for j in range(ylen):
            Z0[i][j] = GaussN(np.array([X[i][j],Y[i][j]]), theta[0][0], theta[0][1])
            Z1[i][j] = GaussN(np.array([X[i][j], Y[i][j]]), theta[1][0], theta[1][1])
            # if Z0[i][j] < Z1[i][j]:
            #     Z0[i][j] = 1
            # else:
            #     Z1[i][j] = 0
    ax = fig.add_subplot(1, 2,2,projection='3d')
    bx = fig.add_subplot(1,2,1, projection='3d')
    ax.plot_surface(X, Y, Z0 , cmap='PuBu')
    bx.plot_surface(X, Y, Z1, cmap='RdPu')
    plt.show()
#just 1 dimension?
def drawROC():
    testData = [np.loadtxt('data/female.txt'), np.loadtxt('data/male.txt')]
    xs = []
    ys = []
    for th in range(140, 200):
        FPR = 0
        Fcnt = len(testData[0])
        TPR = 0
        Tcnt = len(testData[1])
        for person in testData[0]:
            if person[0] < th:
                FPR += 1
        for person in testData[1]:
            if person[0] < th:
                TPR += 1
        FPR /= Fcnt
        TPR /= Tcnt
        xs.append(FPR)
        ys.append(TPR)
    plt.xlabel("Female Positive Rate")
    plt.ylabel("Male Positive Rate")
    plt.plot(xs, ys)
    plt.show()



def hTest():
    T = trainHeight()
    errCnt = 0
    totCnt = 0
    testData = np.loadtxt('data/male.txt')

    for person in testData:
        totCnt+=1
        if classify(person[0], T) == 0:
            print("err: ",person, 'male result female')
            errCnt+=1
    testData = np.loadtxt('data/female.txt')
    for person in testData:
        totCnt += 1
        if classify(person[0], T) == 1:
            print("err: ", person, 'female result male')
            errCnt += 1
    print(errCnt / totCnt)

if __name__ == '__main__':
    drawROC()
    theta = trainHW()