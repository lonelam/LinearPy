import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


preFemale = 0.3


def Gauss(x, avg, var):
    return np.exp(-((x-avg) **2) / var / 2.0) / np.sqrt(2.0 * np.pi * var)
def GaussN(X, Mu, Sigma):
    return np.exp(np.dot((X - Mu) ,np.linalg.pinv(Sigma) ).dot((X - Mu).transpose()) * (-0.5))
def windowF(u):
    return 1/np.sqrt(2*np.pi) * np.exp(-0.5*(u ** 2))
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
    P0 = len(hData[0]) / (len(hData[0]) + len(hData[1]))
    #print(hData)
    theta = np.array([(np.average(gender), np.var(gender)) for gender in hData])
    print(theta)
    lF, = plt.plot(range(140, 200), [Gauss(x, theta[0][0], theta[0][1]) * P0 for x in range(140, 200)] , label='P(x|Female)')
    lM, = plt.plot(range(140, 200), [Gauss(x, theta[1][0], theta[1][1]) * (1-P0) for x in range(140, 200)], label='P(x|Male)')
    plt.legend(handles=[lF,lM])
    plt.show()
    return theta, P0
def trainHW():
    rawData = DataCollect()
    hwData = [[[x[0] for x in gender], [x[1] for x in gender]] for gender in rawData]
    P0 = len(hwData[0]) / (len(hwData[0]) + len(hwData[1]))
    theta = [
        [
            np.array([np.average(gender[0]),np.average(gender[1])]) ,
            np.cov(gender[0], gender[1])
        ]
        for gender in hwData
    ]
    return theta, P0

def trainHW01():
    rawData = DataGen()
    hwData = [[[x[0] for x in gender], [x[1] for x in gender]] for gender in rawData]
    P0 = len(hwData[0]) / (len(hwData[0]) + len(hwData[1]))
    theta = [
        [
            np.array([np.average(gender[0]),np.average(gender[1])]) ,
            np.cov(gender[0], gender[1])
        ]
        for gender in hwData
    ]
    return theta, P0

def classify(x, theta, P0):
    if Gauss(x, theta[0][0], theta[0][1]) * P0 > Gauss(x, theta[1][0], theta[1][1]) * (1-P0):
        return 0
    return 1
def classifyN(X, Theta, P0, t):
    #print(GaussN(X, Theta[0][0], Theta[0][1]), GaussN(X, Theta[1][0], Theta[1][1]))
    if (GaussN(X, Theta[0][0], Theta[0][1]) * P0 / (GaussN(X, Theta[1][0], Theta[1][1])) * (1-P0) ) > t:
        return 0
    return 1
eps = 0.1

def drawFace(theta, P0):
    fig = plt.figure()
    X = np.arange(0, 1, 0.05)
    xlen = len(X)
    Y = np.arange(0, 1, 0.05)
    ylen = len(Y)
    X, Y = np.meshgrid(X, Y)
    Z0 = np.zeros((xlen, ylen))
    Z1 = np.zeros((xlen, ylen))
    for i in range(xlen):
        for j in range(ylen):
            Z0[i][j] = GaussN(np.array([X[i][j],Y[i][j]]), theta[0][0], theta[0][1]) * P0
            Z1[i][j] = GaussN(np.array([X[i][j], Y[i][j]]), theta[1][0], theta[1][1]) * (1-P0)
            # if Z0[i][j] < Z1[i][j]:
            #     Z0[i][j] = 1
            # else:
            #     Z1[i][j] = 0
    ax = fig.add_subplot(1,1,1,projection='3d')
    ax.plot_wireframe(X,Y,Z0,color = 'pink')
    ax.plot_wireframe(X,Y,Z1,color='blue')
    #bx = fig.add_subplot(1,2,1, projection='3d')
    #ax.plot_surface(X, Y, Z0 , cmap='PuBu')
    #  ax.plot_surface(X, Y, Z1, cmap='RdPu')
    plt.show()
#just 1 dimension?
def drawROC():
    testData = [np.loadtxt('data/girlnew.txt'), np.loadtxt('data/boynew.txt')]
    xs = []
    ys = []
    for th in range(140, 200):
        FPR = 0
        Fcnt = len(testData[0])
        TPR = 0
        Tcnt = len(testData[1])
        for person in testData[0]:
            if person[0] > th:
                FPR += 1
        for person in testData[1]:
            if person[0] > th:
                TPR += 1
        FPR /= Fcnt
        TPR /= Tcnt
        xs.append(FPR)
        ys.append(TPR)
    plt.xlabel("Female Positive Rate")
    plt.ylabel("Male Positive Rate")
    plt.plot(xs, ys)
    plt.show()


def drawROC2(theta, P0):
    testData = [np.loadtxt('data/girlnew.txt'), np.loadtxt('data/boynew.txt')]
    xs = []
    ys = []
    for t in np.linspace(-10,10,100):
        t = np.exp(t)
        #print(t)
        FPR = 0
        Fcnt = len(testData[0])
        TPR = 0
        Tcnt = len(testData[1])
        for person in testData[0]:
            if classifyN(person[:2], theta, P0, t) == 1:
                FPR += 1
                #print("err: ", person, 'male result female')
        for person in testData[1]:
            if classifyN(person[:2], theta, P0, t) == 1:
                TPR += 1
                #print("err: ", person, 'female result male')
        FPR /= Fcnt
        TPR /= Tcnt
        xs.append(FPR)
        ys.append(TPR)



    print(len(xs), len(ys))
    plt.xlabel("Female Positive Rate")
    plt.ylabel("Male Positive Rate")
    plt.plot(xs, ys, '-')
    t = 1
    # print(t)
    FPR = 0
    Fcnt = len(testData[0])
    TPR = 0
    Tcnt = len(testData[1])
    for person in testData[0]:
        if classifyN(person[:2], theta, P0, t) == 1:
            FPR += 1
            # print("err: ", person, 'male result female')
    for person in testData[1]:
        if classifyN(person[:2], theta, P0, t) == 1:
            TPR += 1
            # print("err: ", person, 'female result male')
    FPR /= Fcnt
    TPR /= Tcnt
    plt.plot(FPR, TPR, 'o', color='red')
    plt.show()

def hTest():
    T, P0 = trainHeight()
    errCnt = 0
    totCnt = 0
    testData = np.loadtxt('data/male.txt')

    for person in testData:
        totCnt+=1
        if classify(person[0], T, P0) == 0:
            print("err: ",person, 'male result female')
            errCnt+=1
    testData = np.loadtxt('data/female.txt')
    for person in testData:
        totCnt += 1
        if classify(person[0], T, P0) == 1:
            print("err: ", person, 'female result male')
            errCnt += 1
    print(errCnt / totCnt)
def Parzen(x, h, cls):
    data = DataCollect()
    px = 0
    for xi in data[cls]:
        px += windowF((x - xi[0]) / h)
    px /= len(data[cls]) * h
    return px


def ParzenPlot():
    h = 2
    xs = []
    ys = []
    for th in range(140, 200):
        xs.append(th)
        ys.append(Parzen(th, h, 0))
    plt.plot(xs, ys)
    xs = []
    ys = []
    for th in range(140, 200):
        xs.append(th)
        ys.append(Parzen(th, h, 1))
    plt.plot(xs, ys)
    plt.show()
def ParzenClassify(x, h, P0):
    if Parzen(x, h, 0) * P0 > Parzen(x, h, 1) * (1 - P0):
        return 0
    return 1
def ParzenROC():
    testData = [np.loadtxt('data/girlnew.txt'), np.loadtxt('data/boynew.txt')]
    data = DataCollect()
    P0 = len(data[0]) / (len(data[0]) + len(data[1]))
    xs = []
    ys = []
    for h in np.linspace(0, 3, 10):
        FPR = 0
        Fcnt = len(testData[0])
        TPR = 0
        Tcnt = len(testData[1])
        for person in testData[0]:
            if ParzenClassify(person[0], h, P0) == 1:
                FPR += 1
                # print("err: ", person, 'male result female')
        for person in testData[1]:
            if ParzenClassify(person[0], h, P0) == 1:
                TPR += 1
                # print("err: ", person, 'female result male')
        FPR /= Fcnt
        TPR /= Tcnt
        xs.append(FPR)
        ys.append(TPR)
    print(xs, ys)
    plt.plot(xs, ys, '-')
    plt.show()
if __name__ == '__main__':
    ParzenROC()
