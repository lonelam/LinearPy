import numpy as np
import matplotlib.pyplot as plt

def Gauss(x, avg, var):
    return np.exp(-((x-avg) **2) / var / 2.0) / np.sqrt(2.0 * np.pi * var)

def DataCollect():
    femaleDocPath = "data/girl.txt"
    maleDocPath = "data/boy.txt"
    femaleFile = open(femaleDocPath, "r")
    maleFile = open(maleDocPath, "r")
    data = [np.loadtxt(femaleFile), np.loadtxt(maleFile)]
    return data

def DataGen():
    data = DataCollect()
    for i in range(3):
        a = min(data[:,:,i])
        b = max(data[:,:,i])
        data[:,:,i] -= a
        data[:, :, i] /= (b-a)
    return data

def trainHeight():
    rawData = DataCollect()
    hData = [[x[0] for x in gender ] for gender in rawData]
    print(hData)
    theta = np.array([(np.average(gender), np.var(gender)) for gender in hData])
    print(theta)
    lF, = plt.plot(range(140, 200), [Gauss(x, theta[0][0], theta[0][1]) for x in range(140, 200)] , label='P(x|Female)')
    lM, = plt.plot(range(140, 200), [Gauss(x, theta[1][0], theta[1][1]) for x in range(140, 200)], label='P(x|Male)')
    plt.legend(handles=[lF,lM])
    plt.show()
    return theta

if __name__ == '__main__':
    trainHeight()
