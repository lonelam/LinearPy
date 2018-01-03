from kNN import *
import matplotlib
import matplotlib.pyplot as plt



if __name__ == '__main__':
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    #print(shape(datingDataMat))
    normMat, ranges, minVals = autoNorm(datingDataMat)
    #print(normMat)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(normMat[:,1], normMat[:,2],
               15.0 * array(datingLabels), 15.0 * array(datingLabels))
    plt.show()