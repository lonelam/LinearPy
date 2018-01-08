import numpy as np
def dataGen():
    X = []
    ys = []
    for i in range(20):
        x0 = np.random.randint(0, 100)
        x1 = np.random.randint(0, 100)
        y = np.random.randint(-20, 20) + 5 * x0 - x1
        X.append([x0, x1])
        ys.append(y)

    np.savetxt("NormalEData/X.txt", X)
    np.savetxt("NormalEData/y.txt", ys)


if __name__ == '__main__':
    dataGen()