import operator

class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        self.parent = parentNode
        self.children = {}
    def inc(self, numOccur):
        self.count += numOccur
    def __repr__(self):
        return "<Node " + self.name + ", " + str(self.count) + ">"
    def disp(self, ind=1):
        print(' '*ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(ind+1)

def createTree(dataSet, minSup=1):
    headerTable = {}
    for trans, cnt in dataSet.items():
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + cnt

    for k, cnt in list(headerTable.items()):
        if cnt < minSup:
            del(headerTable[k])
    freqItemSet = set(headerTable.keys())
    #并没有频繁项
    if len(freqItemSet) == 0:
        return None, None
    #树根初始权值为1，父节点为None
    retTree = treeNode('Null Set', 1, None)
    #第二次遍历dataSet，把每个计数超过最小支持度的项抽出
    for tranSet, count in dataSet.items():
        #从数据集的每一个项集中重新创建一个计数器
        localD = {}
        #从这个项集中把每一项都抽出来，并按照headerTable的计数排序
        for item in tranSet:
            if item in freqItemSet:
                localD[item] = headerTable[item][0]
        if len(localD) > 0:
            #按计数从大到小排序
            #orderedItems = [v[0] for v in sorted(localD.items(), key = lambda p: p[1], reverse = True)]
            orderedItems = [v[0] for v in sorted(localD.items(), key = lambda p: (p[1],p[0]), reverse = True)]
            updateTree(orderedItems, retTree, headerTable, count)
    return retTree, headerTable

def updateTree(items, inTree, headerTable, count):
    # items0为全局计数最大的那一项，也是该项集的首项
    # 如果已经有这条路径，则增加计数，否则创建一个节点
    if items[0] in inTree.children:
        inTree.children[items[0]].inc(count)
    else:
        #创建子节点
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        #更新或创建headerTable的新项，对headerTable的value来说，第0项为计数，第1项为引用
        headerTable[items[0]].append(inTree.children[items[0]])
    #如果已经走完了那就结束，否则继续往下走
    if len(items) > 1:
        #在整个过程当中count一直不变
        updateTree(items[1:], inTree.children[items[0]], headerTable, count)


def updateHeader(headerRow, targetNode):
    headerRow.extend(targetNode)

def ascendTree(leafNode, prefixPath):
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)

def findPrefixPath(basePat, tableRow):
    condPats = {}
    for treeNode in tableRow:
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1:
            #不同节点的前缀应当保证不同
            condPats[frozenset(prefixPath[1:])] = treeNode.count
    return condPats
#这里的preFix并不是指树上的前缀，而是指条件集
def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    #遍历，按headerTable的值（[计数, 头节点]，下标0处为计数值）排序后（当前的频繁项节点）
    bigL = [v[0] for v in sorted(list(headerTable.items()), key = lambda p: p[1][0])]
    for basePat in bigL:
        newFreqSet = preFix.copy()
        #这里要扩充条件集
        newFreqSet.add(basePat)
        freqItemList.append(newFreqSet)
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1:])
        #用新的条件模式基建树
        myCondTree, myHead = createTree(condPattBases, minSup)
        #在新的条件模式树上递归查找频繁项集，直到条件模式树为空，即没有频繁项集了
        if myHead != None:
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)


def loadSimpDat():
    simpDat = [list('rzhjp'),
               list('zyxwvuts'),
               list('z'),
               list('rxnos'),
               list('yrxzqtp'),
               list('yzxeqstm')]
    # simpDat = [list('abcd')[i:] for i in range(3)]
    return simpDat
def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict
if __name__ == '__main__':
    simpDat = loadSimpDat()
    initSet = createInitSet(simpDat)
    myFpTree, myHeaderTab = createTree(initSet, 3)
    myFpTree.disp()
    freqItems = []
    mineTree(myFpTree, myHeaderTab, 3, set([]), freqItems)
    print(freqItems)

