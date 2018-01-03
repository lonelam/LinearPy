'''
 @Time    : 2017/12/24 上午12:06
 @Author  : Eric Wang
 @File    : FP-tree.py
 @license : Copyright(C), Eric Wang
 @Contact : eric.w385@gmail.com
'''

class Node:
    def __init__(self, Nodevalue, numOccur, parentNode):
        '''
        basic Node class
        :param Nodevalue: the key value for numOccur (list)
        :param numOccur: number of the item in the dataset (int)
        :param parentNode: (Node)
        '''
        self.value = Nodevalue
        self.count = numOccur
        self.Nodelink = None
        self.parent = parentNode
        self.children = {}

    def inc(self, numOccur):
        '''
        add numOccur to node count
        :param numOccur:
        '''
        self.count += numOccur

    def disp(self, ind=1):
        '''
        draw the FP-tree
        '''
        print('  ' * ind, self.value, ' ', self.count)
        for child in self.children.values():
            child.disp(ind + 1)

def loadSimpDat():
    '''
    load test data
    :return: dataset(list)
    '''
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat

def createInitSet(dataSet):
    '''
    normalize the raw data
    :param dataSet: raw dataset(list)
    :return: nomalized dataset(dic)
    '''
    retDict = {}
    for trans in dataSet:
        if not retDict.__contains__(frozenset(trans)):
            retDict[frozenset(trans)] = 1
        else:
            retDict[frozenset(trans)] += 1
    return retDict

def updateTree(items, Tree, header, count):
    '''
    update the FP-tree by sorted events
    :param items: sorted item key flitered by minSup (list)
    :param Tree: FP-Tree
    :param header: sorted item flitered by minSup (dic)
    :param count: number of the item in the dataset (int)
    '''

    if items[0] in Tree.children:
        Tree.children[items[0]].inc(count)
    else:
        #add a new children node for the FP-Tree
        Tree.children[items[0]] = Node(items[0], count, Tree)
        #if still any item in items then update head pointer
        if header[items[0]][1] == None:
            header[items[0]][1] = Tree.children[items[0]]
        else:
            updateHeader(header[items[0]][1], Tree.children[items[0]])
    if len(items) > 1:
        updateTree(items[1::], Tree.children[items[0]], header, count)

def updateHeader(nodeToTest, targetNode):
    '''
    update the head pointer
    :param nodeToTest: node flitered by the minSup
    :param targetNode: next Node
    '''
    #update the key value of head pointer to next Node
    while (nodeToTest.Nodelink != None):
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode


def createTree(dataSet, minSup=1):
    '''
    create the FP-tree
    :param dataSet: dataset(dic)
    :param minSup: support threshold(int)
    :return: FP-tree
    '''
    header = {}
    #calclate the number of each item
    for trans in dataSet:
        for item in trans:
            header[item] = header.get(item,0) + dataSet[frozenset(trans)]
    # delete the item below minSup
    for k in list(header):
        if header[k] < minSup:
            del(header[k])
    freqItemSet = set(header.keys())
    if len(freqItemSet) == 0:
        return None, None
    for k in header:
        header[k] = [header[k], None]
    Tree = Node('Null Set', 1, None)
    #create the tree
    for transSet, count in dataSet.items():
        localID = {}
        for item in transSet:
            if item in freqItemSet:
                localID[item] = header[item][0]
        # sort the item in each event
        if len(localID) > 0:
            orderedItems = [v[0] for v in sorted(localID.items(),
                                                 key=lambda p: p[1],
                                                 reverse=True)]
            updateTree(orderedItems, Tree, header, count)
    return Tree, header

def ascendTree(leafNode, prefixPath):
    '''
    search the parent Node from a leaf Node
    :param leafNode: the Node to search
    :param prefixPath: the name of Nodes(list)
    '''

    if leafNode.parent is not None:
        prefixPath.append(leafNode.value)
        ascendTree(leafNode.parent, prefixPath)

def findPrefixPath(basePat, Node):
    '''
    find the basic fluent item dataset
    :param basePat: the value to search
    :param Node: the Node to search
    :return:
    '''

    condPats = {}
    while Node is not None:
        prefixPath = []
        #search the parent Node
        ascendTree(Node, prefixPath)
        if len(prefixPath) > 1:
            condPats[frozenset(prefixPath[1:])] = Node.count
        Node = Node.Nodelink
    return condPats

def mineTree(Tree, header, minSup, preFix, freqItemList):
    '''
    create the Fp-Tree and mine it
    :param Tree: FP-Tree
    :param header: sorted item flitered by minSup (dic)
    :param minSup: support threshold(int)
    :param preFix:
    :param freqItemList: frequent items(list)
    :return:
    '''
    print(header)
    bigL = [v[0] for v in sorted(header.items(), key=lambda p: p[1][0])]
    print('-----', sorted(header.items(), key=lambda p: p[1][0]))
    print('bigL=', bigL)
    #
    for basePat in bigL:
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)

        freqItemList.append(newFreqSet)
        condPattBases = findPrefixPath(basePat, header[basePat][1])

        myCondTree, myHead = createTree(condPattBases, minSup)
        if myHead is not None:
            myCondTree.disp(1)
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)


if __name__ == "__main__":
    dataset = loadSimpDat()
    initSet = createInitSet(dataset)
    print(initSet)

    FPtree ,myHeader = createTree(initSet, 3)
    FPtree.disp()

    print('x --->', findPrefixPath('x', myHeader['x'][1]))
    print('z --->', findPrefixPath('z', myHeader['z'][1]))
    print('r --->', findPrefixPath('r', myHeader['r'][1]))

    freqItemList = []
    mineTree(FPtree, myHeader, 3, set([]), freqItemList)
    print(freqItemList)


