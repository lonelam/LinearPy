from fpGrowth import createTree, mineTree, createInitSet
import csv

datafile = open("data/trade.csv", "r", encoding="utf-8")
reader = csv.reader(datafile.readlines())


user = {}
print(reader.__next__())
# cnt = 0
for row in reader:
    # cnt+= 1
    if row[4] not in user:
        user[row[4]] = {}
    if row[0] not in user[row[4]]:
        user[row[4]][row[0]] = set()
    user[row[4]][row[0]].add(int(row[7]))
    # user[row[4]][row[0]][row[7]] = user[row[4]][row[0]].get(row[7], 0) + 1


MINSUP = 2# print(cnt)
for vip, uidset in user.items():
    print("vipno #", vip, "的支持度大于", MINSUP, "的频繁项集如下")
    data = createInitSet(uidset.values())
    # print(data)
    myFpTree, myHeaderTab = createTree(data, MINSUP)
    if myFpTree is not None:
        freqItems = []
        mineTree(myFpTree, myHeaderTab, MINSUP, set([]), freqItems)
    print(freqItems)