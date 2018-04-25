from fpGrowth import createTree, mineTree, createInitSet
import csv
import time
import math
datafile = open("data/trade.csv", "r", encoding="utf-8")
reader = csv.reader(datafile.readlines())
user = {}
print(reader.__next__())
# cnt = 0
namedict = {}
#plu
# NAMEPOSITON = 8
# NOPOSITION = 6
#dpt
NAMEPOSITON = 12
NOPOSITION = 11
#bnd
# NAMEPOSITON = 14
# NOPOSITION = 13
namedict[-1] = "UNKNOWN"
row_by_user = {}

for row in reader:
    # cnt+= 1
    if row[4] not in row_by_user:
        row_by_user[row[4]] = []
    row[1] = time.strptime(row[1], "%Y-%m-%d %H:%M:%S")
    row_by_user[row[4]].append(row)


for u in row_by_user:
    req_num = math.ceil(len(row_by_user[u]) * 0.6)
    # print(len(row_by_user[u]))
    row_by_user[u] = sorted(row_by_user[u], key = lambda p: p[1])[:req_num]
    user[u] = {}
    for row in row_by_user[u]:
        if row[0] not in user[u]:
            user[u][row[0]] = set()
        user[u][row[0]].add(int(row[NOPOSITION]) if row[NOPOSITION] != "" else -1)
        if row[NOPOSITION] != "":
            namedict[int(row[NOPOSITION])] = row[NAMEPOSITON]
    # user[row[4]][row[0]][row[7]] = user[row[4]][row[0]].get(row[7], 0) + 1


for MINSUP in [2,4,8,16,32,64]:
# MINSUP = 2# print(cnt)
    for vip, uidset in user.items():
        # print(uidset)
        data = createInitSet(uidset.values())
        # print(data)
        myFpTree, myHeaderTab = createTree(data, MINSUP)
        if myFpTree is not None:
            print("vipno #", vip, "的支持度大于", MINSUP, "的频繁项集如下")
            freqItems = []
            mineTree(myFpTree, myHeaderTab, MINSUP, set([]), freqItems)
        # print(freqItems)
            for freq in freqItems:
                for x in freq:
                    print(namedict[x], end=",")
                print("\n**************")
        else:
            # pass
            print("vipno #", vip, "并没有支持度大于", MINSUP, "的频繁项集")
