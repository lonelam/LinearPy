from fpGrowth import createTree, mineTree, createInitSet
import csv

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

for row in reader:
    # cnt+= 1
    if row[4] not in user:
        user[row[4]] = {}
    if row[0] not in user[row[4]]:
        user[row[4]][row[0]] = set()
    user[row[4]][row[0]].add(int(row[NOPOSITION]) if row[NOPOSITION] != "" else -1)
    if row[NOPOSITION] != "":
        namedict[int(row[NOPOSITION])] = row[NAMEPOSITON]
    # user[row[4]][row[0]][row[7]] = user[row[4]][row[0]].get(row[7], 0) + 1


for MINSUP in [2,4,8,16,32,64]:
# MINSUP = 2# print(cnt)
    for vip, uidset in user.items():
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
