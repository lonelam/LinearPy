import datetime


ans = {}
def dfs(s, tar):
    #print(s)
    if len(s) == tar:
        try:
            f = 0
            for x in range(2):
                for y in range(2):
                    for z in range(2):
                        f += eval(s.replace('!', ' not '), {'x': x, 'y': y, 'z': z}) << (7 - x*4-y*2-z)
            if ans.get(f) == None:
                ans[f] = s
                global cnt
                cnt += 1
        except:
            pass
        return
    if s == "" or s[-1] in "!&(|":
        for ch in "!(xyz":
            dfs(s + ch, tar)
    else:
    #if s[-1] in ")xyz":
        for ch in "!&)|":
            dfs(s + ch, tar)
    #elif s[-1] in "!&(|":
    # else:
    #     for ch in "(xyz":
    #         dfs(s + ch, tar)
    return
if __name__ == '__main__':
    d = 0
    up = 1 << 8
    cnt = 0
    #chs = "!&()xyz|"
    t_s = datetime.datetime.now()
    while cnt < (1 << 8):
        d += 1
        dfs("", d)
        print(d, ' ', cnt)

    print("finitsh", d)
    print("time usage: ", datetime.datetime.now() - t_s)
    for it in ans.items():
        print('{', ans[0], ',', ans[1], '},')
    n = int(input())
    for i in range(n):
        f = int(input(), base=2)
        print(ans[f])
