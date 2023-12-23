v = 0
adjlist = [[1,4],[0,2,5,6],[1,3],[2,4],[0,3],[1,6],[1,5]]
n= 7
P = []
def dfs(node):
    while adjlist[node]:
        v = adjlist[node].pop()
        adjlist[v].remove(node) # 因为是无向图，这里的原理是删除边，需删除两个位置。
        dfs(v)
    P.append(node)
dfs(v)
print(P)





