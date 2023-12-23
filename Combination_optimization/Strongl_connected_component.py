n = 7
adjlist = [[5],[2],[1],[1,5],[0,1,2,3],[4],[2,4]]
viewed = [0]*n
P_dfs = []
scc = []
def dfs(adjlist,u): # 给到出栈顺序
    viewed[u] = 1
    for v in adjlist[u]:
        if not viewed[v]:
            dfs(adjlist,v)
    P_dfs.append(u)

def reverse_adj(adjlist,n):
    r_adjlist = [[] for _ in range(n)]
    for v in range(n):
        if len(adjlist[v])!= 0:
            for w in adjlist[v]:
                r_adjlist[w].append(v)
    return r_adjlist

def rdfs(adjlist,u,scc):
    viewed[u] = 1
    scc.append(u)
    for v in adjlist[u]:
        if not viewed[v]:
            rdfs(adjlist,v,scc)

dfs(adjlist,0) # 第一次dfs
P_dfs.reverse() # 反回出栈顺序，由于第二次dfs从后向前取点
viewed = [0]*n # 重置一下
r_adj = reverse_adj(adjlist,n) # 反向邻接表
for i in range(n):
    if not viewed[i]: # i还不在任何强连通分支中
        scc = []
        rdfs(r_adj,i,scc)
        print(scc)