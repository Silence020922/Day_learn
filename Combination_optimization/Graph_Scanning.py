test_list = [[1,2,3],[0,5],[0,3,5],[0,2,4],[3,5],[1,4]]
n = 6
s = 0
def Graph_Scan(adjlist,n,s):
    """
    Input: adjlist 邻接表
           n 点数
           s 起始点
    Output: R 从s出发能够到达的所有点的集合
            T R、T组成一个以s为根的树形图/树
    """
    viewed = [0]*n
    viewed[s] = 1
    R = [s]
    Q = [s]
    T = []
    while len(Q) != 0:
        v = Q[-1]
        w = s
        for vertex in adjlist[v]:
            if viewed[vertex] == 0:
                w = vertex
                break
        if w == s: 
            Q.pop()
            break
        else:
            R.append(w)
            Q.append(w)
            T.append([v,w])
            viewed[w] = 1
    return R,T

print(Graph_Scan(test_list,n,s))

    
    