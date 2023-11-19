def Triarray2adjlist(n, vvw):  # 将三元数组转化为邻接表的形式
    adjlist = []
    for i in range(n):
        col_edge = []
        for edge in vvw:
            v1, v2 = edge[0], edge[1]
            if v1 == i:
                col_edge.append((v2, edge[2]))
            if v2 == i:
                col_edge.append((v1, edge[2]))
        adjlist.append(col_edge)
    return adjlist


def Prim(n, adjlist, startv):  # 输入邻接表
    path = [-1] * n
    lowcost = [float("inf")] * n  # 初始定义所有点cost为无穷，起始点cost为0
    lowcost[startv] = 0
    left = set() # 记录未参与到划分中的点。
    left.add(startv)
    while len(left) > 0:
        min, k = float("inf"), -1
        for i in left:
            if lowcost[i] < min:
                min, k = lowcost[i], i
        if k >= 0:
            left.remove(k)
        print("edge  =(" + str(path[k]) + "," + str(k) + "), cost = " + str(lowcost[k]))
        col_edge = adjlist[k] # 刚检测的点的邻接表所在行
        for edge in col_edge:
            j = edge[0]
            if lowcost[j] > edge[1] and j in left: # j in left 才能进行
                lowcost[j], path[j] = edge[1], k
            elif lowcost[j] == float("inf"): # 在首次将所有的都加入了left，并对与现有划分求距离。
                lowcost[j], path[j] = edge[1], k
                left.add(j)  # add


vvw = [
    (0, 1, 1),
    (0, 2, 2),
    (1, 2, 2),
    (1, 3, 4),
    (1, 4, 3),
    (3, 4, 2),
    (2, 4, 4),
    (2, 3, 4),
]
adjlist = Triarray2adjlist(5, vvw)
Prim(5, adjlist, 0)
