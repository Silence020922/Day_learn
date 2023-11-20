vvw=[(1,2,3),(1,3,8),(1,5,-4),(2,4,1),(2,5,7),(3,2,4),(4,1,2),(4,3,-5),(5,4,6)]
n = 5
def find_mini_path(vvw,n):
    '''
    求解所有点对间的最短路径算法
    Input
    vvw :三元组邻接表
    n :点数
    Output
    D :D[i][j]表示点i到点j的最短加权路长
    P :P[i][j]表示i到j途径的点
    '''
    MAX = float('inf')
    m = len(vvw)
    D = [[MAX for i in range(n)] for j in range(n)]
    P = [[MAX for i in range(n)] for j in range(n)]
    for i in range(m):
        v1,v2 = vvw[i][0]-1,vvw[i][1]-1
        D[v1][v2]=vvw[i][2]
        P[v1][v2] = v1
    for k in range(n):
        for i in range(n):
            if i != k:
                for j in range(n):
                    if D[i][k] + D[k][j] < D[i][j]:
                        D[i][j] =  D[i][k] + D[k][j]
                        P[i][j] = P[k][j]
    return D,P+1
