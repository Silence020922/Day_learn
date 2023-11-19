def K_sig(vvw, n):  # 输入三元组，标号判断回路的Kruskal算法
    vvw = sorted(vvw, key=lambda x: x[2])  # 使用lambda函数，意为 key(x) = x[2]
    parent = [i for i in range(n)]  # 首先以每个点作为一棵子树
    tree = []
    for edge in vvw:
        root1 = parent[edge[0]]
        root2 = parent[edge[1]]
        if root1 != root2:
            tree.append(edge)
            if len(tree) == n - 1:
                break
            for i in range(n):
                if parent[i] == root2:
                    parent[i] = root1
    if len(tree) < n - 1:
        print("Don't connected")
    else:
        print(tree)


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
K_sig(vvw, 5)
