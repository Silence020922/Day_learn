def K_root(vvw, n):  # 回路判断使用路径压缩的根树判断思想
    vvw = sorted(vvw, key=lambda x: x[2])
    parent = [-1] * n
    tree = []
    for edge in vvw:
        v1, v2 = edge[0], edge[1]
        # 找到v1的根节点
        root1 = v1
        while parent[root1] != -1:
            root1 = parent[root1]

        # 找到v2的根节点
        root2 = v2
        while parent[root2] != -1:
            root2 = parent[root2]

        if root1 != root2:
            tree.append(edge)
            if len(tree) == n - 1:
                break
        # 将v1到其根节点的所有点归类到v1的根节点
            while parent[v1] != -1:
                tmp = parent[v1]
                parent[v1] = root1
                v1 = tmp
        # 将v2到其根节点的所有点归类到v1的根节点
            while v2 != -1:
                tmp = parent[v2]
                parent[v2] = root1
                v2 = tmp
    if len(tree) < n - 1:
        print("No sapnning tree.")
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
K_root(vvw, 5)
