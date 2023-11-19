def partition_num(num, workers):
    """
    输入数据数量, 将其分配给workers, 输出list
    """
    if num % workers == 0:
        return [num // workers] * workers
    else:
        return [num // workers]*workers + [num % workers]
    
def pre_nxgraph(graph):
    """
    又是一个嘛也没干的大工程，输入图，输出
    node_list
    node_dict
    """
    node2idx = {}
    idx2node = []
    node_idx = 0
    for node in graph.nodes():
        node2idx[node] = node_idx
        idx2node.append(node)
        node_idx += 1
    return idx2node,node2idx

def partition_dict(vertices, workers):
    """
    Input vertices, dict
          workers, int
    Output part_list
    """
    batch_size = (len(vertices) - 1)//workers + 1 # 平均分配给workers
    part_list = []
    part = []
    count = 0
    for v,nbs in vertices.items():
        part.append((v,nbs))
        count += 1
        if count % batch_size == 0: # 不需要每次都重置
            part_list.append(part)
            part = []
    if len(part) > 0 : # count 没有再次达到batch_size的整数倍，part没有被处理。
        part_list.append(part)
    return part_list

def partition_list(vertices, workers):
    """
    Input vertices, list
          workers, int
    Output part_list
    """
    batch_size = (len(vertices) - 1)//workers + 1 # 平均分配给workers
    part_list = []
    part = []
    count = 0
    for v,nbs in enumerate(vertices): # 与partition_dict唯一不同 ，将vertices转化为枚举值
        part.append((v,nbs))
        count += 1
        if count % batch_size == 0: # 不需要每次都重置
            part_list.append(part)
            part = []
    if len(part) > 0 : # count 没有再次达到batch_size的整数倍，part没有被处理。
        part_list.append(part)
    return part_list
list = partition_num(10,workers=2)