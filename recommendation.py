import random
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
# 随机初始化邻接矩阵
# 本文默认A为邻接矩阵


def initial_mat(s):
    X = np.random.randint(0, 2, size=(s, s))
    X = np.triu(X)
    X += X.T - 2*np.diag(X.diagonal())
    return X

# 随机断边函数，保证过程总边数不变


def cut_link(A):
    G = nx.from_numpy_matrix(A)
    m = len(list(G.nodes()))
    # print("m",m)
    t = np.random.randint(0, m)
    # print("t",t)
    all_neighbor_list = list(nx.all_neighbors(G, t))
    if len(all_neighbor_list) != 0:
        G.remove_edge(t, all_neighbor_list[int(
            random.random()*len(all_neighbor_list))])

        A = np.array(nx.adjacency_matrix(G).todense())
        return A

    else:
        return cut_link(A)
# 依据结构相似性及意见相似性推荐朋友,通过改变s进而改变推荐系统的作用程度


def choose_point(A, s, opinion):
    G = nx.from_numpy_matrix(A)
    # A = nx.adjacency_matrix(G).todense()
    nodes_num = len(list(G.nodes()))
    t = np.random.randint(0, nodes_num)
    all_point_list = list((G.nodes()))
    sign_op = np.sign(opinion)
    P = []
    p = 0
    j = 0
    for i in all_point_list:
        p += (len(list(nx.common_neighbors(G, t, i))))**2 + 0.1
        P.append(p)
    # print("P", P)
    N = P[-1]
    n = random.random()*N
    while n > P[j]:
        j += 1
        if j > nodes_num - 2:
            break
    # print("j",j)
    # print(nodes_num)
    # print(t)
    # print(A[t,j])
    if t != j:
        # print("A[t,j]",A[t,j])
        if A[t, j] == 0:
            # print(sign_op[t] == sign_op[j],sign_op[t],sign_op[j])
            if sign_op[t] == sign_op[j]:
                return t, j
            else:
                if random.random()*10 > s:
                    return t, j
                else:
                    return choose_point(A, s, opinion)
        else:
            return choose_point(A, s, opinion)
    else:
        return choose_point(A, s, opinion)
# 主推荐系统函数，返回形式为邻接矩阵


def recommend(A, opinion, s=5):
    A = cut_link(A)
    # print("A_cut",A)
    (t, recommended_point) = choose_point(A, s, opinion)
    A[t, recommended_point] = 1
    A[recommended_point, t] = 1
    return A


# 初始化个体意见,s表示群体数量,K表示社会影响控制,正代表积极观点，负代表消极观点,p表示收敛点比例
def initial_op(n, K, p):
    # 返回初始观点同时返回0-1矩阵，其中0表示收敛点，1表示发散点
    convergence_p = np.random.choice([0], size=(int(n*p)))
    converge_mesa = convergence_p.tolist()
    divergency_p = np.random.choice([1], size=(n-int(p*n)))
    converge_mesa.extend(divergency_p)
    return K*np.random.uniform(-1, 1, size=(n)), np.array(converge_mesa)

# 与邻居交互作用对个体观点的影响采用该模型 $x_i(t+1) = \gamma x_i(t) + K \sum(A_{i,j}) tanh(\alpha x_j(t))/k_i$

# A = initial_mat(4)
# print(A)  # 初始化邻接矩阵
# (opinion_initial1, B1) = initial_op(4, 5, 1)
# print(opinion_initial1)
# for i in np.arange(10):
#     A = recommend(A,opinion_initial1,2)
# print(A)


def opining(opinion, A, lanmda=0, alpha=0.2, gamma=0.7, K=1):  # gamma 为衰减指数
    # tanh 用于非线性约束影响
    sign = np.sign(opinion)
    opinion_sign = np.matrix(sign)
    sign_mat = np.dot(opinion_sign.T, opinion_sign)  # 解决array一维转置仍为一维的问题
    sign_array = np.array(sign_mat)  # 转换为默认的形式，防止出现其他问题
    optaff = np.tanh(alpha*opinion)  # 个人观点转化为影响力
    each_neighbor = np.sum(A, axis=1)  # 邻接矩阵按列求和表邻居个数
    aff_t_one = np.dot(optaff, A*sign_array**lanmda)  # 影响力实际对个人的影响
    mean_af = K * np.divide(aff_t_one, each_neighbor, out=np.zeros_like(
        aff_t_one), where=each_neighbor != 0)  # 避免无人交互/0的情况，使得无人交互时直接将影响取零
    return gamma*opinion + mean_af


# 节点数量，交互次数,p为收敛占比,s推荐系统,alpha观点转化影响力,gamma意见衰减,社会影响约束，
def test_opning(A_initial, opinion_initial, B, num,  n, p, s, alpha=0.2, gamma=0.5, K=5):
    # A_initial = initial_mat(num)
    A = A_initial
    # (opinion_initial, B) = initial_op(num, K, p)  # B为收敛发散标签
    opinion = opinion_initial
    x = np.arange(0, n)  # 生成迭代次数并作为x坐标
    opinion_list = []  # 记录观点变化过程用于画图
    if s == -1:    # 此时不调用推荐系统
        for i in x:
            opinion = opining(opinion, A_initial, B, alpha, gamma, K)
            opinion_list.extend(opinion)
        opinion_array = np.array(opinion_list).reshape(n, num)  # 意见过程
    else:
        for i in np.arange(1, 1000):
            A = recommend(A, opinion, s)

        for i in x:
            # A = recommend(A,  opinion,s)
            opinion = opining(opinion, A, B, alpha, gamma, K)
            opinion_list.extend(opinion)
        opinion_array = np.array(opinion_list).reshape(n, num)
    return int(p*num), x, opinion_array, A


def draw_pic(x, y, mod, n, pic):  # x轴 y轴 两种标签 green red 分别为 收敛 发散点
    for i in range(0, mod):
        pic.plot(x, y[:, i], color='green')
    for j in range(mod, n):
        pic.plot(x, y[:, j], color='red')  # 画图过程
#     plt.show()


def main():
    pic, axes = plt.subplots(nrows=2, ncols=2)  # 初始化画布
    axes[0, 0].set(title='p=1,s=-1,alpha = 0.1')
    axes[0, 1].set(title='p=1,s =-1,alpha =0.2')
    axes[1, 0].set(title='p=0.8,s=-1,alpha =0.2')
    axes[1, 1].set(title='p=1,s=1,alpha=0.2')
    size = 20
    num = 30
    A1_initial = initial_mat(size)  # 初始化邻接矩阵
    (opinion_initial1, B1) = initial_op(size, 10, 1)
    (mod1, x1, y1, A1) = test_opning(A1_initial,
                                     opinion_initial1, B1, size, num, 1, -1, 0.1)
    for i in range(0, mod1):
        axes[0, 0].plot(x1, y1[:, i], color='green', linewidth=0.8)
    for j in range(mod1, size):
        axes[0, 0].plot(x1, y1[:, j], color='red', linewidth=0.8)  # 画图过程

    (mod2, x2, y2, A2) = test_opning(A1_initial,
                                     opinion_initial1, B1, size, num, 1, -1)
    for i in range(0, mod2):
        axes[0, 1].plot(x2, y2[:, i], color='green', linewidth=0.8)
    for j in range(mod2, size):
        axes[0, 1].plot(x2, y2[:, j], color='red', linewidth=0.8)  # 画图过程
    (mod3, x3, y3, A3) = test_opning(
        A1_initial, opinion_initial1, B1, size, num, 1, 8)
    for i in range(0, mod3):
        axes[1, 1].plot(x3, y3[:, i], color='green', linewidth=0.8)
    for j in range(mod3, size):
        axes[1, 1].plot(x3, y3[:, j], color='red', linewidth=0.8)  # 画图过程

    A2_initial = initial_mat(size)  # 初始化邻接矩阵
    (opinion_initial2, B2) = initial_op(size, 5,0.6)
    (mod4, x4, y4, A4) = test_opning(
        A2_initial, opinion_initial2, B2, size, num, 0.6, -1)
    for i in range(0, mod4):
        axes[1, 0].plot(x4, y4[:, i], color='green', linewidth=0.8)
    for j in range(mod4, size):
        axes[1, 0].plot(x4, y4[:, j], color='red', linewidth=0.8)  # 画图过程
    plt.show()


main()

