# 基于斐波那契堆的prim算法求解最小树
# O(m + n + nlogp )
import math

class Node(object):
    def __init__(self):
        self.cost = float('inf') # 到sv的距离
        self.pre = -1 # 顶点的前序点
        self.degree = 0 # 孩子的个数
        self.parent = -1 # 双亲指针
        self.child = -1 # 孩子指针
        self.left = -1 # 双向循环链表的左指针
        self.right = -1 # 双向循环链変的右指针
        self.flag = 0 # 0此点为白点，1黑，2已进入最小树
        
def plant(A,k1,Head): #Head 堆的根指针，合并树根用，Head[i]为第i棵树对应图的顶点的下标。要插入的为k1。A用来存储点的信息和斐波那契堆
    deg_i = A[k1].degree # 找到要录入点的出度
    A[k1].flag, A[k1].left,A[k1].right,A[k1].parent = 0, -1, -1, -1 #初始化
    while Head[deg_i] >= 0: # 由于初始化为-1, 若为-1则将k1直接作为度为deg_i的根节点接入即可，否则
        k2 = Head[deg_i] #找到当前这位根节点,这里以cost值对应算法中的d(u)
        if A[k2].cost <= A[k1].cost:
            A[k1].parent = k2
            A[k2].degree += 1
            if A[k2].degree == 1: # 也就是说原来k1 和 k2 都没有孩子节点
                A[k2].child,A[k1].left,A[k1].right = k1,k1,k1
            else:
                k3 = A[k2].child  # 把k1放到k2节点的孩子k3 k4中间，并调整相应的指向
                k4 = A[k3].right
                A[k1].left = k3,A[k1].right = k4
                A[k3].right = k1,A[k4].left = k1
            k1 = k2 
        else: # 此时A[k1].cost 和 A[k2].cost 有其他关系
            A[k2].parent = k1 # 重复刚才操作
            A[k1].degree += 1 
            if A[k1].degree == 1: # 也就是说原来k1 和 k2 都没有孩子节点
                A[k1].child,A[k2].left,A[k2].right = k2,k2,k2
            else:
                k3 = A[k1].child  
                k4 = A[k3].right
                A[k2].left = k3,A[k2].right = k4
                A[k3].right = k2,A[k4].left = k2
        Head[deg_i] = -1 # 无论将k1给到k2 还是将k2给到k1 都会使得原本度为deg_i的位置消失
        deg_i += 1
    Head[deg_i] = k1

def delMin(A,Head): # 删除最小值
    Min, cost = -1, float('inf') # 由堆特殊性只从子树根节点寻找
    for i in Head:
        if i >= 0 and A[i].cost < cost:
            Min,cost = i,A[i].cost
    if Min >= 0:
        k1,Degree = A[Min].child,A[Min].degree # 某个孩子
        Head[Degree] = -1 # 先将Min从堆中删除，置对应根指针数值为-1
        for i in range(Degree): #由于为循环指针，仅right能够遍历Min所有孩子
            k2 = A[k1].right 
            plant(A,k1,Head)
            k1 = k2
    return Min

def decreaseKey(A,i,Head): # 堆中元素i对应的值减少后向上修正堆
    i_p = A[i].parent 
    if i_p >=0 and A[i].cost < A[i_p].cost:
        A[i_p].degree -= 1
        if A[i_p].degree == 0 : # 如果此次操作使得没有孩子
            A[i_p].child = -1
        else:
            A[i_p].child = A[i].right
            i_l,i_r = A[i].left,A[i].right #这时候将i从左右邻居扣掉，所以对应左邻居右邻居和右邻居的左邻居发生变化
            A[i_l].right = i_r, A[i_r].left = i_l
        plant(A,i,Head)

        i = i_p
        while A[i].parent >= 0:
            i_p = A[i].parent
            if A[i].flag == 0:
                A[i].flag == 1
                break
            else:
                A[i_p].degree -= 1
                if A[i_p].degree == 0:
                    A[i_p].child = -1
                else:
                    A[i_p].child = A[i].right # 把i删除了
                    i_l,i_r = A[i].left,A[i].right
                    A[i_l].right, A[i_r].left = i_r,i_l
                plant(A,i,Head)
            i = i_p

        if A[i].parent == -1:
            j = A[i].degree +1
            Head[j] = -1
            plant(A,i,Head)   

n, sv, adjlist = 6,2,[]
# 主程序
A = [Node() for i in range(n)]
m = int(2*math.log(n))+2
Head = [-1 for i in range(m)]
Head[0],A[sv].cost,count = sv,0,0
while True: # 6
    count += 1
    Min = delMin(A,Head)
    if Min == -1 : break
    A[min].flag = 2 # 在树中
    if Min == sv: print('最小数为[(点，点)：值]：',end=' ')
    else:print('('+str(A[Min].pre) + ',' + str(A[Min]) + '):' + str(A[Min].cost) ,end=';')
    for j, weight in adjlist[Min]:
        if A[j].cost == float('inf'): 
            A[j].pre = Min, A[j].cost = weight
            plant(A,j,Head)
        elif A[j].flag <= 1 and A[j].cost > weight:
            A[j].cost = weight,A[j].pre = Min
            decreaseKey(A,j,Head)

if count < n :
    print('\n\n 树不连通')