
# 开始使用常规思路进行求解，主要思路为一轮一轮的查询所有课程，将该轮碰
# 到的所有无前置需要的课程进行学习，并做好计数，同时将已学习的课程需求删
# 除，进行下一轮查询，直到某一轮查询完成后并没有发现无前置课程，此时说明
# 无剩余课程或剩下课程不可选，但运行速度很慢。

class Solution(object):
    def canFinish(self,numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: bool    
        """
        if len(prerequisites) <2:
            return True
        adjlist_re = [[] for _ in range(numCourses)]
        ind = [0] * numCourses
        for course in prerequisites:
            adjlist_re[course[1]].append(course[0]) # 记录该点是哪些点的前置点
            ind[course[0]] += 1
        viewd = [0]*numCourses
        tmp = 0 # 代表当前检测的课程数
        course_num = 0 # 代表已经选择的课程
        while tmp == 0:
            tmp = 1
            for vertex in range(numCourses):
                if ind[vertex] == 0 and viewd[vertex] == 0:
                    viewd[vertex] = 1 # 检测过
                    course_num+=1
                    for v in adjlist_re[vertex]:
                        ind[v] -= 1
                    tmp = 0 # 再次从头开始
        if course_num == numCourses: return True
        return False

# 改进，考虑到一开始可以将所有的无前置课程选出，而之后每次的无前置
# 课程只能是因为在上一次中将其所需的课程选出产生，所以n阶段只需要
# 考虑以n-1阶段选的课为前置的课即可。同时，这也说明若遍历当前阶段
# 的所有无前置课程后，下一阶段未发现新产生的无前置课程则遍历结束。
# 这大大的缩短了运行时间，下面代码达到了100% 92%。
class Solution(object):
    def canFinish(self,numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: bool
        """
        if len(prerequisites) <2:
            return True
        queue = collections.deque()
        adjlist_re = [[] for _ in range(numCourses)]
        ind = [0] * numCourses # 入度
        for course in prerequisites:
            adjlist_re[course[1]].append(course[0]) # 记录该点是哪些点的前置点
            ind[course[0]] += 1
        for i in range(numCourses):
            if ind[i] == 0:
                queue.append(i)
        while queue:
            current = queue.popleft() # 弹出一个点
            numCourses -= 1
            for v in adjlist_re[current]:
                ind[v] -= 1
                if ind[v] == 0: queue.append(v)
        if numCourses == 0: return True
        return False

