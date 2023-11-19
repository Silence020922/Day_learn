class Solution(object):
    def findSmallestSetOfVertices(self, n, edges):
        """
        :type n: int
        :type edges: List[List[int]]
        :rtype: List[int]
        """
        ans = []
        node_list = [-1] * n
        for i in edges:
            node_list[i[1]] = 1
        for i in range(n):
            if  node_list[i] == -1:
                ans.append(i)
        return ans
            

