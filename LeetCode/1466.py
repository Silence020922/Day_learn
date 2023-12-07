class Solution(object):
    def minReorder(self, n, connections):
        """
        :type n: int
        :type connections: List[List[int]]
        :rtype: int
        """
        Adjlist = [[] for _ in range(n)]
        UdAdjlist = [[] for _ in range(n)]
        for edge in connections:
            Adjlist[edge[0]].append(edge[1])
            UdAdjlist[edge[0]].append(edge[1])
            UdAdjlist[edge[1]].append(edge[0])
        path = []
        visited = [0]*n
        def Dfs(Adjlist,UdAdjlist,v,n):
            global ans
            path.append(v)
            visited[v] = 1
            for i in UdAdjlist[v]:
                if visited[i] == 0:
                    visited[i] == 1
                    if v not in Adjlist[i]:
                        ans += 1
                    Dfs(Adjlist,UdAdjlist,i,n)

        global ans
        ans = 0
        Dfs(Adjlist,UdAdjlist,0,n)
        return ans
            
        

