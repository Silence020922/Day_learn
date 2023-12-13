class Solution(object):
    def minimumEffortPath(self, heights):
        """
        :type heights: List[List[int]]
        :rtype: int
        """
        height_list = []
        for height in heights:
            height_list.extend(height)
        n_row = len(heights)
        n_col = len(heights[0])
        total = n_row*n_col
        dir = [1,-1,n_col,-n_col] # 四个方向
        dist = [float('inf')]*total
        pq = [] # 优先栈
        dist[0] = 0
        pq.append((0,0)) # 第一个为dist
        view = [1]*total
        while pq:
            d, vtx = heapq.heappop(pq) #取最小
            view[vtx] = 0
            for i in dir:
                if ((vtx%n_col == 0 and i!= -1) or (vtx%n_col == n_col-1 and i!= 1) or vtx%n_col not in {0,n_col-1}) and 0<=vtx+i<total:
                    new_d = max(d ,abs(height_list[vtx] - height_list[i+vtx]))
                    if dist[i+vtx] > new_d:
                        dist[i+vtx] = new_d
                        heapq.heappush(pq,(new_d,vtx+i))

        return dist[total-1]


