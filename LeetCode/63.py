class Solution(object):
    def uniquePathsWithObstacles(self, obstacleGrid):
        """
        :type obstacleGrid: List[List[int]]
        :rtype: int
        """
        n = len(obstacleGrid) # 行
        m = len(obstacleGrid[0]) # 列
        dp = [[0] * m for _ in range(n)] # m*n 的矩阵 记录到达该点的路的数目
        # (0,0)这个格子可能有障碍物
        if obstacleGrid[0][0] == 1: return 0
        dp[0][0] = 1

        # 处理第一列
        for i in range(1, n):
            if obstacleGrid[i][0] == 1 or dp[i - 1][0] == 0: # 一旦第i个元素为1有障碍，那么其下的>i的元素都无法到达
                dp[i][0] = 0

            else:
                dp[i][0] = 1

        # 处理第一行
        for j in range(1, m):
            if obstacleGrid[0][j] == 1 or dp[0][j - 1] == 0:
                dp[0][j] = 0

            else:
                dp[0][j] = 1

        for i in range(1, n):
            for j in range(1, m):
                # 如果当前格子是障碍物
                if obstacleGrid[i][j] == 1:
                    dp[i][j] = 0

                # 路径总数来自于上方(dp[i-1][j])和左方(dp[i][j-1])
                else:
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1]

        return dp[-1][-1]



