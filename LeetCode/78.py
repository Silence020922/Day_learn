class Solution(object):
    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        ans = []
        path = []
        n = len(nums)
        def dfs(startid):
            ans.append(path[:])
            for i in range(startid,n):
                path.append(nums[i])
                dfs(i+1)
                path.pop()
        dfs(0)
        return ans
                

