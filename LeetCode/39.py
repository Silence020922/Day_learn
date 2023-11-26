class Solution(object):
    def combinationSum(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        l = len(candidates)
        set = []
        ans = []
        def dfs(startid,sum):
            if sum == target:
                ans.append(set[:])
            elif sum > target:
                return
            for j in range(startid,l):
                set.append(candidates[j])
                sum += candidates[j]
                dfs(j,sum)
                set.pop()
                sum -= candidates[j]
        dfs(0,0)
        return(ans)
            

