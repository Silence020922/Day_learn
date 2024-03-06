# 很数学的一个题
class Solution(object):
    def maximumProduct(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # 需要记录三个最大值和两个最小值
        sort_num = sorted(nums)
        if sort_num[0]*sort_num[1]*sort_num[-1] > sort_num[-1]*sort_num[-2]*sort_num[-3]:
            return sort_num[0]*sort_num[1]*sort_num[-1]
        else: return sort_num[-1]*sort_num[-2]*sort_num[-3]

