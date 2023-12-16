class Solution(object):
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        a = 1
        b = 2
        if n == 1 or n==2 : return n
        for i in range(n-2):
            c = a+b
            a = b
            b = c
        return c

