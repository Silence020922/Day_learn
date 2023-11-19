class Solution(object):
    def reverse(self, x):
        """
        :type x: int
        :rtype: int
        """
        ans = 0
        abx = abs(x)
        while abx >= 10:
            tmp = abx % 10
            abx = abx/10
            ans = ans *10+tmp
        ans = ans*10+int(abx)
        if x > 0 and ans <= 2**31 -1:
            return ans
        elif x < 0 and -ans >= -2**31:
            return -ans
        return 0

