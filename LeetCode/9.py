class Solution(object):
    def isPalindrome(self, x):
        """
        :type x: int
        :rtype: bool
        """
        if x < 0:
            return False
        i = 0
        while x%(10**i) != x:
            i = i+1
        for j in range(int(i/2)):
            if x%(10**(j+1))//(10**j) != x%(10**(i-j))//(10**(i-1-j)):
                return False
        return True

