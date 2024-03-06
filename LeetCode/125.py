# 对ord()函数现学现卖，代码质量过关
class Solution(object):
    def isPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        l = len(s)
        i = 0
        j = -1
        while i-j < l:
            a = ord(s[i])
            b = ord(s[j])
            if  47<a<58 or 64<a<91 or 96<a<123:
                if 47<b<58 or 64<b<91 or 96<b<123:
                    if a == b or (a>57 and b>57 and (a-b)%32 == 0):
                        i +=1
                        j -= 1
                    else:
                        return False
                else: j -=1

            else: i+=1
        return True
