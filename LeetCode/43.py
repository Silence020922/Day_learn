# 做的很烂，非常基础的做法
class Solution(object):
    def multiply(self, num1, num2):
        """
        :type num1: str
        :type num2: str
        :rtype: str
        """
        if num1 == '0' or num2 == '0': return '0'
        def mul_str_1(a,b):
            """
            a 为str
            b 为单个字符
            """ 
            ans = str()
            b = int(b) # 转换整形
            l = len(a)
            tmp = 0 # 进位字符
            for i in range(l):
                mul = int(a[-(i+1)])*b + tmp
                tmp = mul // 10
                nloc = mul % 10
                ans = str(nloc) + ans
            if tmp > 0:
                ans = str(tmp) + ans
            return ans

        def str_add(a,b): # 字符加和操作
            """
            a 为str
            b 为str
            """
            l1 = len(a)
            l2 = len(b)
            if l1 - l2 >0:
                b = '0'*(l1-l2) + b
                l = l1
            else:
                a = '0'*(l2-l1) + a
                l = l2
            tmp = 0 # 保存进位操作
            ans = ""
            for i in range(l):
                add = int(a[-(i+1)]) + int(b[-(i+1)]) + tmp
                tmp = add // 10
                nloc = add % 10
                ans = str(nloc) + ans
            if tmp > 0:
                return "1" + ans
            return ans
        l = len(num2)
        ans = str(0)
        for i in range(l):
            tmp = mul_str_1(num1,num2[-(i+1)]) + '0'*i
            ans = str_add(ans,tmp)
        return ans

# 转化为ASCII字符，过于巧妙
class Solution(object):
    def multiply(self, num1, num2):
        """
        :type num1: str
        :type num2: str
        :rtype: str
        """
        a, ans = 0, 0
        for n in num1:
            a = a*10+ord(n)-ord('0')
        for n in num2:
            ans = ans*10+a*(ord(n)-ord('0'))
        return str(ans)
