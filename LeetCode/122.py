class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        inlist = []
        outlist = []
        ans =0
        isin = True
        isout = False
        day = len(prices)
        if day <= 1:
            return ans
        for i in range(day-1):
            if prices[i] < prices[i+1] and isin:
                inlist.append(i)
                isin=False
                isout=True
            elif prices[i] > prices[i+1] and isout:
                outlist.append(i)
                isin = True
                isout =False
        if isout:
            outlist.append(day-1)
        if len(inlist) == len(outlist) and len(inlist) > 0:
            for i in range(len(inlist)):
                ans += prices[outlist[i]] - prices[inlist[i]]
        return ans       

