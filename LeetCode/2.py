# Definition for singly-linked list.
# class ListNode(object):.
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        ans= rln = ListNode()
        b = 0       
        while l1 or l2: 
            if not l1:
                l1 = ListNode(0)
            if not l2:
                l2 = ListNode(0)
            g = (l1.val + l2.val + b)%10
            b = (l1.val + l2.val + b)//10
            ans.next = ListNode(val = g)
            ans = ans.next
            l1 = l1.next
            l2 = l2.next
            if b != 0:
                ans.next = ListNode(val = 1)
        return rln.next
                
            
            
            
            
            
        

