class Solution(object):
    def findMedianSortedArrays(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: float
        """
        l1 = len(nums1)
        l2 = len(nums2)
        i = 0
        if (l1+l2)%2 == 0:
            mid_num = (l1+l2)/2
            if l1 == 0:return (float(nums2[mid_num-1]) + nums2[mid_num])/2
            if l2 == 0:return (float(nums1[mid_num-1]) + nums1[mid_num])/2
            while i < mid_num:
                i += 1
                if nums1[-1] > nums2[-1]:
                    mid = nums1.pop()
                    if len(nums1) ==0:
                        if i == mid_num:
                            return (float(mid) + nums2[-1])/2
                        else:
                            return (float(nums2[len(nums2)-mid_num + i]) + nums2[len(nums2) - mid_num + i-1])/2
                else:
                    mid = nums2.pop()
                    if len(nums2) ==0:
                        if i == mid_num:
                            return (float(mid) + nums1[-1])/2
                        else:
                            return (float(nums1[len(nums1)-mid_num + i]) + nums1[len(nums1) - mid_num + i-1])/2
            return (float(mid) + max(nums1[-1],nums2[-1]))/2
        else:
            mid_num = (l1+l2+1)/2
            if l1 == 0:return nums2[mid_num-1]
            if l2 == 0:return nums1[mid_num-1]
            while i < mid_num - 1:
                i+= 1
                if nums1[-1] > nums2[-1]:
                    mid = nums1.pop()
                    if len(nums1) == 0:return nums2[len(nums2)-mid_num + i]
                else:
                    mid = nums2.pop()
                    if len(nums2) ==0:return nums1[len(nums1) - mid_num + i]
            return max(nums1[-1],nums2[-1])
                

