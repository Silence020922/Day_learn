class Solution(object):
    def setZeroes(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: None Do not return anything, modify matrix in-place instead.
        """
        row = len(matrix)
        col = len(matrix[0])
        col_flag = False
        row_flag = False
        for i in range(row):
            if matrix[i][0] == 0:
                col_flag = True
                break
        for j in range(col):
            if matrix[0][j] == 0:
                row_flag = True
                break
        # 把第一行或者第一列作为 标志位
        for i in range(1, row):
            for j in range(1, col):
                if matrix[i][j] == 0:
                    matrix[i][0] = matrix[0][j] = 0

        # 置0
        for i in range(1, row):
            for j in range(1, col):
                if matrix[i][0] == 0 or matrix[0][j] == 0:
                    matrix[i][j] = 0

        if col_flag:
            for i in range(row):
                matrix[i][0] = 0
        if row_flag:
            matrix[0] = [0]*col
