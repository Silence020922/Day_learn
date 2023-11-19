class Solution(object):
    def isValidSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: bool
        """
        # 位运算
        col_list= [0]*9
        coll_list = [0]*9
        box_list = [0]*9
        for i in range(9):
            for j in range(9):
                if board[i][j] == '.':continue
                num = 1 << int(board[i][j])
                box_n = (i//3)*3 + j//3
                if num & col_list[i]or num & coll_list[j] or num & box_list[box_n]:
                    return False
                col_list[i] |= num
                coll_list[j] |= num
                box_list[box_n] |= num
        return True             


