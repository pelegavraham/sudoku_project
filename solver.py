import math


def find_next_cell(grid, i, j):
    """
    find the next cell that we can solve
    :param grid:
    :param i:
    :param j:
    :return:
    """
    for x in range(i, 9):
        for y in range(j, 9):
            if grid[x][y] == 0:
                return x, y
    for x in range(0, 9):
        for y in range(0, 9):
            if grid[x][y] == 0:
                return x, y
    return -1, -1


def is_valid(grid, row, col, num):
    """
    check if can put num in the board
    :param row: row in board
    :param col: col in board
    :param num: num value
    :return: true if there is no same number in row and col and the square of num
    """
    # check row
    for i in range(9):
        if grid[row][i] == num and i != col:
            return False

    # check col
    trans_board = list(map(list, zip(*grid)))
    for j in range(9):
        if trans_board[col][j] == num and j != row:
            return False

    # check square
    start_row = math.floor(row / 3) * 3
    start_col = math.floor(col / 3) * 3
    for i in range(3):
        for j in range(3):
            if grid[start_row + i][start_col + j] == num and not (start_row + i == row and start_col + j == col):
                return False

    return True


def solve_sudoku(grid, i=0, j=0):
    """
    solve the sudoku grid
    :param grid:
    :param i:
    :param j:
    :return:
    """
    i, j = find_next_cell(grid, i, j)
    if i == -1:
        return True
    for e in range(1, 10):
        if is_valid(grid, i, j, e):
            grid[i][j] = e
            if solve_sudoku(grid, i, j):
                return True
            # if not work return the cell to be empty
            grid[i][j] = 0
    return False

