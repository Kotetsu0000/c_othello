import numpy as np
cimport numpy as cnp
cimport cython
from cython.parallel import prange
from libcpp.vector cimport vector
from libcpp.string cimport string
from cpython cimport PyUnicode_FromString

# 定数定義
cdef int BLACK = 1
cdef int WHITE = 2
cdef int NONE = 0

# DTYPEの定義
DTYPE = np.int8
ctypedef cnp.int8_t DTYPE_t

cdef dict Y_TO_NUM = {chr(i): i - ord('A') for i in range(ord('A'), ord('I'))}

# 初期盤面を生成する関数
cdef DTYPE_t[:, :] _initial_board() nogil:
    cdef DTYPE_t[:, :] board
    with gil:
        board = np.zeros((8, 8), dtype=DTYPE)
    board[3, 3] = WHITE
    board[4, 4] = WHITE
    board[3, 4] = BLACK
    board[4, 3] = BLACK
    return board
###

cpdef cnp.ndarray[DTYPE_t, ndim=2] initial_board():
    return np.asarray(_initial_board())
###

# 盤面の入力から、どこに置けるかを返却する関数
cdef DTYPE_t[:, :] _get_valid_board(DTYPE_t[:, :] board, int color) nogil:
    cdef DTYPE_t[:, :] valid_moves
    cdef int x, y, dx, dy
    cdef int opponent_color = 3 - color
    cdef int nx, ny, flipped
    with gil:
        valid_moves = np.zeros((8, 8), dtype=DTYPE)

    for y in range(8):
        for x in range(8):
            if board[y, x] != NONE:
                continue

            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if dx == 0 and dy == 0:
                        continue

                    flipped = 0
                    nx = x + dx
                    ny = y + dy

                    while 0 <= nx < 8 and 0 <= ny < 8 and board[ny, nx] == opponent_color:
                        flipped += 1
                        nx += dx
                        ny += dy

                    if flipped > 0 and 0 <= nx < 8 and 0 <= ny < 8 and board[ny, nx] == color:
                        valid_moves[y, x] = 1
    return valid_moves
###

cpdef cnp.ndarray[DTYPE_t, ndim=2] get_valid_board(DTYPE_t[:, :] board, int color):
    return np.asarray(_get_valid_board(board, color))
###

# 置ける場所を返却する関数
cdef vector[string] _get_valid_moves(DTYPE_t[:, :] board, int color):
    cdef DTYPE_t[:, :] valid_board = _get_valid_board(board, color)
    cdef vector[string] valid_list
    cdef int x, y
    for y in prange(8, nogil=True):
        for x in range(8):
            if valid_board[y, x] == 1:
                with gil:
                    valid_list.push_back((chr(y + ord('A')) + str(x)).encode(encoding='utf-8'))
    return valid_list
###

cpdef list get_valid_moves(DTYPE_t[:, :] board, int color):
    cdef vector[string] valid_list = _get_valid_moves(board, color)
    cdef list ret = [PyUnicode_FromString(s.c_str()) for s in valid_list]
    return ret
###

# 置いた時の石をひっくり返す処理
cdef DTYPE_t[:, :] _put(DTYPE_t[:, :] board, int color, str position) nogil:
    cdef int x, y, dx, dy, nx, ny
    cdef int opponent_color = change_turn(color)
    cdef int flipped
    with gil:
        x = int(position[1])
        y = Y_TO_NUM[position[0]]

    if not (0 <= x < 8 and 0 <= y < 8 and board[y,x] == NONE):
        return board

    board[y, x] = color

    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if dx == 0 and dy == 0:
                continue

            nx = x + dx
            ny = y + dy
            flipped = 0

            while 0 <= nx < 8 and 0 <= ny < 8 and board[ny, nx] == opponent_color:
                flipped += 1
                nx += dx
                ny += dy

            if flipped > 0 and 0 <= nx < 8 and 0 <= ny < 8 and board[ny, nx] == color:
                nx = x + dx
                ny = y + dy
                while 0 <= nx < 8 and 0 <= ny < 8 and board[ny, nx] == opponent_color:
                    board[ny, nx] = color
                    nx += dx
                    ny += dy

    return board
###

cpdef cnp.ndarray[DTYPE_t, ndim=2] put(DTYPE_t[:, :] board, int color, str position):
    if position not in get_valid_moves(board, color):
        raise ValueError('Invalid move')
    return np.asarray(_put(board, color, position))
###

# 手番を入れ替える関数
cpdef int change_turn(int color) nogil:
    return 3 - color
###

# 棋譜から盤面を生成する関数
cdef DTYPE_t[:, :] _record_from_board(str record):
    cdef DTYPE_t[:, :] board = _initial_board()
    cdef int color = BLACK
    cdef str position
    while len(record) > 0:
        if np.sum(get_valid_board(board, color)) == 0:
            color = change_turn(color)
            continue
        position = record[:2]
        record = record[2:]
        board = _put(board, color, position)
        color = change_turn(color)
    return board
###

cpdef cnp.ndarray[DTYPE_t, ndim=2] record_from_board(str record):
    return np.asarray(_record_from_board(record))
###

# 盤面の白黒の数を返却する関数
cpdef dict count_discs(DTYPE_t[:, :] board):
    cdef int black = 0
    cdef int white = 0
    cdef int x, y
    for y in prange(8, nogil=True):
        for x in range(8):
            if board[y, x] == BLACK:
                black += 1
            elif board[y, x] == WHITE:
                white += 1
    return {'black': black, 'white': white}
###
