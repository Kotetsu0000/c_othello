import random
import time

import numpy as np
from c_othello.c_othello_bit import (
    initial_board,      # 初期盤面を返す
    get_valid_moves,    # 盤面と手番から合法手を返す(リスト)
    get_valid_board,    # 盤面と手番から合法手を返す(numpy.ndarray shape=(8, 8))
    put,                # 盤面と手番と手を受け取り、石を置いた盤面を返す
    change_turn,        # 手番を受け取り、相手番を返す
    record_from_board,  # 棋譜情報を受け取り、盤面と手番を返す
    count_discs,        # 盤面を受け取り、石の数を返す
    proc_board,         # 盤面と手番を受け取り、入力データを返す(numpy.ndarray shape=(1, 3, 8, 8)→(batch, channel, height, width))
    BLACK,              # 黒石番(int) 
    WHITE,              # 白石番(int)
    simulate_game,      # 盤面と手番とシミュレーション回数を受け取り、勝率を返す
)

def get_move_prob(board, player, move):
    new_board = put(board, player, move)
    return simulate_game(new_board, player, 10000)

def random_game_():
    boards = []
    policies = []
    board = initial_board()
    player = BLACK
    turn = 0
    while True:
        turn += 1
        # 合法手の取得
        moves = get_valid_moves(board, player)

        simulate_num = 1000
        win = simulate_game(board, player, simulate_num) if player==BLACK else 1-simulate_game(board, player, simulate_num)
        print(f'Turn{turn}: {win*100:.2f}%')

        # 置ける場所がない場合
        if len(moves)==0:
            # 相手番が置けるかどうかの確認
            player = change_turn(player)
            moves = get_valid_moves(board, player)
            # 相手番も置けない場合
            if len(moves)==0:
                break # ゲーム終了
            else:
                continue # 相手番が置ける場合は相手番に交代

        if player == WHITE:
            # ランダムに手を選択
            move = random.choice(moves)
        else:
            prob_list = [get_move_prob(board, player, move) for move in moves]
            max_index = np.argmax(prob_list)
            move = moves[max_index]

        # 石を置く
        board = put(board, player, move)

        # 盤面の保存
        boards.append(proc_board(board, player))

        # 手番の交代
        player = change_turn(player)

    # 勝敗の判定
    count = count_discs(board)
    if count['black']>count['white']:
        print('black win')
    elif count['black']<count['white']:
        print('white win')
    else:
        print('draw')

def random_game_bit():
    board = initial_board()
    player = BLACK
    turn = 0
    while True:
        turn += 1
        # 合法手の取得
        moves = get_valid_moves(board, player)

        # 置ける場所がない場合
        if len(moves)==0:
            # 相手番が置けるかどうかの確認
            player = change_turn(player)
            moves = get_valid_moves(board, player)
            # 相手番も置けない場合
            if len(moves)==0:
                break # ゲーム終了
            else:
                continue # 相手番が置ける場合は相手番に交代

        # ランダムに手を選択
        move = random.choice(moves)

        # 石を置く
        board = put(board, player, move)

        # 手番の交代
        player = change_turn(player)

    # 勝敗の判定
    count = count_discs(board)

if __name__ == '__main__':
    start = time.perf_counter()
    for _ in range(10000):random_game_bit()
    print(f'Bit time: {time.perf_counter()-start:.2f}s')
    start = time.perf_counter()
    simulate_game(initial_board(), BLACK, 10000)
    print(f'Bit time: {time.perf_counter()-start:.2f}s')
