#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <omp.h>
#include <cstdint>
#include <vector>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <iostream>

namespace py = pybind11;

// プレイヤー定数
static const int BLACK = 1;
static const int WHITE = 2;

// 8方向
static const int DIRECTION[8][2] = {
    {-1, 0}, {1, 0}, {0, -1}, {0, 1},
    {-1, -1}, {-1, 1}, {1, -1}, {1, 1}
};

// 盤面をビットボードで表すクラス
// blackBB, whiteBB はそれぞれ 64ビットの盤面情報
//  bit i (0 <= i < 64) が 1 のとき、i番目のマスに石がある
//  i番目のマスは row*8 + col とし、(row, col) = (0,0) が bit 0, (7,7) が bit 63 とする。
//  上から順番にインデックスを増やすイメージ。
struct OthelloBitBoard {
    uint64_t blackBB;
    uint64_t whiteBB;

    OthelloBitBoard() : blackBB(0ULL), whiteBB(0ULL) {}
};

// (row, col) をビットインデックスに変換
static inline int rc_to_bit(int row, int col) {
    return row * 8 + col;  // 0 <= row, col < 8
}

// (row, col) のビットを1にするためのマスク
static inline uint64_t mask_rc(int row, int col) {
    return (1ULL << rc_to_bit(row, col));
}

// 特定のビットが立っているかチェック
static inline bool is_occupied(uint64_t bb, int row, int col) {
    return (bb & mask_rc(row, col)) != 0ULL;
}

// そのマスに誰の石もないか？
static inline bool is_empty(const OthelloBitBoard &board, int row, int col) {
    uint64_t occ = board.blackBB | board.whiteBB;
    return (occ & mask_rc(row, col)) == 0ULL;
}

// 初期盤面を返す
//   中央に4つ石がおいてある
//   (3,3), (4,4) が BLACK, (3,4), (4,3) が WHITE (0-index)
OthelloBitBoard initial_board_cpp() {
    OthelloBitBoard board;
    // BLACK
    board.blackBB |= mask_rc(3,3);
    board.blackBB |= mask_rc(4,4);
    // WHITE
    board.whiteBB |= mask_rc(3,4);
    board.whiteBB |= mask_rc(4,3);
    return board;
}

// player が打ったときに裏返る石のビットマスクを返す
//  (row, col) に置くとき、返せる相手の石の集合をビットマスクで返す
uint64_t compute_flip(const OthelloBitBoard &board, int player, int row, int col) {
    // 相手番
    int opponent = (player == BLACK ? WHITE : BLACK);

    uint64_t selfBB   = (player == BLACK ? board.blackBB : board.whiteBB);
    uint64_t enemyBB  = (opponent == BLACK ? board.blackBB : board.whiteBB);

    // すでに石がある場合は無効
    if(!is_empty(board, row, col)) {
        return 0ULL;
    }

    uint64_t flips = 0ULL;

    // 8方向に対して調べる
    for(auto &dir : DIRECTION) {
        int r = row + dir[0];
        int c = col + dir[1];
        uint64_t mask = 0ULL;
        bool canFlip = false;

        // まず相手色が続くかどうかをチェック
        while(r >= 0 && r < 8 && c >= 0 && c < 8) {
            if(is_occupied(enemyBB, r, c)) {
                // 相手の石があるので裏返し候補に追加
                mask |= mask_rc(r, c);
            } else {
                // 自分の石があればそこまでの相手石を裏返し確定
                if(is_occupied(selfBB, r, c)) {
                    canFlip = true;
                }
                break;
            }
            r += dir[0];
            c += dir[1];
        }
        if(canFlip && mask != 0ULL) {
            flips |= mask;
        }
    }
    return flips;
}

// player が置けるかどうか(裏返しが発生するか)を判定する
bool can_put(const OthelloBitBoard &board, int player, int row, int col) {
    return compute_flip(board, player, row, col) != 0ULL;
}

// 盤面全体に対して、player の合法手をすべて取得
std::vector<std::pair<int,int>> get_valid_moves_cpp(const OthelloBitBoard &board, int player) {
    std::vector<std::pair<int,int>> moves;
    moves.reserve(64);

    // OpenMP 並列化 (単純な例)
    //   ただし、この程度のループでパフォーマンス向上が見込めるかは環境次第
    #pragma omp parallel
    {
        std::vector<std::pair<int,int>> local_moves;
        local_moves.reserve(64);

        #pragma omp for nowait
        for(int r = 0; r < 8; r++){
            for(int c = 0; c < 8; c++){
                if(can_put(board, player, r, c)) {
                    local_moves.emplace_back(r, c);
                }
            }
        }

        #pragma omp critical
        moves.insert(moves.end(), local_moves.begin(), local_moves.end());
    }

    return moves;
}

// 盤面と手番から、置ける場所を 8x8 の numpy.ndarray (int型) で返す
//  置ける場所は 1, 置けない場所は 0
py::array_t<int> get_valid_board_cpp(const OthelloBitBoard &board, int player) {
    // shape (8,8) の配列を作る
    auto result = py::array_t<int>({8, 8});
    auto buf = result.mutable_unchecked<2>();

    // 並列化(簡易)
    #pragma omp parallel for
    for(int r = 0; r < 8; r++){
        for(int c = 0; c < 8; c++){
            buf(r, c) = can_put(board, player, r, c) ? 1 : 0;
        }
    }
    return result;
}

// (row, col) に player が石を置く
//  戻り値は新しい盤面
OthelloBitBoard put_cpp(const OthelloBitBoard &board, int player, std::pair<int,int> move) {
    int row = move.first;
    int col = move.second;

    // 裏返し量を計算
    uint64_t flipMask = compute_flip(board, player, row, col);
    if(flipMask == 0ULL && !is_empty(board, row, col)){
        // 合法手でない or すでに空いていないならそのまま返す(実運用では例外でも可)
        return board;
    }

    OthelloBitBoard newBoard = board;

    // 石を置く
    uint64_t place = mask_rc(row, col);
    if(player == BLACK) {
        newBoard.blackBB |= place;
        // flip
        newBoard.blackBB |= flipMask;
        newBoard.whiteBB &= ~flipMask;
    } else {
        newBoard.whiteBB |= place;
        // flip
        newBoard.whiteBB |= flipMask;
        newBoard.blackBB &= ~flipMask;
    }
    return newBoard;
}

// 手番を変える
int change_turn_cpp(int player) {
    return (player == BLACK ? WHITE : BLACK);
}

// 棋譜情報を受け取り、盤面と手番を返す (サンプルではダミー実装)
std::pair<OthelloBitBoard,int> record_from_board_cpp(const std::string &record) {
    // ここでは単に初期盤面と BLACK を返すダミー
    OthelloBitBoard initB = initial_board_cpp();
    return std::make_pair(initB, BLACK);
}

// 石数を数える(bit count)
static inline int popcount64(uint64_t x) {
#if defined(__GNUC__)
    return __builtin_popcountll(x);
#else
    // C++20 なら std::popcount(x) が使える
    // ここでは簡易実装
    int c = 0;
    while(x){
        x &= (x - 1);
        c++;
    }
    return c;
#endif
}

// 盤面にある石の数を返す { 'black': x, 'white': y }
py::dict count_discs_cpp(const OthelloBitBoard &board) {
    int b = popcount64(board.blackBB);
    int w = popcount64(board.whiteBB);
    py::dict result;
    result["black"] = b;
    result["white"] = w;
    return result;
}

// 盤面 + 手番 から shape=(1,3,8,8) のデータを返す
//   チャンネル0: black石の位置(0/1), チャンネル1: white石(0/1),
//   チャンネル2: 現手番(全てを player==BLACKなら1, そうでなければ0 とする例)
py::array_t<float> proc_board_cpp(const OthelloBitBoard &board, int player) {
    // shape = (1, 3, 8, 8)
    auto result = py::array_t<float>({1, 3, 8, 8});
    auto buf = result.mutable_unchecked<4>();

    // 全て0で初期化
    for(int c = 0; c < 3; c++){
        for(int r = 0; r < 8; r++){
            for(int col = 0; col < 8; col++){
                buf(0, c, r, col) = 0.0f;
            }
        }
    }

    // black
    for(int r = 0; r < 8; r++){
        for(int c = 0; c < 8; c++){
            if(is_occupied(board.blackBB, r, c)){
                buf(0, 0, r, c) = 1.0f;
            }
        }
    }
    // white
    for(int r = 0; r < 8; r++){
        for(int c = 0; c < 8; c++){
            if(is_occupied(board.whiteBB, r, c)){
                buf(0, 1, r, c) = 1.0f;
            }
        }
    }
    // 現手番チャネル: player==BLACKなら1, そうでなければ0(全部埋める例)
    if(player == BLACK) {
        for(int r = 0; r < 8; r++){
            for(int c = 0; c < 8; c++){
                buf(0, 2, r, c) = 1.0f;
            }
        }
    }
    return result;
}

// ランダムプレイアウトを 1回行い、勝者(BLACK/WHITE) or 0(引分) を返す
//   currentPlayer からランダムにゲーム終了まで進める
//   (非常に単純な乱数で分散していない例)
static thread_local std::mt19937 rng(std::random_device{}());

int single_playout(const OthelloBitBoard &startBoard, int startPlayer) {
    // コピーしてランダムプレイ
    OthelloBitBoard board = startBoard;
    int player = startPlayer;
    int passCount = 0;

    while(true){
        // 合法手
        auto moves = get_valid_moves_cpp(board, player);
        if(moves.empty()) {
            passCount++;
            if(passCount >= 2) {
                // 連続パス => 終了
                break;
            }
            player = change_turn_cpp(player);
            continue;
        }
        passCount = 0;

        // ランダムに手を選ぶ
        std::uniform_int_distribution<int> dist(0, (int)moves.size()-1);
        auto mv = moves[dist(rng)];

        // 着手
        board = put_cpp(board, player, mv);

        // ターン交代
        player = change_turn_cpp(player);
    }
    // 結果判定
    int b = popcount64(board.blackBB);
    int w = popcount64(board.whiteBB);
    if(b > w) return BLACK;
    else if(b < w) return WHITE;
    else return 0; // draw
}

// シミュレーション回数だけランダムプレイして、
//  startPlayer が勝利する確率(勝率)を返す
double simulate_game_cpp(const OthelloBitBoard &board, int startPlayer, int num_sim) {
    if(num_sim <= 0) return 0.0;

    int winCount = 0;

    // OpenMP で並列化
    #pragma omp parallel for reduction(+:winCount)
    for(int i = 0; i < num_sim; i++){
        int result = single_playout(board, startPlayer);
        if(result == startPlayer) {
            winCount++;
        }
    }
    return double(winCount) / double(num_sim);
}

// Pybind11 モジュール定義
PYBIND11_MODULE(c_othello_bit, m) {
    m.doc() = "Pybind11 OthelloBit Board with OpenMP";

    // 定数を Python 側にエクスポート
    m.attr("BLACK") = BLACK;
    m.attr("WHITE") = WHITE;

    // 盤面クラス(OthelloBitBoard)をラップ（テストコード上は直接使わないため最小限）
    py::class_<OthelloBitBoard>(m, "OthelloBitBoard")
        .def(py::init<>())
        .def_readwrite("blackBB", &OthelloBitBoard::blackBB)
        .def_readwrite("whiteBB", &OthelloBitBoard::whiteBB)
        ;

    // 関数群
    m.def("initial_board", &initial_board_cpp, 
          "Return the initial 8x8 Othello board as a bitboard");
    m.def("get_valid_moves", &get_valid_moves_cpp, 
          "Return list of valid moves as (row,col) for given board & player");
    m.def("get_valid_board", &get_valid_board_cpp, 
          "Return an 8x8 ndarray (int) of valid moves");
    m.def("put", &put_cpp, 
          "Place a stone on given (row, col). Return new board");
    m.def("change_turn", &change_turn_cpp, 
          "Change turn (BLACK <-> WHITE)");
    m.def("record_from_board", &record_from_board_cpp,
          "Dummy function returning (initial_board, BLACK) for example");
    m.def("count_discs", &count_discs_cpp, 
          "Count discs and return dict with {'black': int, 'white': int}");
    m.def("proc_board", &proc_board_cpp,
          "Convert board+player into shape=(1,3,8,8) float ndarray");
    m.def("simulate_game", &simulate_game_cpp, 
          "Simulate random playouts and return winning rate for startPlayer");
}