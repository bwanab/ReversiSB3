/*
 * Exact Othello endgame solver: negamax with alpha-beta over bitboards.
 *
 * Boards are two 64-bit masks from the side to move's view: P (side to move) and O (opponent).
 * Bit i is square i = row * 8 + col, the same indexing as the Python code (util/search.py).
 * Scores are final disc differences for the side to move with perfect play by both sides,
 * empty squares going to the winner (the official rule, and Edax's convention).
 *
 * Build: solver/build.sh (produces solver/libendgame.dylib or .so); Python wrapper: util/endgame.py.
 */

#include <stdint.h>

typedef uint64_t u64;

static long long g_nodes;

static int popcount(u64 x) { return __builtin_popcountll(x); }

/* shift a bitboard one step in direction d (0..7), dropping squares that leave the board */
static const int DIR_SHIFT[8] = {1, -1, 8, -8, 9, -9, 7, -7};   /* E W S N SE NW SW NE */
static const u64 NOT_A = 0xFEFEFEFEFEFEFEFEULL;                /* clears column 0 */
static const u64 NOT_H = 0x7F7F7F7F7F7F7F7FULL;                /* clears column 7 */

static inline u64 shift(u64 x, int d) {
    switch (d) {
        case 0: return (x << 1) & NOT_A;   /* E: col+1 */
        case 1: return (x >> 1) & NOT_H;   /* W: col-1 */
        case 2: return x << 8;             /* S: row+1 */
        case 3: return x >> 8;             /* N: row-1 */
        case 4: return (x << 9) & NOT_A;   /* SE */
        case 5: return (x >> 9) & NOT_H;   /* NW */
        case 6: return (x << 7) & NOT_H;   /* SW */
        default: return (x >> 7) & NOT_A;  /* NE */
    }
}

/* legal moves of P against O */
u64 eg_moves(u64 P, u64 O) {
    u64 empty = ~(P | O), moves = 0;
    for (int d = 0; d < 8; d++) {
        u64 t = shift(P, d) & O;
        for (int i = 0; i < 5; i++) t |= shift(t, d) & O;
        moves |= shift(t, d) & empty;
    }
    return moves;
}

/* discs flipped when P plays on square sq */
u64 eg_flips(u64 P, u64 O, int sq) {
    u64 m = 1ULL << sq, flips = 0;
    for (int d = 0; d < 8; d++) {
        u64 run = 0, x = shift(m, d);
        while (x & O) { run |= x; x = shift(x, d); }
        if (x & P) flips |= run;
    }
    return flips;
}

static int final_score(u64 P, u64 O) {
    int p = popcount(P), o = popcount(O), e = 64 - p - o;
    if (p > o) return p - o + e;
    if (o > p) return p - o - e;
    return 0;
}

static int solve(u64 P, u64 O, int alpha, int beta, int passed);

/* search the moves in `moves`, ordered by the opponent's resulting mobility when many empties remain */
static int search_moves(u64 P, u64 O, u64 moves, int alpha, int beta, int *best_move) {
    int sq_list[32], key[32], n = 0;
    int empties = 64 - popcount(P | O);
    while (moves) {
        int sq = __builtin_ctzll(moves);
        moves &= moves - 1;
        sq_list[n] = sq;
        if (empties > 6) {
            u64 f = eg_flips(P, O, sq);
            u64 np = P | f | (1ULL << sq), no = O & ~f;
            int k = popcount(eg_moves(no, np)) * 4;                    /* fewer replies first */
            if ((1ULL << sq) & 0x8100000000000081ULL) k -= 8;          /* corners first */
            key[n] = k;
        } else key[n] = 0;
        n++;
    }
    for (int i = 1; i < n; i++) {                                     /* insertion sort by key */
        int s = sq_list[i], k = key[i], j = i - 1;
        while (j >= 0 && key[j] > k) { sq_list[j + 1] = sq_list[j]; key[j + 1] = key[j]; j--; }
        sq_list[j + 1] = s; key[j + 1] = k;
    }
    int best = -65;
    for (int i = 0; i < n; i++) {
        int sq = sq_list[i];
        u64 f = eg_flips(P, O, sq);
        int v = -solve(O & ~f, P | f | (1ULL << sq), -beta, -alpha, 0);
        if (v > best) {
            best = v;
            if (best_move) *best_move = sq;
            if (v > alpha) alpha = v;
            if (alpha >= beta) break;
        }
    }
    return best;
}

static int solve(u64 P, u64 O, int alpha, int beta, int passed) {
    g_nodes++;
    u64 moves = eg_moves(P, O);
    if (!moves) {
        if (passed || !eg_moves(O, P)) return final_score(P, O);
        return -solve(O, P, -beta, -alpha, 1);
    }
    return search_moves(P, O, moves, alpha, beta, 0);
}

/*
 * Exact score of the position for the side to move, and its best move (-1 if it has to pass or the
 * game is over). Search window [alpha, beta]: with the full window (-64, 64) the score is exact;
 * with a narrower one the result is exact inside it and a bound outside it.
 */
int eg_solve(u64 P, u64 O, int alpha, int beta, int *best_move, long long *nodes) {
    g_nodes = 0;
    *best_move = -1;
    int score;
    u64 moves = eg_moves(P, O);
    if (!moves) score = solve(P, O, alpha, beta, 0);
    else score = search_moves(P, O, moves, alpha, beta, best_move);
    if (nodes) *nodes = g_nodes;
    return score;
}
