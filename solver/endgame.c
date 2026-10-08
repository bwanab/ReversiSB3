/*
 * Exact Othello endgame solver: negamax with alpha-beta over bitboards.
 *
 * Boards are two 64-bit masks from the side to move's view: P (side to move) and O (opponent).
 * Bit i is square i = row * 8 + col, the same indexing as the Python code (util/search.py).
 * Scores are final disc differences for the side to move with perfect play by both sides,
 * empty squares going to the winner (the official rule, and Edax's convention).
 *
 * Speedups (2026-10-08, see CLAUDE.md step 27): a per-thread transposition table (bounds and best
 * move, kept across solves in the same thread), principal-variation search (null windows after the
 * first move), and for the last SHALLOW empties a search over the empty squares in parity order
 * (squares in regions with an odd number of empties first) with no move generation.
 *
 * Build: solver/build.sh (produces solver/libendgame.dylib or .so); Python wrapper: util/endgame.py.
 */

#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>

typedef uint64_t u64;

#ifndef ETC_MIN_EMPTIES
#define ETC_MIN_EMPTIES 8  /* enhanced transposition cutoff at or above this many empties */
#endif
#ifndef STAB_ALPHA
#define STAB_ALPHA 16      /* try the stability cutoff when alpha is at least this */
#endif
#ifndef SHALLOW
#define SHALLOW 6          /* empties at or below which solve_shallow takes over */
#endif
#ifndef TT_MIN_EMPTIES
#define TT_MIN_EMPTIES 7   /* positions with at least this many empties use the table */
#endif

#define TT_BITS 19         /* 2^19 entries of 24 bytes = 12 MB per thread */

static __thread long long g_nodes;

static int popcount(u64 x) { return __builtin_popcountll(x); }

/* one direction's contribution to the legal moves: runs of O starting next to P, ending on an empty */
#define MOVES_DIR(SH, MASK) do { \
        u64 m = O & (MASK), t = m & SH(P); \
        t |= m & SH(t); t |= m & SH(t); t |= m & SH(t); t |= m & SH(t); t |= m & SH(t); \
        moves |= SH(t); } while (0)
#define L1(x) ((x) << 1)
#define R1(x) ((x) >> 1)
#define L8(x) ((x) << 8)
#define R8(x) ((x) >> 8)
#define L9(x) ((x) << 9)
#define R9(x) ((x) >> 9)
#define L7(x) ((x) << 7)
#define R7(x) ((x) >> 7)

/* legal moves of P against O */
u64 eg_moves(u64 P, u64 O) {
    u64 moves = 0;
    const u64 H = 0x7E7E7E7E7E7E7E7EULL, V = 0x00FFFFFFFFFFFF00ULL, D = 0x007E7E7E7E7E7E00ULL;
    MOVES_DIR(L1, H); MOVES_DIR(R1, H); MOVES_DIR(L8, V); MOVES_DIR(R8, V);
    MOVES_DIR(L9, D); MOVES_DIR(R9, D); MOVES_DIR(L7, D); MOVES_DIR(R7, D);
    return moves & ~(P | O);
}

/* one direction's flips: the run of O from the move, kept if it ends on P */
#define FLIPS_DIR(SH, MASK) do { \
        u64 m = O & (MASK), t = m & SH(b); \
        t |= m & SH(t); t |= m & SH(t); t |= m & SH(t); t |= m & SH(t); t |= m & SH(t); \
        if (SH(t) & P) flips |= t; } while (0)

/* discs flipped when P plays on square sq */
u64 eg_flips(u64 P, u64 O, int sq) {
    u64 b = 1ULL << sq, flips = 0;
    const u64 H = 0x7E7E7E7E7E7E7E7EULL, V = 0x00FFFFFFFFFFFF00ULL, D = 0x007E7E7E7E7E7E00ULL;
    FLIPS_DIR(L1, H); FLIPS_DIR(R1, H); FLIPS_DIR(L8, V); FLIPS_DIR(R8, V);
    FLIPS_DIR(L9, D); FLIPS_DIR(R9, D); FLIPS_DIR(L7, D); FLIPS_DIR(R7, D);
    return flips;
}

static int final_score(u64 P, u64 O) {
    int p = popcount(P), o = popcount(O), e = 64 - p - o;
    if (p > o) return p - o + e;
    if (o > p) return p - o - e;
    return 0;
}

/* ---- transposition table: one per thread, freed when the thread exits ---- */

typedef struct { u64 P, O; int8_t lower, upper, move, pad; } tt_entry;   /* P = O = 0: empty slot */

static pthread_key_t tt_key;
static pthread_once_t tt_once = PTHREAD_ONCE_INIT;
static __thread tt_entry *tt;

static void tt_make_key(void) { pthread_key_create(&tt_key, free); }

static tt_entry *tt_slot(u64 P, u64 O) {
    if (!tt) {
        pthread_once(&tt_once, tt_make_key);
        tt = calloc((size_t)1 << TT_BITS, sizeof(tt_entry));
        if (!tt) return 0;
        pthread_setspecific(tt_key, tt);
    }
    u64 h = P * 0x9E3779B97F4A7C15ULL ^ (O + 0x632BE59BD9B4E019ULL) * 0xC2B2AE3D27D4EB4FULL;
    return &tt[(h ^ (h >> 29)) & (((u64)1 << TT_BITS) - 1)];
}

/* ---- last few empties: iterate the empty squares in parity order, no move generation ---- */

static const u64 QUADRANT[4] = {0x000000000F0F0F0FULL, 0x00000000F0F0F0F0ULL,
                                0x0F0F0F0F00000000ULL, 0xF0F0F0F000000000ULL};

static int solve_shallow(u64 P, u64 O, int alpha, int beta, int passed) {
    g_nodes++;
    u64 empty = ~(P | O);
    if (!empty) return final_score(P, O);
    if (!(empty & (empty - 1))) {                                 /* one empty square left */
        int sq = __builtin_ctzll(empty);
        u64 b = 1ULL << sq, f = eg_flips(P, O, sq);
        if (f) return final_score(P | f | b, O & ~f);
        f = eg_flips(O, P, sq);
        if (f) return final_score(P & ~f, O | f | b);
        return final_score(P, O);
    }
    u64 odd = 0;
    for (int q = 0; q < 4; q++)
        if (popcount(empty & QUADRANT[q]) & 1) odd |= QUADRANT[q];
    int best = -65, any = 0;
    for (int round = 0; round < 2; round++) {
        u64 set = empty & (round == 0 ? odd : ~odd);
        while (set) {
            int sq = __builtin_ctzll(set);
            set &= set - 1;
            u64 f = eg_flips(P, O, sq);
            if (!f) continue;
            any = 1;
            int v = -solve_shallow(O & ~f, P | f | (1ULL << sq), -beta, -alpha, 0);
            if (v > best) {
                best = v;
                if (v > alpha) { alpha = v; if (alpha >= beta) return best; }
            }
        }
    }
    if (!any) {
        if (passed) return final_score(P, O);
        return -solve_shallow(O, P, -beta, -alpha, 1);
    }
    return best;
}

static int solve(u64 P, u64 O, int alpha, int beta, int passed);

/* ---- stability: discs that can never be flipped bound the final score ---- */

static u64 LINES[4][15];   /* rows, columns, diagonals, anti-diagonals (unused slots 0) */
static pthread_once_t lines_once = PTHREAD_ONCE_INIT;

static void make_lines(void) {
    for (int i = 0; i < 64; i++) {
        int r = i / 8, c = i % 8;
        LINES[0][r] |= 1ULL << i;
        LINES[1][c] |= 1ULL << i;
        LINES[2][r - c + 7] |= 1ULL << i;
        LINES[3][r + c] |= 1ULL << i;
    }
}

/* squares whose line along each axis is completely filled */
static void full_lines(u64 occ, u64 full[4]) {
    for (int a = 0; a < 4; a++) {
        full[a] = 0;
        for (int l = 0; l < 15; l++)
            if (LINES[a][l] && (occ & LINES[a][l]) == LINES[a][l]) full[a] |= LINES[a][l];
    }
}

/* O's stable discs: along every axis the line is full, the disc is on the border, or a neighbour on
 * that axis is a stable O disc (it could only be flipped together with that neighbour) */
static u64 stable_discs(u64 P, u64 O) {
    pthread_once(&lines_once, make_lines);
    const u64 COLS = 0x8181818181818181ULL, ROWS = 0xFF000000000000FFULL, BORDER = COLS | ROWS;
    u64 full[4], stable = 0;
    full_lines(P | O, full);
    for (;;) {
        u64 h = full[0] | COLS | ((stable << 1) & 0xFEFEFEFEFEFEFEFEULL) | ((stable >> 1) & 0x7F7F7F7F7F7F7F7FULL);
        u64 v = full[1] | ROWS | (stable << 8) | (stable >> 8);
        u64 d = full[2] | BORDER | ((stable << 9) & 0xFEFEFEFEFEFEFEFEULL) | ((stable >> 9) & 0x7F7F7F7F7F7F7F7FULL);
        u64 a = full[3] | BORDER | ((stable << 7) & 0x7F7F7F7F7F7F7F7FULL) | ((stable >> 7) & 0xFEFEFEFEFEFEFEFEULL);
        u64 next = O & h & v & d & a;
        if (next == stable) return stable;
        stable = next;
    }
}

/* search the moves in `moves`: the table's move first, then by the opponent's resulting mobility
 * (fewest replies first, corners preferred); the first move with the full window, the rest with a null
 * window and a re-search when one beats alpha (principal-variation search) */
static int search_moves(u64 P, u64 O, u64 moves, int alpha, int beta, int *best_move, int tt_move) {
    int sq_list[32], key[32], n = 0;
    while (moves) {
        int sq = __builtin_ctzll(moves);
        moves &= moves - 1;
        sq_list[n] = sq;
        if (sq == tt_move) key[n] = -1000;
        else {
            u64 f = eg_flips(P, O, sq);
            u64 np = P | f | (1ULL << sq), no = O & ~f;
            int k = popcount(eg_moves(no, np)) * 4;                    /* fewer replies first */
            if ((1ULL << sq) & 0x8100000000000081ULL) k -= 8;          /* corners first */
            key[n] = k;
        }
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
        u64 np = O & ~f, no = P | f | (1ULL << sq);
        int v;
        if (i == 0) v = -solve(np, no, -beta, -alpha, 0);
        else {
            v = -solve(np, no, -alpha - 1, -alpha, 0);
            if (v > alpha && v < beta) v = -solve(np, no, -beta, -alpha, 0);
        }
        if (v > best) {
            best = v;
            if (best_move) *best_move = sq;
            if (v > alpha) alpha = v;
            if (alpha >= beta) break;
        }
    }
    return best;
}

/* search with the table: probe (cutoffs, narrowed window, move ordering), search, store bounds */
static int search_tt(u64 P, u64 O, u64 moves, int alpha, int beta, int *best_move, int probe_cutoff) {
    tt_entry *e = 64 - popcount(P | O) >= TT_MIN_EMPTIES ? tt_slot(P, O) : 0;
    int tt_move = -1;
    if (e && e->P == P && e->O == O) {
        if (probe_cutoff) {
            if (e->lower >= beta) return e->lower;
            if (e->upper <= alpha) return e->upper;
            if (e->lower == e->upper) return e->lower;
            if (e->lower > alpha) alpha = e->lower;
            if (e->upper < beta) beta = e->upper;
        }
        tt_move = e->move;
    }
    if (probe_cutoff && 64 - popcount(P | O) >= ETC_MIN_EMPTIES) {
        /* enhanced transposition cutoff: a move whose result the table already bounds >= beta */
        for (u64 ms = moves; ms; ms &= ms - 1) {
            int sq = __builtin_ctzll(ms);
            u64 f = eg_flips(P, O, sq), cp = O & ~f, co = P | f | (1ULL << sq);
            tt_entry *c = tt_slot(cp, co);
            if (c && c->P == cp && c->O == co && -c->upper >= beta) {
                if (best_move) *best_move = sq;
                return -c->upper;
            }
        }
    }
    int a0 = alpha, move = -1;
    int best = search_moves(P, O, moves, alpha, beta, &move, tt_move);
    if (best_move) *best_move = move;
    if (e) {
        int lower = -64, upper = 64;
        if (best <= a0) upper = best;
        else if (best >= beta) lower = best;
        else lower = upper = best;
        if (e->P == P && e->O == O) {                                  /* keep the tighter bounds */
            if (e->lower > lower) lower = e->lower;
            if (e->upper < upper) upper = e->upper;
        }
        e->P = P; e->O = O; e->lower = (int8_t)lower; e->upper = (int8_t)upper; e->move = (int8_t)move;
    }
    return best;
}

static int solve(u64 P, u64 O, int alpha, int beta, int passed) {
    if (64 - popcount(P | O) <= SHALLOW) return solve_shallow(P, O, alpha, beta, passed);
    g_nodes++;
    u64 moves = eg_moves(P, O);
    if (!moves) {
        if (passed || !eg_moves(O, P)) return final_score(P, O);
        return -solve(O, P, -beta, -alpha, 1);
    }
    if (alpha >= STAB_ALPHA) {
        int bound = 64 - 2 * popcount(stable_discs(P, O));          /* P's best possible score */
        if (bound <= alpha) return bound;
    }
    return search_tt(P, O, moves, alpha, beta, 0, 1);
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
    else if (64 - popcount(P | O) <= SHALLOW) {                        /* root needs its best move */
        score = search_moves(P, O, moves, alpha, beta, best_move, -1);
    } else score = search_tt(P, O, moves, alpha, beta, best_move, 0);
    if (nodes) *nodes = g_nodes;
    return score;
}
