"""
第三小问（选做）：Minimax + Alpha-Beta，置换表缓存。
"""

import math

from dlgo.gotypes import Player, Point
from dlgo.goboard import GameState, Move
from dlgo.scoring import compute_game_result

__all__ = ["MinimaxAgent", "GameResultCache"]


class GameResultCache:
    """
    置换表缓存。

    zobrist_hash 参数可为 int，或 (board_hash, next_player) 元组以区分同形不同行棋方。
    """

    def __init__(self):
        self.cache = {}

    def get(self, zobrist_hash):
        return self.cache.get(zobrist_hash)

    def put(self, zobrist_hash, depth, value, flag="exact"):
        prev = self.cache.get(zobrist_hash)
        if prev is None or prev[0] <= depth:
            self.cache[zobrist_hash] = (depth, value, flag)


def _non_resign_moves(game_state: GameState):
    moves = game_state.legal_moves()
    return [m for m in moves if not m.is_resign] or moves


def _move_order_key(game_state: GameState, move: Move):
    if not move.is_play:
        return (0, 0)
    board = game_state.board
    me = game_state.next_player
    pt = move.point
    captures = 0
    for nb in pt.neighbors():
        if not board.is_on_grid(nb):
            continue
        s = board.get_go_string(nb)
        if s is not None and s.color != me and s.num_liberties == 1:
            captures += len(s.stones)
    threats = sum(
        1
        for nb in pt.neighbors()
        if board.is_on_grid(nb) and board.get(nb) == me
    )
    return (-captures, -threats)


class MinimaxAgent:
    def __init__(self, max_depth=3, evaluator=None, use_cache=True):
        self.max_depth = max_depth
        self.evaluator = evaluator or self._default_evaluator
        self.use_cache = use_cache
        self._cache = GameResultCache()
        self._root_player = Player.black

    def select_move(self, game_state: GameState) -> Move:
        self._root_player = game_state.next_player
        moves = _non_resign_moves(game_state)
        if len(moves) == 1:
            return moves[0]

        ordered = sorted(moves, key=lambda m: _move_order_key(game_state, m))
        best_move = ordered[0]
        best_val = -math.inf

        for move in ordered:
            child = game_state.apply_move(move)
            val = self.alphabeta(
                child,
                self.max_depth - 1,
                -math.inf,
                math.inf,
            )
            if val > best_val:
                best_val = val
                best_move = move
        return best_move

    def minimax(self, game_state, depth, maximizing_player):
        if game_state.is_over():
            return self._terminal_score_for_root(game_state)
        if depth == 0:
            return self._eval_for_root(game_state)

        moves = _non_resign_moves(game_state)
        if maximizing_player:
            v = -math.inf
            for m in moves:
                v = max(
                    v,
                    self.minimax(
                        game_state.apply_move(m),
                        depth - 1,
                        False,
                    ),
                )
            return v
        v = math.inf
        for m in moves:
            v = min(
                v,
                self.minimax(
                    game_state.apply_move(m),
                    depth - 1,
                    True,
                ),
            )
        return v

    def alphabeta(self, game_state, depth, alpha, beta):
        maximizing = game_state.next_player == self._root_player

        if game_state.is_over():
            return self._terminal_score_for_root(game_state)
        if depth == 0:
            return self._eval_for_root(game_state)

        key = None
        if self.use_cache:
            key = (
                game_state.board.zobrist_hash(),
                game_state.next_player,
            )
            hit = self._cache.get(key)
            if hit is not None:
                d_stored, val, _flag = hit
                if d_stored >= depth:
                    return val

        moves = sorted(
            _non_resign_moves(game_state),
            key=lambda m: _move_order_key(game_state, m),
        )

        if maximizing:
            value = -math.inf
            for m in moves:
                value = max(
                    value,
                    self.alphabeta(game_state.apply_move(m), depth - 1, alpha, beta),
                )
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
        else:
            value = math.inf
            for m in moves:
                value = min(
                    value,
                    self.alphabeta(game_state.apply_move(m), depth - 1, alpha, beta),
                )
                beta = min(beta, value)
                if alpha >= beta:
                    break

        if self.use_cache and key is not None:
            self._cache.put(key, depth, value, "exact")
        return value

    def _terminal_score_for_root(self, game_state: GameState) -> float:
        w = game_state.winner()
        if w is None:
            return 0.0
        return 1.0 if w == self._root_player else -1.0

    def _eval_for_root(self, game_state: GameState) -> float:
        raw = self.evaluator(game_state)
        if game_state.next_player == self._root_player:
            return raw
        return -raw

    def _default_evaluator(self, game_state: GameState):
        """
        对「当前行棋方」有利的正值：子数差 + 气差 + 领土启发。
        """
        board = game_state.board
        me = game_state.next_player
        opp = me.other
        my_stones = opp_stones = 0
        my_lib = opp_lib = 0
        seen = set()
        for r in range(1, board.num_rows + 1):
            for c in range(1, board.num_cols + 1):
                p = Point(r, c)
                col = board.get(p)
                if col == me:
                    my_stones += 1
                    s = board.get_go_string(p)
                    if s is not None and id(s) not in seen:
                        seen.add(id(s))
                        my_lib += s.num_liberties
                elif col == opp:
                    opp_stones += 1
                    s = board.get_go_string(p)
                    if s is not None and id(s) not in seen:
                        seen.add(id(s))
                        opp_lib += s.num_liberties

        stone = (my_stones - opp_stones) * 3.0
        lib = (my_lib - opp_lib) * 0.8

        try:
            gr = compute_game_result(game_state)
            b = gr.b
            w = gr.w + gr.komi
            if me == Player.black:
                terr = b - w
            else:
                terr = w - b
        except Exception:
            terr = 0.0

        return stone + lib + terr * 0.15

    def _get_ordered_moves(self, game_state):
        moves = _non_resign_moves(game_state)
        return sorted(moves, key=lambda m: _move_order_key(game_state, m))
