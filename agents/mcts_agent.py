"""
MCTS (蒙特卡洛树搜索) 智能体。

标准 MCTS：选择、扩展、模拟、反向传播；含启发式走子与深度限制等优化。
"""

import math
import random
import time
from typing import Optional

from dlgo.gotypes import Player
from dlgo.goboard import GameState, Move
from dlgo.scoring import compute_game_result

__all__ = ["MCTSAgent", "MCTSNode"]


def _moves_for_mcts(game_state: GameState):
    """MCTS 中不使用认输（除非仅剩认输），减少无意义分支。"""
    moves = game_state.legal_moves()
    non_resign = [m for m in moves if not m.is_resign]
    return non_resign if non_resign else moves


class MCTSNode:
    """MCTS 树节点。"""

    def __init__(self, game_state, parent=None, prior=1.0, move: Optional[Move] = None):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.children = []
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self._unexpanded = None

    @property
    def value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_leaf(self):
        return len(self.children) == 0

    def is_terminal(self):
        return self.game_state.is_over()

    def is_fully_expanded(self):
        if self.is_terminal():
            return True
        if self._unexpanded is None:
            return False
        return len(self._unexpanded) == 0

    def best_child(self, c=1.414):
        """
        UCT 选择：从父节点（当前节点）视角最大化
        exploit = 1 - child.value（子节点统计为「下一手方」视角）
        exploration = c * prior * sqrt(ln(parent_visits) / child_visits)
        """
        parent_visits = max(1, self.visit_count)
        best_score = float("-inf")
        best = None
        for child in self.children:
            if child.visit_count == 0:
                uct = float("inf")
            else:
                exploit = 1.0 - child.value
                exploration = (
                    c
                    * child.prior
                    * math.sqrt(math.log(parent_visits) / child.visit_count)
                )
                uct = exploit + exploration
            if uct > best_score:
                best_score = uct
                best = child
        return best

    def expand(self):
        if self._unexpanded is None:
            moves = _moves_for_mcts(self.game_state)
            random.shuffle(moves)
            self._unexpanded = moves
        if not self._unexpanded:
            return self
        move = self._unexpanded.pop()
        child_state = self.game_state.apply_move(move)
        prior = _move_prior(self.game_state, move)
        child = MCTSNode(child_state, parent=self, prior=prior, move=move)
        self.children.append(child)
        return child

    def backup(self, value):
        """value 为当前节点 game_state.next_player 视角的得分（胜 1 / 负 0 / 和 0.5）。"""
        node = self
        v = float(value)
        while node is not None:
            node.visit_count += 1
            node.value_sum += v
            v = 1.0 - v
            node = node.parent


def _terminal_playout_value(game_state: GameState) -> float:
    if not game_state.is_over():
        return 0.5
    if game_state.last_move is not None and game_state.last_move.is_resign:
        return 1.0
    w = game_state.winner()
    if w is None:
        return 0.5
    p = game_state.next_player
    return 1.0 if w == p else 0.0


def _move_prior(game_state: GameState, move: Move) -> float:
    """简单先验：提子、邻己方、靠角略高；pass 较低。"""
    if move.is_pass or move.is_resign:
        return 0.3
    board = game_state.board
    me = game_state.next_player
    pt = move.point
    cap = 0
    for nb in pt.neighbors():
        if not board.is_on_grid(nb):
            continue
        s = board.get_go_string(nb)
        if s is not None and s.color != me and s.num_liberties == 1:
            cap += len(s.stones)
    adj_me = 0
    for nb in pt.neighbors():
        if board.is_on_grid(nb) and board.get(nb) == me:
            adj_me += 1
    edge = 0
    if pt.row == 1 or pt.row == board.num_rows:
        edge += 1
    if pt.col == 1 or pt.col == board.num_cols:
        edge += 1
    return 0.5 + 0.12 * min(cap, 4) + 0.06 * adj_me - 0.03 * edge


def _heuristic_pick_move(game_state: GameState, moves: list) -> Move:
    """启发式走子：偏好吃子、连块，低概率 pass。"""
    play_moves = [m for m in moves if m.is_play]
    if not play_moves:
        return random.choice(moves)
    board = game_state.board
    me = game_state.next_player
    scored = []
    for m in play_moves:
        pt = m.point
        cap = 0
        for nb in pt.neighbors():
            if not board.is_on_grid(nb):
                continue
            s = board.get_go_string(nb)
            if s is not None and s.color != me and s.num_liberties == 1:
                cap += len(s.stones)
        adj_me = sum(
            1
            for nb in pt.neighbors()
            if board.is_on_grid(nb) and board.get(nb) == me
        )
        scored.append((cap * 3 + adj_me + random.random() * 0.01, m))
    scored.sort(key=lambda x: -x[0])
    top = [m for _, m in scored[: max(3, len(scored) // 4)]]
    return random.choice(top)


class MCTSAgent:
    """
    MCTS 智能体。

    优化：
    1. 启发式 rollout（偏好吃子/联络）
    2. 限制模拟深度，截断后用局面评估（子数 + 气数 + 快速地盘）
    """

    def __init__(
        self,
        num_rounds=2000,
        temperature=1.0,
        time_limit=15.0,
        max_rollout_depth=28,
        exploration_constant=1.414,
    ):
        self.num_rounds = num_rounds
        self.temperature = temperature
        self.time_limit = time_limit
        self.max_rollout_depth = max_rollout_depth
        self.exploration_constant = exploration_constant

    def select_move(self, game_state: GameState) -> Move:
        moves = _moves_for_mcts(game_state)
        if len(moves) == 1:
            return moves[0]

        root = MCTSNode(game_state)
        deadline = time.perf_counter() + self.time_limit
        rounds = 0

        while rounds < self.num_rounds and time.perf_counter() < deadline:
            node = root
            while not node.is_terminal() and node.is_fully_expanded():
                node = node.best_child(self.exploration_constant)

            if node.is_terminal():
                val = _terminal_playout_value(node.game_state)
                node.backup(val)
            else:
                child = node.expand()
                val = self._simulate(child.game_state)
                child.backup(val)
            rounds += 1

        return self._select_best_move(root)

    def _simulate(self, game_state: GameState) -> float:
        """
        随机模拟 + 启发式选子 + 深度上限后启发式估值（视为「和棋倾向」的终局近似）。
        """
        state = game_state
        depth = 0
        while not state.is_over() and depth < self.max_rollout_depth:
            moves = _moves_for_mcts(state)
            if not moves:
                break
            if random.random() < 0.85:
                move = _heuristic_pick_move(state, moves)
            else:
                move = random.choice(moves)
            state = state.apply_move(move)
            depth += 1

        if state.is_over():
            return _terminal_playout_value(state)

        return self._rollout_eval(state)

    def _rollout_eval(self, game_state: GameState) -> float:
        """截断局面：用领土 + 子力粗略估计，映射到 [0,1]（对 next_player）。"""
        p = game_state.next_player
        try:
            gr = compute_game_result(game_state)
            b = gr.b
            w = gr.w + gr.komi
            if p == Player.black:
                diff = b - w
            else:
                diff = w - b
        except Exception:
            diff = 0.0
        scale = 8.0
        x = max(-1.0, min(1.0, diff / scale))
        return 0.5 + 0.5 * x

    def _select_best_move(self, root: MCTSNode) -> Move:
        if not root.children:
            return random.choice(_moves_for_mcts(root.game_state))
        best = max(root.children, key=lambda ch: ch.visit_count)
        if best.move is not None:
            return best.move
        return random.choice(_moves_for_mcts(root.game_state))
