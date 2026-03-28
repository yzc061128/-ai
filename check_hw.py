#!/usr/bin/env python3
"""
作业快速自检（围棋）：dlgo、随机 / MCTS / Minimax 可导入并可落子。
用法：python check_hw.py
"""

import sys


def main() -> int:
    print("1) dlgo GameState …")
    from dlgo import GameState

    g = GameState.new_game(5)
    assert g.board.num_rows == 5
    print("   OK")

    print("2) RandomAgent …")
    from agents.random_agent import RandomAgent

    m = RandomAgent().select_move(g)
    assert g.is_valid_move(m)
    print("   OK:", m)

    print("3) MCTSAgent …")
    from agents.mcts_agent import MCTSAgent

    m = MCTSAgent(num_rounds=50, time_limit=2.0).select_move(g)
    assert g.is_valid_move(m)
    print("   OK:", m)

    print("4) MinimaxAgent …")
    from agents.minimax_agent import GameResultCache, MinimaxAgent

    c = GameResultCache()
    c.put((0, g.next_player), 3, 0.5, "exact")
    assert c.get((0, g.next_player)) is not None
    m = MinimaxAgent(max_depth=2).select_move(g)
    assert g.is_valid_move(m)
    print("   OK:", m)

    print("全部通过。")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print("失败:", e, file=sys.stderr)
        raise SystemExit(1)
