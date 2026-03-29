"""
围棋人机对弈图形界面（Tkinter）。

运行：python gui_go.py
      python gui_go.py --size 7 --agent random    # 可选初始白方：mcts / minimax / random
"""

from __future__ import annotations

import argparse
import tkinter as tk
from tkinter import messagebox

from dlgo import GameState, Player, Point
from dlgo.goboard import Move

from agents.mcts_agent import MCTSAgent
from agents.random_agent import RandomAgent


def _load_minimax():
    try:
        from agents.minimax_agent import MinimaxAgent
        return MinimaxAgent
    except ImportError:
        return None


class GoGUI:
    def __init__(self, root, board_size=5, initial_agent: str = "mcts"):
        self.root = root
        self.board_size = board_size
        self.cell = 44
        self.margin = 36
        self.plies = []
        self._minimax_cls = _load_minimax()

        self.agent_mode = tk.StringVar(value=initial_agent)
        if initial_agent == "minimax" and self._minimax_cls is None:
            self.agent_mode.set("mcts")

        w = self.margin * 2 + self.cell * (board_size - 1)
        h = self.margin * 2 + self.cell * (board_size - 1) + 80

        self._update_window_title()

        ai_bar = tk.Frame(root)
        ai_bar.pack(fill=tk.X, padx=8, pady=(6, 0))
        tk.Label(ai_bar, text="白方 AI：").pack(side=tk.LEFT)
        tk.Radiobutton(
            ai_bar,
            text="MCTS",
            variable=self.agent_mode,
            value="mcts",
            command=self._on_agent_mode_change,
        ).pack(side=tk.LEFT, padx=(4, 0))
        tk.Radiobutton(
            ai_bar,
            text="Minimax",
            variable=self.agent_mode,
            value="minimax",
            command=self._on_agent_mode_change,
        ).pack(side=tk.LEFT, padx=(8, 0))
        tk.Radiobutton(
            ai_bar,
            text="随机",
            variable=self.agent_mode,
            value="random",
            command=self._on_agent_mode_change,
        ).pack(side=tk.LEFT, padx=(8, 0))

        self.canvas = tk.Canvas(root, width=w, height=h, bg="#dcb35c", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        bar = tk.Frame(root)
        bar.pack(fill=tk.X)
        tk.Button(bar, text="新对局", command=self.new_game).pack(side=tk.LEFT, padx=4, pady=4)
        tk.Button(bar, text="停一手", command=self.human_pass).pack(side=tk.LEFT, padx=4, pady=4)
        tk.Button(bar, text="悔棋", command=self.undo).pack(side=tk.LEFT, padx=4, pady=4)

        self.status = tk.Label(root, text="", anchor="w")
        self.status.pack(fill=tk.X, padx=8, pady=(0, 6))

        self._rebuild_ai()

        self.human_player = Player.black
        self.ai_player = Player.white

        self.canvas.bind("<Button-1>", self.on_click)
        self._dialog_shown = False
        self.new_game()

    def _update_window_title(self):
        mode = self.agent_mode.get()
        label = {"mcts": "MCTS", "minimax": "Minimax", "random": "随机 AI"}.get(
            mode, mode.upper()
        )
        self.root.title(f"围棋 {self.board_size}x{self.board_size} — 白方 {label}")

    def _rebuild_ai(self):
        mode = self.agent_mode.get()
        if mode == "random":
            self.ai = RandomAgent()
        elif mode == "minimax" and self._minimax_cls is not None:
            self.ai = self._minimax_cls(max_depth=3)
        else:
            if mode == "minimax":
                self.agent_mode.set("mcts")
            self.ai = MCTSAgent(num_rounds=2500, time_limit=8.0)
        self._update_window_title()

    def _on_agent_mode_change(self):
        self._rebuild_ai()

    def new_game(self):
        self.plies = []
        self._dialog_shown = False
        self._rebuild_from_plies()
        self.draw_board()
        self.update_status()
        self.root.after(80, self.maybe_ai_move)

    def _rebuild_from_plies(self):
        self.game = GameState.new_game(self.board_size)
        for m in self.plies:
            self.game = self.game.apply_move(m)

    def undo(self):
        if not self.plies:
            return
        self.plies.pop()
        self._dialog_shown = False
        self._rebuild_from_plies()
        self.draw_board()
        self.update_status()

    def human_pass(self):
        if self.game.is_over():
            return
        if self.game.next_player != self.human_player:
            return
        self.game = self.game.apply_move(Move.pass_turn())
        self.plies.append(Move.pass_turn())
        self.draw_board()
        self.update_status()
        self.root.after(80, self.maybe_ai_move)

    def on_click(self, event):
        if self.game.is_over():
            return
        if self.game.next_player != self.human_player:
            return
        pt = self.pixel_to_point(event.x, event.y)
        if pt is None:
            return
        move = Move.play(pt)
        if not self.game.is_valid_move(move):
            return
        self.game = self.game.apply_move(move)
        self.plies.append(move)
        self.draw_board()
        self.update_status()
        if self.game.is_over():
            self.end_dialog()
        else:
            self.root.after(80, self.maybe_ai_move)

    def maybe_ai_move(self):
        if self.game.is_over():
            self.end_dialog()
            return
        if self.game.next_player != self.ai_player:
            return
        self.root.config(cursor="watch")
        self.root.update_idletasks()
        try:
            mv = self.ai.select_move(self.game)
        finally:
            self.root.config(cursor="")
        self.game = self.game.apply_move(mv)
        self.plies.append(mv)
        self.draw_board()
        self.update_status()
        if self.game.is_over():
            self.end_dialog()

    def end_dialog(self):
        if self._dialog_shown:
            return
        self._dialog_shown = True
        w = self.game.winner()
        if w is None:
            msg = "对局结束：平局"
        else:
            msg = f"对局结束：{w.name} 胜"
        messagebox.showinfo("终局", msg)

    def update_status(self):
        if self.game.is_over():
            w = self.game.winner()
            extra = f" 胜者: {w.name}" if w else " 平局"
            self.status.config(text=f"终局.{extra}")
            return
        turn = self.game.next_player.name
        lm = self.game.last_move
        last = str(lm) if lm else "-"
        self.status.config(text=f"当前: {turn}  |  上一手: {last}")

    def point_to_pixel(self, p: Point):
        x = self.margin + (p.col - 1) * self.cell
        y = self.margin + (p.row - 1) * self.cell
        return x, y

    def pixel_to_point(self, x, y):
        for r in range(1, self.board_size + 1):
            for c in range(1, self.board_size + 1):
                cx, cy = self.point_to_pixel(Point(r, c))
                if (x - cx) ** 2 + (y - cy) ** 2 <= (self.cell * 0.35) ** 2:
                    return Point(r, c)
        return None

    def draw_board(self):
        self.canvas.delete("all")
        n = self.board_size
        for i in range(n):
            x0 = self.margin + i * self.cell
            y0 = self.margin
            x1 = x0
            y1 = self.margin + (n - 1) * self.cell
            self.canvas.create_line(x0, y0, x1, y1, width=1, fill="#333")
            y0, y1 = self.margin, self.margin + (n - 1) * self.cell
            x0 = self.margin
            x1 = self.margin + (n - 1) * self.cell
            yy = self.margin + i * self.cell
            self.canvas.create_line(x0, yy, x1, yy, width=1, fill="#333")

        board = self.game.board
        for r in range(1, n + 1):
            for c in range(1, n + 1):
                p = Point(r, c)
                cx, cy = self.point_to_pixel(p)
                stone = board.get(p)
                rad = self.cell * 0.42
                if stone == Player.black:
                    self.canvas.create_oval(
                        cx - rad,
                        cy - rad,
                        cx + rad,
                        cy + rad,
                        fill="#111",
                        outline="#222",
                    )
                elif stone == Player.white:
                    self.canvas.create_oval(
                        cx - rad,
                        cy - rad,
                        cx + rad,
                        cy + rad,
                        fill="#f7f7f7",
                        outline="#999",
                    )


def main():
    parser = argparse.ArgumentParser(
        description="围棋人机对弈 (Tkinter)，界面内选择白方 MCTS / Minimax / 随机"
    )
    parser.add_argument("--size", type=int, default=5, help="棋盘边长")
    parser.add_argument(
        "--agent",
        choices=["mcts", "minimax", "random"],
        default="mcts",
        help="启动时默认选中的白方 AI（可在窗口内随时切换）",
    )
    args = parser.parse_args()

    root = tk.Tk()
    GoGUI(root, board_size=args.size, initial_agent=args.agent)
    root.mainloop()


if __name__ == "__main__":
    main()
