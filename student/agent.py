"""Điểm bắt đầu của bài nộp — sinh viên chỉ cần sửa file này.

Baseline bên dưới nhìn trước đúng một nước. Nó minh họa API và cách mô phỏng
nước đi, nhưng cố ý chưa xử lý ô ngẫu nhiên và chưa phải một thuật toán mạnh.
"""

from __future__ import annotations

import time

import numpy as np

import student.search as search_module
from game2048.observation import Observation
from game2048.rules import move_board
from game2048.types import Move
from student.heuristic import evaluate_board
from student.search import shallow_search_best_move


class Agent:
    # Đổi tên này thành tên đội; nó sẽ xuất hiện trong UI và file kết quả.
    name = "heuristic-2048"

    def __init__(self):
        self.last_nodes = 0
        self.last_completed_depth = 0

    def reset(self, seed: int | None = None) -> None:
        """Xóa cache/trạng thái giữa hai ván nếu thuật toán của bạn có dùng."""

    @staticmethod
    def evaluate(board: np.ndarray, merge_score: int) -> float:
        """Để tương thích với starter API, wrapper lại heuristic chính."""
        return evaluate_board(board, merge_score)

    def choose_move(self, obs: Observation, time_limit_ms: float) -> Move:
        """Trả về đúng một phần tử ``Move`` có trong ``obs.legal_moves``."""
        if not obs.legal_moves:
            raise ValueError("choose_move được gọi ở trạng thái kết thúc")

        board = obs.board
        empty_count = int(np.count_nonzero(board == 0))
        max_tile = int(board.max())

        if empty_count >= 14:
            phase = "early"
        elif empty_count >= 7:
            phase = "mid"
        else:
            phase = "late"

        if phase == "early":
            depth = 4
            move_budget = min(time_limit_ms * 0.75, 40.0)
        elif phase == "mid":
            depth = 5
            move_budget = min(time_limit_ms * 0.70, 32.0)
        else:
            depth = 6
            move_budget = min(time_limit_ms * 0.68, 28.0)

        if max_tile >= 1024:
            depth += 1
            move_budget = min(time_limit_ms * 0.78, 36.0)
        if max_tile >= 2048:
            depth += 1
            move_budget = min(time_limit_ms * 0.82, 40.0)

        if empty_count <= 4:
            depth += 1
            move_budget = min(time_limit_ms * 0.72, 30.0)

        if obs.move_count <= 2:
            depth = 3
            move_budget = min(move_budget, 10.0)

        start = time.time()
        try:
            move = shallow_search_best_move(
                board,
                obs.legal_moves,
                depth=depth,
                heuristic_fn=evaluate_board,
                time_limit_ms=move_budget,
            )
        except Exception:
            move = obs.legal_moves[0]
        finally:
            self.last_nodes = search_module.LAST_NODES
            self.last_completed_depth = search_module.LAST_COMPLETED_DEPTH

        elapsed_ms = (time.time() - start) * 1000.0
        if elapsed_ms > time_limit_ms * 0.95:
            best_move = None
            best_score = float("-inf")
            for candidate in obs.legal_moves:
                next_board, merge_score, changed = move_board(board, candidate)
                if not changed:
                    continue
                value = evaluate_board(next_board, merge_score)
                if value > best_score:
                    best_score = value
                    best_move = candidate
            if best_move is not None:
                return best_move

        return move
