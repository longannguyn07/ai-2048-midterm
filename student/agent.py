"""Điểm bắt đầu của bài nộp — sinh viên chỉ cần sửa file này.

Baseline bên dưới nhìn trước đúng một nước. Nó minh họa API và cách mô phỏng
nước đi, nhưng cố ý chưa xử lý ô ngẫu nhiên và chưa phải một thuật toán mạnh.
"""

from __future__ import annotations

import numpy as np

from game2048.observation import Observation
from game2048.types import Move
from student.heuristic import evaluate_board
from student.search import shallow_search_best_move


class Agent:
    # Đổi tên này thành tên đội; nó sẽ xuất hiện trong UI và file kết quả.
    name = "heuristic-2048"

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

        return shallow_search_best_move(obs.board, obs.legal_moves, depth=2, heuristic_fn=evaluate_board)
