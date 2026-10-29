"""Điểm bắt đầu của bài nộp — sinh viên chỉ cần sửa file này.

Baseline bên dưới nhìn trước đúng một nước. Nó minh họa API và cách mô phỏng
nước đi, nhưng cố ý chưa xử lý ô ngẫu nhiên và chưa phải một thuật toán mạnh.
"""

from __future__ import annotations

import time

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

        # Chia thời gian hợp lý theo số ô trống: giai đoạn đầu cần nhanh, cần tìm trước khi bàn đầy.
        empty_count = int(np.count_nonzero(obs.board == 0))
        max_tile = int(obs.board.max())

        if empty_count >= 14:
            depth = 5
            move_budget = min(time_limit_ms * 0.55, 35.0)
        elif empty_count >= 8:
            depth = 4
            move_budget = min(time_limit_ms * 0.45, 28.0)
        elif empty_count >= 4:
            depth = 4
            move_budget = min(time_limit_ms * 0.35, 22.0)
        else:
            depth = 5
            move_budget = min(time_limit_ms * 0.30, 18.0)

        if max_tile >= 1024:
            depth += 1
            move_budget = min(time_limit_ms * 0.6, 32.0)

        # Không dùng hết khoản thời gian ở nước đầu, để có thời gian cho phần sau của ván.
        if obs.move_count <= 2:
            depth = 3
            move_budget = min(move_budget, 8.0)

        start = time.time()
        move = shallow_search_best_move(
            obs.board,
            obs.legal_moves,
            depth=depth,
            heuristic_fn=evaluate_board,
            time_limit_ms=move_budget,
        )

        elapsed_ms = (time.time() - start) * 1000.0
        if elapsed_ms > time_limit_ms * 0.95:
            # Nếu gần hết budget, fallback to safest legal move by heuristic instead of risking timeout.
            best_move = None
            best_score = float("-inf")
            for candidate in obs.legal_moves:
                from game2048.rules import move_board
                next_board, merge_score, changed = move_board(obs.board, candidate)
                if not changed:
                    continue
                value = evaluate_board(next_board, merge_score)
                if value > best_score:
                    best_score = value
                    best_move = candidate
            if best_move is not None:
                return best_move

        return move
